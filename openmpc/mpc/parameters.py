import control
import numpy as np
from openmpc.invariant_sets import Polytope, invariant_set
from openmpc.support.constraints import Constraint, TimedConstraint
from openmpc.models import LinearSystem, NonlinearSystem


class MPCProblem:

    """
    Base class to define all the parameters for the MPC controller.
    """

    def __init__(self, system               : LinearSystem| NonlinearSystem , 
                       horizon              : int,
                       Q                    : np.ndarray,
                       R                    : np.ndarray, 
                       QT                   : np.ndarray | None = None,
                       global_penalty_weight: float             = 1.0, 
                       solver               : str |None         = None, 
                       slack_penalty        : str               = "SQUARE"):

        """
        Initializes the MPCParameters with the given parameters.

        :param system: The state-space model of the system. The state space model should be described in desceret time.
        :type system: LinearSystem| NonlinearSystem
        :param horizon: The prediction horizon for the MPC.
        :type horizon: int
        :param Q: The state weighting matrix.
        :type Q: numpy.ndarray
        :param R: The input weighting matrix.
        :type R: numpy.ndarray
        :param QT: The terminal state weighting matrix.
        :type QT: numpy.ndarray | None
        :param global_penalty_weight: The global penalty weight for the cost function.
        :type global_penalty_weight: float
        :param solver: The solver to be used for optimization.
        :type solver: str |None
        :param slack_penalty: The type of penalty for slack variables, default is 'SQUARE'.
        :type slack_penalty: str
        :param u_constraints: List of input constraints.
        :type u_constraints: list[Constraint]
        :param x_constraints: List of state constraints.
        :type x_constraints: list[Constraint]
        :param y_constraints: List of output constraints.
        :type y_constraints: list[Constraint]
        :param terminal_constraints: List of terminal constraints.
        :type terminal_constraints: list[Constraint]
        :param terminal_set: The terminal set for the MPC.
        :type terminal_set: Polytope | None
        :param dual_mode_controller: The dual mode controller for the MPC.
        :type dual_mode_controller: numpy.ndarray
        :param dual_mode_horizon: The horizon for the dual mode.
        :type dual_mode_horizon: int
        :param reference_controller: The reference controller for the MPC.
        :type reference_controller: numpy.ndarray
        :param soft_tracking: Whether soft tracking is enabled.
        :type soft_tracking: bool
        :param tracking_penalty_weight: The penalty weight for soft tracking.
        :type tracking_penalty_weight: float
        :param reference_reached_at_steady_state: Whether the reference is reached at steady state.
        :type reference_reached_at_steady_state: bool
        :param dt: The time step for the system.
        :type dt: float
        """
        
        # Store the state-space model
        self.system = system
        self.dt     = system.dt

        # MPC parameters
        self.horizon = horizon
        self.Q       = Q
        self.R       = R
        self.QT      = np.zeros((system.size_state, system.size_state)) if QT is None else QT

        self.global_penalty_weight = global_penalty_weight
        self.solver                = solver #! we need to check which solvers are available

        self.slack_penalty         = slack_penalty  # Changed from slack_norm to slack_penalty

        # Constraints as lists of `Constraint` objects
        self.u_constraints          : list[Constraint] = []
        self.x_constraints          : list[Constraint] = []
        self.y_constraints          : list[Constraint] = []
        self.terminal_constraints   : list[Constraint] = []
        self.terminal_set           : Constraint | None  = None

        # Dual mode parameters
        self.dual_mode_controller : np.ndarray = np.zeros((self.system.size_input, self.system.size_state))
        self.dual_mode_horizon    : int        = 0
        self.reference_controller : np.ndarray = np.zeros((self.system.size_input, self.system.size_state)) # This is the controller that creates the reference input to the system as U_ref = -L_ref*X


        # Tracking parameters
        self.soft_tracking           = False
        self.tracking_penalty_weight = 1.  # Default penalty weight for soft tracking
        self.reference_reached_at_steady_state = True
        
        # Advanced
        self.state_time_constraints : list[TimedConstraint] = []
    
    def add_input_magnitude_constraint(self, limit :float , input_index : int | None = None, is_hard : bool = True, penalty_weight : float =1.):
        """
        Add input magnitude constraint: -limit <= u <= limit.
        
        :param limit: The magnitude limit for the input.
        :type limit: float
        :param input_index: The index of the input to apply the constraint to. If None, apply to all inputs.
        :type input_index: int | None
        :param is_hard: Whether the constraint is hard or soft.
        :type is_hard: bool
        :param penalty_weight: The penalty weight for soft constraints.
        :type penalty_weight: float
        """

        limit = float(limit)
        input_index = int(input_index) if input_index is not None else None
        penalty_weight = float(penalty_weight)

        if penalty_weight < 0:
            raise ValueError("penalty_weight must be non-negative.")


        if input_index is None:
            A = np.vstack([np.eye(self.system.size_input), -np.eye(self.system.size_input)])
            b = np.array([limit] * self.system.size_input * 2)
        else:
            A = np.array([[1 if i == input_index else 0 for i in range(self.system.size_input)]])
            A = np.vstack([A, -A])
            b = np.array([limit, limit])
        
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.u_constraints.append(constraint)

    def add_input_bound_constraint(self, limits : tuple | float, input_index : int | None = None, is_hard : bool =True, penalty_weight : float = 1.):
        """
        Adds input bounds constraints. The constraints can be symmetric or asymmetric based on `limits`.

        :param limits: If a single float, the bounds are symmetric: -limits <= u_t <= limits.
                          If a tuple (lb, ub), the bounds are asymmetric: lb <= u_t <= ub.
        :type limits: float or tuple
        :param input_index: If None, apply the constraints to all inputs uniformly.
                           If an int, apply the constraint to a specific input.
        :type input_index: int | None
        :param is_hard: Whether the constraint is hard or soft.
        :type is_hard: bool
        :param penalty_weight: The penalty weight for soft constraints.
        :type penalty_weight: float
        """

        # Determine the bounds
        if isinstance(limits, (int, float)):
            lb, ub = -limits, limits  # Symmetric bounds
        elif isinstance(limits, tuple) and len(limits) == 2:
            lb, ub = limits  # Asymmetric bounds
        else:
            raise ValueError("limits must be a single value or a tuple of two values (lb, ub).")

        if input_index is None:
            # Apply uniformly to all inputs
            n_inputs = self.system.size_input
            A = np.vstack([np.eye(n_inputs), -np.eye(n_inputs)])  # Identity and negative identity for constraints
            b = np.hstack([ub * np.ones(n_inputs), -lb * np.ones(n_inputs)])
        else:
            # Apply to a specific input
            A = np.zeros((2, self.system.size_input))
            A[0, input_index] = 1  # Upper bound for the specified input
            A[1, input_index] = -1  # Lower bound for the specified input
            b = np.array([ub, -lb])

        # Create and store the constraint
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.u_constraints.append(constraint)
        
        
    def add_output_magnitude_constraint(self, limit :float , output_index : int | None = None, is_hard : bool = True, penalty_weight : float =1.):
        """Add magnitude constraint on a specific output: -limit <= y[output_index] <= limit."""
        
        if output_index is None:

            A = np.vstack([np.eye(self.system.size_output), -np.eye(self.system.size_output)])
            b = np.array([limit] * self.system.size_output * 2)
        else:
            A = np.array([[1 if i == output_index else 0 for i in range(self.system.size_output)]])
            A = np.vstack([A, -A])
            b = np.array([limit, limit])

        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.y_constraints.append(constraint)

    def add_output_bound_constraint(self, limits : tuple | float, output_index : int | None = None, is_hard : bool =True, penalty_weight : float = 1.):
        """
        Adds output bounds constraints. The constraints can be symmetric or asymmetric based on `limits`.

        :param limits: If a single float, the bounds are symmetric: -limits <= y_t <= limits.
                            If a tuple (lb, ub), the bounds are asymmetric: lb <= y_t <= ub.
        :param output_index: If None, apply the constraints to all outputs uniformly.
                            If an int, apply the constraint to a specific output.
        :type output_index: int | None
        :param is_hard: Whether the constraint is hard or soft.
        :type is_hard: bool
        :param penalty_weight: The penalty weight for soft constraints.
        :type penalty_weight: float
        """
        # Determine the bounds
        if isinstance(limits, (int, float)):
            lb, ub = -limits, limits  # Symmetric bounds
        elif isinstance(limits, tuple) and len(limits) == 2:
            lb, ub = limits  # Asymmetric bounds
        else:
            raise ValueError("limits must be a single value or a tuple of two values (lb, ub).")

        if output_index is None:
            # Apply uniformly to all outputs
            A = np.vstack([np.eye(self.system.size_output), -np.eye(self.system.size_output)])
            b = np.hstack([ub * np.ones(self.system.size_output), -lb * np.ones(self.system.size_output)])
        
        else:
            # Apply to a specific input
            A = np.zeros((2, self.system.size_output))
            A[0, output_index] = 1  # Upper bound for the specified input
            A[1, output_index] = -1  # Lower bound for the specified input
            b = np.array([ub, -lb])


        # Create and store the constraint
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.y_constraints.append(constraint)
        
        
    def add_state_magnitude_constraint(self, limit :float , state_index  : int | None = None, is_hard : bool = True, penalty_weight : float =1.):
        """Add state magnitude constraint: -limit <= x[state_index] <= limit."""
        
        if state_index is None:
            A = np.vstack([np.eye(self.system.size_state), -np.eye(self.system.size_state)])
            b = np.array([limit] * self.system.size_state * 2)
        else:
            A = np.array([[1 if i == state_index else 0 for i in range(self.system.size_state)]])
            A = np.vstack([A, -A])
            b = np.array([limit, limit])
        
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.x_constraints.append(constraint)


    def add_state_bound_constraint(self,  limits : tuple | float, state_index  : int | None = None, is_hard : bool =True, penalty_weight : float = 1.):
        """
        Adds state bounds constraints. The constraints can be symmetric or asymmetric based on `limits`.
        
        :param limits: If a single float, the bounds are symmetric: -limits <= x_t <= limits.
                            If a tuple (lb, ub), the bounds are asymmetric: lb <= x_t <= ub.
        :type limits: float or tuple
        :param state_index: If None, apply the constraints to all states uniformly.
                            If an int, apply the constraint to a specific state.
        :type state_index: int | None
        :param is_hard: Whether the constraint is hard or soft.
        :type is_hard: bool
        :param penalty_weight: The penalty weight for soft constraints.
        :type penalty_weight: float
        """
        # Determine the bounds
        if isinstance(limits, (int, float)):
            lb, ub = -limits, limits  # Symmetric bounds
        elif isinstance(limits, tuple) and len(limits) == 2:
            lb, ub = limits  # Asymmetric bounds
        else:
            raise ValueError("limits must be a single value or a tuple of two values (lb, ub).")

        if state_index is None:
            # Apply uniformly to all states
            n_states = self.system.size_state
            A = np.vstack([np.eye(n_states), -np.eye(n_states)])  # Identity and negative identity for constraints
            b = np.hstack([ub * np.ones(n_states), -lb * np.ones(n_states)])
        else:
            # Apply to a specific state
            A = np.zeros((2, self.system.size_state))  # Number of columns matches state dimension
            A[0, state_index] = 1  # Upper bound for the specified state
            A[1, state_index] = -1  # Lower bound for the specified state
            b = np.array([ub, -lb])

        # Create and store the constraint
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.x_constraints.append(constraint)
        
        
    def add_general_state_constraints(self, Hx : np.ndarray, bx : np.ndarray, is_hard : bool =True, penalty_weight : int=1):
        """Add general state constraints of the form Hx * x <= bx."""


        if Hx.shape[1] != self.system.size_state:
            raise ValueError("The number of columns in A must match the state dimension of the system.")
        

        constraint = Constraint(Hx, bx, is_hard=is_hard, penalty_weight=penalty_weight)
        self.x_constraints.append(constraint)

    def add_general_output_constraints(self, Hy : np.ndarray, by : np.ndarray, is_hard : bool =True, penalty_weight : int=1):
        """Add general output constraints of the form Hy * y <= by."""
        
        if Hy.shape[1] != self.system.size_output:
            raise ValueError("The number of columns in A must match the output dimension of the system.")
        
        constraint = Constraint(Hy, by, is_hard=is_hard, penalty_weight=penalty_weight)
        self.y_constraints.append(constraint)

    def add_general_input_constraints(self, Hu : np.ndarray, bu : np.ndarray, is_hard : bool =True, penalty_weight : int=1):
        """Add general input constraints of the form Hu * u <= bu."""
        
        if Hu.shape[1] != self.system.size_input:
            raise ValueError("The number of columns in A must match the input dimension of the system.")
        
        constraint = Constraint(Hu, bu, is_hard=is_hard, penalty_weight=penalty_weight)
        self.u_constraints.append(constraint)

    def add_general_state_time_constraints(self, Hx : np.ndarray, bx : np.ndarray, start_time:float, end_time:float,is_hard : bool =True, penalty_weight : int=1):
        """Add general time state constraints of the form A[x,t] <= bx."""
        
        if Hx.shape[1] != self.system.size_state+1:
            raise ValueError("The number of columns in A must match the state dimension of the system + 1 to account for the time dimension.")
        
        constraint = TimedConstraint(Hx, bx, start_time, end_time, is_hard=is_hard, penalty_weight=penalty_weight)
        self.state_time_constraints.append(constraint)

    
    def reach_refererence_at_steady_state(self, option :bool = True):
        """
        Set the option to reach the reference at steady state or let the reference be reached from any state.
        
        :param option: If True, the reference is reached at steady state. If False, the reference can be reached from any state.
        :type option: bool
        """

        self.reference_reached_at_steady_state = option

    
    def add_terminal_ingredients(self, controller : np.ndarray | None = None):
        """Add terminal ingredients using a controller (defaults to LQR)."""

        if isinstance(self.system, NonlinearSystem): #! feature not supported for nonlinear systems
            raise ValueError("Terminal ingredients are only supported for linear systems for now.")
        
        if controller is None:
            # Use LQR controller as the default
            controller, _, _ = control.dlqr(self.system.A, self.system.B, self.Q, self.R)

        # Compute terminal cost using the discrete Lyapunov equation
        A_cl = self.system.A - self.system.B @ controller
        P = control.dlyap(A_cl.T, self.Q + controller.T @ self.R @ controller)
        self.QT = P
    
        # Construct the terminal set using the constraints
        Ax_list, bx_list = [], []
        for constraint in self.x_constraints:
            A, b = constraint.to_polytope()
            Ax_list.append(A)
            bx_list.append(b)

        for constraint in self.y_constraints:
            A, b = constraint.to_polytope()
            Ax_list.append(A @ self.system.C)
            bx_list.append(b)

        for constraint in self.u_constraints:
            A, b = constraint.to_polytope()            
            Ax_list.append(A@controller)
            bx_list.append(b)

        Px                : Polytope   = Polytope(np.vstack(Ax_list), np.concatenate(bx_list))
        terminal_set      : Polytope   = invariant_set(A_cl, Px) # set computations
        self.terminal_set : Constraint = Constraint(terminal_set.A, terminal_set.b, is_hard=True)

                
    def add_dual_mode(self, horizon : int, controller : np.ndarray ) : 
        """ 
        Add a dual mode to the MPC with an optional controller and horizon.

        For a linear system you will typically put an LQR controller here.

        :param dual_mode_horizon: The horizon for the dual mode.
        :type dual_mode_horizon: int
        :param controller: The controller for the dual mode.
        :type controller: numpy.ndarray
        """
              
        # set the dual mode controller
        self.dual_mode_controller = controller
        # Set the dual mode horizon
        self.dual_mode_horizon = int(horizon)

        if np.all(controller.flatten() == 0):
            raise ValueError("The controller must be non-zero.")

        if self.dual_mode_controller.shape[0] != self.system.size_input or self.dual_mode_controller.shape[1] != self.system.size_state:
            raise ValueError(f"The controller shape must be {self.system.size_input,self.system.size_state}. Given {self.dual_mode_controller.shape}")
        if self.dual_mode_horizon < 0:
            raise ValueError("The horizon must be non-negative.")
    

    def add_reference_controller(self, controller : np.ndarray):
        """
        Add a reference controller to the MPC.

        :param controller: The reference controller for the MPC.
        :type controller: numpy.ndarray
        """
        self.reference_controller = controller

        if self.reference_controller.shape[0] != self.system.size_input or self.reference_controller.shape[1] != self.system.size_state:
            raise ValueError(f"The controller shape must be {self.system.size_input,self.system.size_state}. Given {self.reference_controller.shape}")

        
    def soften_tracking_constraint(self, penalty_weight : float = 1.):
        """Enable softening of the tracking constraint with the specified penalty weight."""

        if penalty_weight < 0.:
            raise ValueError("penalty_weight must be non-negative.")
        
        self.soft_tracking           = True
        self.tracking_penalty_weight = penalty_weight
        

    def check_offsetfree_conditions(self):
        """Check the conditions for offset-free MPC."""

        if not isinstance(self.system, LinearSystem): #! feature not supported for nonlinear systems
            raise ValueError("Offset-free MPC is only supported for linear systems for now.")


        n  = self.system.size_state
        p  = self.system.size_output
        nd = self.system.size_disturbance  # Number of disturbances

        # Condition 1: rank([A-I, Bd; C, Cd]) == n + nd
        A_I = self.system.A - np.eye(n)
        top_block_1 = np.hstack([A_I, self.system.Bd])
        bottom_block_1 = np.hstack([self.system.C, self.system.Cd])
        matrix_1 = np.vstack([top_block_1, bottom_block_1])
        rank_condition_1 = np.linalg.matrix_rank(matrix_1) == (n + nd)

        # Condition 2: rank([A-I, B; C, 0]) == n + p
        top_block_2 = np.hstack([A_I, self.system.B])
        bottom_block_2 = np.hstack([self.system.C, np.zeros((p, self.system.size_input))])
        matrix_2 = np.vstack([top_block_2, bottom_block_2])
        rank_condition_2 = np.linalg.matrix_rank(matrix_2) == (n + p)

        return rank_condition_1 and rank_condition_2
    
    
    def __str__(self):

        str_out = ""
        str_out += f"System: {self.system}\n"
        str_out += f"Time Step: {self.dt}\n"
        str_out += f"System type: {type(self.system)}\n"
        str_out += f"Horizon: {self.horizon}\n"
        str_out += f"Q: {self.Q}\n"
        str_out += f"R: {self.R}\n"
        str_out += f"QT: {self.QT}\n"
        str_out += f"Global Penalty Weight: {self.global_penalty_weight}\n"
        str_out += f"Solver: {self.solver}\n"
        str_out += f"Slack Penalty: {self.slack_penalty}\n"
        str_out += f"dual mode horizon: {self.dual_mode_horizon} (0 means deactivated)\n"
        str_out += f"dual mode controller: {self.dual_mode_controller}\n"
        str_out += f"Reference controller: {self.reference_controller}\n"
        str_out += f"Soft tracking: {self.soft_tracking}\n"
        str_out += f"Tracking Penalty Weight: {self.tracking_penalty_weight}\n"
        
        return str_out


