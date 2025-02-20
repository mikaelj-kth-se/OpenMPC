import control
import numpy as np
from openmpc.invariant_sets import Polytope, invariant_set
from openmpc.support.constraints import Constraint
from openmpc.models import LinearSystem, NonlinearSystem


class MPCParameters:
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

        Args:
            system    (LinearSystem): The state-space model of the system. The state space model should be described in desceret time.
            horizon   (int): The prediction horizon for the MPC.
            Q         (numpy.ndarray): The state weighting matrix.
            R         (numpy.ndarray): The input weighting matrix.
            QT        (numpy.ndarray, optional): The terminal state weighting matrix.
            global_penalty_weight (float): The global penalty weight for the cost function.
            solver (optional): The solver to be used for optimization.
            slack_penalty (str): The type of penalty for slack variables, default is 'SQUARE'.
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
        self.u_constraints        : list[Constraint] = []
        self.x_constraints        : list[Constraint] = []
        self.y_constraints        : list[Constraint] = []
        self.terminal_constraints : list[Constraint] = []
        self.terminal_set         : Polytope | None  = None

        # Dual mode parameters
        self.dual_mode_controller : np.ndarray = np.zeros((self.system.size_input, self.system.size_state))
        self.dual_mode_horizon    : int        = 0
        self.reference_controller : np.ndarray = np.zeros((self.system.size_input, self.system.size_state)) # This is the controller that creates the reference input to the system as U_ref = -L_ref*X


        # Tracking parameters
        self.soft_tracking           = False
        self.tracking_penalty_weight = 1.  # Default penalty weight for soft tracking

    
    def add_input_magnitude_constraint(self, limit :float , input_index : int | None = None, is_hard : bool = True, penalty_weight : float =1.):
        """Add input magnitude constraint: -limit <= u <= limit.
        
        Args:
            limits (float or tuple): If a single float, the bounds are symmetric: -limits <= u_t <= limits.
                                    If a tuple (lb, ub), the bounds are asymmetric: lb <= u_t <= ub.
            input_index (int or None): If None, apply the constraints to all inputs uniformly.
                                       If an int, apply the constraint to a specific input.
            is_hard (bool): Whether the constraint is hard or soft.
            penalty_weight (float): The penalty weight for soft constraints.
        
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

        Args:
            limits (float or tuple): If a single float, the bounds are symmetric: -limits <= u_t <= limits.
                                    If a tuple (lb, ub), the bounds are asymmetric: lb <= u_t <= ub.
            input_index (int or None): If None, apply the constraints to all inputs uniformly.
                                       If an int, apply the constraint to a specific input.
            is_hard (bool): Whether the constraint is hard or soft.
            penalty_weight (float): The penalty weight for soft constraints.
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

        Args:
            limits (float or tuple): If a single float, the bounds are symmetric: -limits <= y_t <= limits.
                                     If a tuple (lb, ub), the bounds are asymmetric: lb <= y_t <= ub.
            output_index (int or None): If None, apply the constraints to all outputs uniformly.
                                        If an int, apply the constraint to a specific output.
            is_hard (bool): Whether the constraint is hard or soft.
            penalty_weight (float): The penalty weight for soft constraints.
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

        Args:
            limits (float or tuple): If a single float, the bounds are symmetric: -limits <= x_t <= limits.
                                     If a tuple (lb, ub), the bounds are asymmetric: lb <= x_t <= ub.
            state_index (int or None): If None, apply the constraints to all states uniformly.
                                       If an int, apply the constraint to a specific state.
            is_hard (bool): Whether the constraint is hard or soft.
            penalty_weight (float): The penalty weight for soft constraints.
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
        
        
    def add_general_state_constraints(self, Ax : np.ndarray, bx : np.ndarray, is_hard : bool =True, penalty_weight : int=1):
        """Add general state constraints of the form Ax * x <= bx."""


        if Ax.shape[1] != self.system.size_state:
            raise ValueError("The number of columns in A must match the state dimension of the system.")
        

        constraint = Constraint(Ax, bx, is_hard=is_hard, penalty_weight=penalty_weight)
        self.x_constraints.append(constraint)

    def add_terminal_ingredients(self, controller=None):
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

        Px = Polytope(np.vstack(Ax_list), np.concatenate(bx_list))
        self.terminal_set = invariant_set(A_cl, Px)


                
    def add_dual_mode(self, dual_mode_horizon : int, controller : np.ndarray ) : 
        """ 
        Add a dual mode to the MPC with an optional controller and horizon.

        For a linear system you will typically put an LQR controller here.
        Example:
        ```python
            dual_mode_horizon = 10
            Q = np.eye(2)
            R = np.eye(1)
            controller,_,_ = system.get_lqr_solution(Q,R)
            mpd_parameters.add_dual_mode(dual_mode_horizon, controller)
        ```
        
        """
              
        # set the dual mode controller
        self.dual_mode_controller = controller
        # Set the dual mode horizon
        self.dual_mode_horizon = int(dual_mode_horizon)

        if self.dual_mode_controller.shape[0] != self.system.size_input or self.dual_mode_controller.shape[1] != self.system.size_state:
            raise ValueError(f"The controller shape must be {self.system.size_input,self.system.size_state}. Given {self.dual_mode_controller.shape}")
        if self.dual_mode_horizon < 0:
            raise ValueError("The horizon must be non-negative.")
        

        
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


