import cvxpy as cp
import numpy as np
from openmpc.mpc.parameters import MPCProblem
from openmpc.invariant_sets import Polytope
from openmpc.support        import Constraint, TimedConstraint
from openmpc.models         import LinearSystem

class MPC:
    r"""
    MPC controller class for linear systems.

    The MPC class is supposed to create a controller minimizing the following optimal control problem 

    .. math::
        &\min_{x,u} \sum_{t=0}^{N-1} (x_t^T Q x_t + u_t^T R u_t) + x_N^T Q_T x_N\\
        &\text{s.t.} \\
        &x_{t+1} = A x_t + B u_t \\
        &H_u u_t \leq h_u \\
        &H_x x_t \leq h_x \\
        &H_y (C x_t + D u_t) \leq h_y \\
        &x_0 = x_0
    """
    def __init__(self, mpc_params : MPCProblem):
        """
        MPC controller class for linear systems.

        :param mpc_params: The parameters for the MPC controller.
        :type mpc_params: MPCParameters
        """

        # Extract MPC parameters
        self.params               = mpc_params
        self.system               : LinearSystem = self.params.system
        self.A, self.B, self.C, self.D = self.system.get_system_matrices()

        self.n                    : int = self.system.size_state
        self.m                    : int = self.system.size_input
        self.N                    : int = self.params.horizon
        self.Q                    : np.ndarray = self.params.Q
        self.R                    : np.ndarray = self.params.R
        self.QT                   : np.ndarray = self.params.QT
        
        self.u_constraints         : list[Constraint] = self.params.u_constraints
        self.x_constraints         : list[Constraint] = self.params.x_constraints
        self.y_constraints         : list[Constraint] = self.params.y_constraints
        self.terminal_constraints  : Constraint       = self.params.terminal_constraints
        self.global_penalty_weight : float            = self.params.global_penalty_weight
        
        self.solver               : str        = self.params.solver
        self.slack_penalty        : float      = self.params.slack_penalty
        self.terminal_set         : Constraint = self.params.terminal_set
        self.dual_mode_controller : np.ndarray = self.params.dual_mode_controller
        self.dual_mode_horizon    : int        = self.params.dual_mode_horizon

        # Define optimization variables
        self.x  = cp.Variable((self.n, self.N + 1))
        self.u  = cp.Variable((self.m, self.N))
        self.x0 = cp.Parameter(self.n)  # Initial state parameter

        
        # Set up the MPC problem
        self._setup_problem()

    def _setup_problem(self):
        """
        Set up of the MPC constraints and cost functions.
        """

        self.cost        = 0
        slack_variables  = []  # To collect slack variables for soft constraints

        self.constraints = [self.x[:, 0] == self.x0] # initial state constraints
        
        # Main MPC horizon loop
        for t in range(self.N):
            # Add the stage cost
            self.cost += cp.quad_form(self.x[:, t], self.Q) + cp.quad_form(self.u[:, t], self.R)
            
            # System dynamics
            self.constraints += [self.x[:, t + 1] == self.A @self.x[:, t] + self.B @ self.u[:, t]]
            
            # Add input constraints
            for constraint in self.u_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.u[:, t] <= b]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @self.u[:, t] <= b + np.ones(H.shape[0]) * slack]

            # Add state constraints 
            for constraint in self.x_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.x[:, t] <= b]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @self.x[:, t] <= b + np.ones(H.shape[0]) * slack]

            # Add output constraints
            for constraint in self.y_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @ (self.C @ self.x[:, t] + self.D @ self.u[:, t]) <= b]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @ (self.C @ self.x[:, t] + self.D @ self.u[:, t]) <= b + np.ones(H.shape[0]) * slack]

        
        # Dual mode implementation
        # 
        # (TODO): Brief explanation of dual mode
        if self.dual_mode_horizon != 0 :
            # Predict states using the dual mode controller beyond the control horizon
            x_dual = self.x[:, self.N]  # Initial state for the dual mode phase
            for t in range(self.dual_mode_horizon):
                # Compute control using the dual mode controller
                u_dual = -self.dual_mode_controller @ x_dual

                # Add state update for the dual mode
                x_next = cp.Variable(self.n)
                self.constraints += [x_next == self.A @x_dual + self.B @ u_dual]
                x_dual = x_next

                # Add state and input constraints during dual mode
                for constraint in self.x_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [H @x_dual <= b]

                for constraint in self.u_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [H @u_dual <= b]
                
                for constraint in self.y_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [H @(self.C @ x_dual + self.D @ u_dual) <= b]

            # Apply terminal cost and constraints at the end of the dual mode horizon
            self.cost += cp.quad_form(x_dual, self.QT)
            
            if self.terminal_set:
                H_terminal, b_terminal = self.terminal_set.to_polytope()
                self.constraints += [H_terminal @ x_dual <= b_terminal]
        else:
            # Apply terminal cost and constraints at the end of the main horizon (if no dual mode is specified)
            
            self.cost += cp.quad_form(self.x[:, self.N], self.QT)
            if self.terminal_set:
                H_terminal, b_terminal = self.terminal_set.to_polytope()
                self.constraints += [H_terminal @ self.x[:, self.N] <= b_terminal]

        # Add slack penalties to the cost function
        for slack, penalty_weight in slack_variables:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.global_penalty_weight * penalty_weight * slack  # Linear penalty
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.global_penalty_weight * penalty_weight * cp.square(slack)  # Quadratic penalty
            else :
                raise ValueError("Invalid slack penalty type. Must be 'LINEAR' or 'SQUARE'.")

        # Create the problem instance
        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)

    def compute(self, x0,*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the optimal control action for a given initial state.

        :param x0: The initial state.
        :type x0: np.ndarray
        :return: The optimal state and control trajectories.
        :rtype: tuple
        """
          
        self.x0.value = x0
        # Use the solver specified in the parameters; if None, cvxpy will select the default solver
        self.problem.solve(solver=self.solver)
        return self.x.value, self.u.value

    def get_control_action(self, x0,*args, **kwargs):
        """
        Get the first control action for a given initial state.

        :param x0: The initial state.
        :type x0: np.ndarray
        :return: The first control action.
        :rtype: float
        """
        _, u_pred = self.compute(x0)
        return np.atleast_1d(u_pred[0])[0]  # Return the first control input as a scalar




class SetPointTrackingMPC:
    r"""
    MPC controller class for set-point tracking of linear systems.


    Main controller :

    .. math::

        &\min_{x,u} sum_{t=0}^{N-1} (x_t - x_{ref})^T Q (x_t - x_{ref}) + (u_t - u_{ref})^T R (u_t - u_{ref})\\
        &s.t.\\
        &x_{t+1} = A x_t + B u_t + B_d d_t\\
        &H_u u_t \leq h_u \\
        &H_x x_t \leq h_x \\
        &H_y (C x_t + D u_t) \leq h_y \\
        &x_0 = x_0
    """

    def __init__(self, mpc_params : MPCProblem):

        """
        Initializes the TrackingMPC with the given MPC parameters.

        :param mpc_params: The parameters for the MPC.
        :type mpc_params: MPCParameters
        """

        # Extract MPC parameters
        self.params               : MPCProblem   = mpc_params
        self.system               : LinearSystem = self.params.system
        self.A, self.B, self.C, self.D = self.system.get_system_matrices()


        self.n                    : int = self.system.size_state
        self.m                    : int = self.system.size_input
        self.N                    : int = self.params.horizon
        self.Q                    : np.ndarray = self.params.Q
        self.R                    : np.ndarray = self.params.R
        self.QT                   : np.ndarray = self.params.QT
        
        self.u_constraints         : list[Constraint] = self.params.u_constraints
        self.x_constraints         : list[Constraint] = self.params.x_constraints
        self.y_constraints         : list[Constraint] = self.params.y_constraints
        self.terminal_constraints  : Constraint       = self.params.terminal_constraints
        self.global_penalty_weight : float            = self.params.global_penalty_weight
        
        self.solver               : str        = self.params.solver
        self.slack_penalty        : float      = self.params.slack_penalty
        self.terminal_set         : Constraint = self.params.terminal_set
        self.dual_mode_controller : np.ndarray = self.params.dual_mode_controller
        self.dual_mode_horizon    : int        = self.params.dual_mode_horizon


        # Tracking reference and siturbance parameters
        self.r       = cp.Parameter(self.system.size_output)  
        self.d       = cp.Parameter(self.system.size_disturbance)     # Disturbance parameter
        
        self.r.value = np.zeros(self.system.size_output)        # Initialize with a default value (e.g., zero)
        self.d.value = np.zeros(self.system.size_disturbance)

        # Disturbance matrices
        self.Bd = self.system.Bd
        self.Cd = self.system.Cd
        self.nd = self.system.size_disturbance
        
        # Define decision variables
        self.x     = cp.Variable((self.n, self.N + 1))
        self.u     = cp.Variable((self.m, self.N))
        self.x_ref = cp.Variable(self.n)
        self.u_ref = cp.Variable(self.m)
        self.x0    = cp.Parameter(self.n)  # Initial state parameter

        # Option for soft tracking constraint from mpc_params
        self.soft_tracking           = self.params.soft_tracking
        self.tracking_penalty_weight = self.params.tracking_penalty_weight
        
        # Define the slack variable if soft tracking is enabled
        if self.soft_tracking:
            self.slack_tracking = cp.Variable(self.system.size_output)  # Slack variable
        else:
            self.slack_tracking = None
               
        # Set up the MPC tracking problem
        self._setup_problem()

    def set_reference(self, reference : np.ndarray):
        """
        Set the reference for tracking.

        :param reference: The reference trajectory.
        :type reference: np.ndarray
        """
        
        self.r.value = np.array(reference).reshape(self.system.size_output,)  # Explicitly set to correct shape

    def set_disturbance(self, disturbance : np.ndarray):
        """
        Set the disturbance parameter if disturbances are defined (known disturbance entering the model).

        :param disturbance: The disturbance.
        :type disturbance: np.ndarray
        """
         
        if self.d is not None:
            self.d.value = np.array(disturbance).reshape(self.nd,)

    def _setup_problem(self):
        
        self.cost = 0
        self.constraints = [self.x[:, 0] == self.x0]
        
        if self.params.reference_reached_at_steady_state :
            # Steady-state reference constraints with or without disturbance
            self.constraints += [self.A @self.x_ref + self.B @ self.u_ref + self.Bd @ self.d == self.x_ref]
      
        # Tracking constraint with or without disturbance
        if self.soft_tracking:
            self.constraints += [self.C @ self.x_ref + self.D @ self.u_ref + self.Cd @ self.d == self.r + self.slack_tracking]
        else:
            self.constraints += [self.C @ self.x_ref + self.D @ self.u_ref + self.Cd @ self.d == self.r]
   

        # Main MPC horizon loop
        slack_variables = []  # To collect slack variables for other soft constraints
        for t in range(self.N):
            # Add the tracking cost
            self.cost += cp.quad_form(self.x[:, t] - self.x_ref, self.Q) + cp.quad_form(self.u[:, t] - self.u_ref, self.R)

            # System dynamics including disturbance if it exists
            self.constraints += [self.x[:, t + 1] == self.A @self.x[:, t] + self.B @ self.u[:, t] + self.Bd @ self.d]

            # Add input constraints
            for constraint in self.u_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.u[:, t] <= b]
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @self.u[:, t] <= b + np.ones(H.shape[0]) * slack]

            # Add state constraints
            for constraint in self.x_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.x[:, t] <= b]
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @self.x[:, t] <= b + np.ones(H.shape[0]) * slack]

            for constraint in self.y_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.C @ self.x[:, t] <= b]
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @ (self.C @ self.x[:, t] + self.D @ self.u[:,t])  <= b + np.ones(H.shape[0]) * slack]

        # Terminal cost
        self.cost += cp.quad_form(self.x[:, self.N] - self.x_ref, self.QT)

        # Terminal set constraint
        if self.terminal_set:
            H_terminal, b_terminal = self.terminal_set.to_polytope()
            self.constraints += [H_terminal @ self.x[:, self.N] <= b_terminal]

        # Dual-mode implementation
        if self.dual_mode_horizon :
            
            # Predict states using the dual-mode controller beyond the main horizon
            x_dual = self.x[:, self.N]  # Initial state for the dual mode phase
            
            for t in range(self.dual_mode_horizon):
                # Compute control using the dual-mode controller
                u_dual = -self.dual_mode_controller @ (x_dual - self.x_ref) + self.u_ref

                # Add state update for the dual mode
                x_next = cp.Variable(self.n)
                self.constraints += [x_next == self.A @x_dual + self.B @ u_dual]
        
                # Add stage cost for the dual mode
                self.cost += cp.quad_form(x_dual - self.x_ref, self.Q) + cp.quad_form(u_dual - self.u_ref, self.R)
        
                # Update for the next iteration
                x_dual = x_next

                # Add state and input constraints during dual mode
                for constraint in self.x_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [H @x_dual <= b]

                for constraint in self.u_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [H @u_dual <= b]

                for constraint in self.y_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [H @ (self.C @ x_dual + self.D @ u_dual)  <= b]

        
        # Add slack penalties for other soft constraints
        for slack, penalty_weight in slack_variables:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.global_penalty_weight * penalty_weight * slack  # Linear penalty
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.global_penalty_weight * penalty_weight * cp.square(slack)  # Quadratic penalty
            else :
                raise ValueError("Invalid slack penalty type. Must be 'LINEAR' or 'SQUARE'.")
        
        if self.soft_tracking:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.tracking_penalty_weight * cp.norm(self.slack_tracking, 1)
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.tracking_penalty_weight * cp.sum_squares(self.slack_tracking)

        # Create the problem instance
        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        

    def get_control_action(self, x0 : np.ndarray, reference : np.ndarray, disturbance : np.ndarray | None =None, **kwargs):
        
        """Compute the control action for a given state, reference, and optional disturbance."""
        self.set_reference(reference)
        if disturbance is not None and self.system.has_disturbance:
            self.set_disturbance(disturbance)
        _, u_pred = self.compute(x0)

        return np.atleast_1d(u_pred[:, 0])  # Return the entire vector for the first control input


    def compute(self, x0):
        self.x0.value = x0
        self.problem.solve(solver=self.solver)
        if self.problem.status != cp.OPTIMAL:
            raise ValueError(f"MPC problem is {self.problem.status}.")
        return self.x.value, self.u.value



class TimedMPC(SetPointTrackingMPC):
    r"""
    MPC controller class for set-point tracking of linear systems.


    Main controller :

    .. math::

        &\min_{x,u} sum_{t=0}^{N-1} (x_t - x_{ref})^T Q (x_t - x_{ref}) + (u_t - u_{ref})^T R (u_t - u_{ref})\\
        &s.t.\\
        &x_{t+1} = A x_t + B u_t + B_d d_t\\
        &H_u u_t \leq h_u \\
        &H_x x_t \leq h_x \\
        &H_y (C x_t + D u_t) \leq h_y \\
        &x_0 = x_0
    """

    def __init__(self, mpc_params : MPCProblem):

        """
        Initializes the TrackingMPC with the given MPC parameters.

        :param mpc_params: The parameters for the MPC.
        :type mpc_params: MPCParameters
        """

        self.time_state_constraints         : list[TimedConstraint] = mpc_params.state_time_constraints
        self.time_step                       :float                 = mpc_params.system.dt
        self.time_par                       : cp.Parameter          = cp.Parameter(1)  # Time parameter for the time state constraints
        
        
        self.activation_parameters : dict[Constraint,cp.Parameter] = {}
        for constraint in self.time_state_constraints:
            self.activation_parameters[constraint] = cp.Parameter(shape=(mpc_params.horizon+1,),value =np.ones(mpc_params.horizon+1,),integer = True)
               
        # Set up the MPC tracking problem
        super().__init__(mpc_params)

        self._setup_problem()  # Call the parent class's setup method to initialize the problem
        
    def set_time(self, time : float):
        """
        Set the current time for the timed constraints.

        :param time: The current time.
        :type time: float
        """
        
        self.time_par.value = np.array(time).reshape(1,)

    def update_activation_parameters(self, time : float):
        """
        Update the activation parameters for the timed constraints.

        :param time: The current time.
        :type time: float
        """
        
        time_horizon_range = np.linspace(time , self.time_step*self.N + time ,self.N+1)
        for constraint in self.time_state_constraints:
            active_set = np.bitwise_and(time_horizon_range >= constraint.start, time_horizon_range <= constraint.end)
            self.activation_parameters[constraint].value = active_set
    

    def _setup_problem(self):

        self.cost = 0
        self.constraints = [self.x[:, 0] == self.x0]
        
        if self.params.reference_reached_at_steady_state :
            # Steady-state reference constraints with or without disturbance
            self.constraints += [self.A @self.x_ref + self.B @ self.u_ref + self.Bd @ self.d == self.x_ref]
      
        # Tracking constraint with or without disturbance
        if self.soft_tracking:
            self.constraints += [self.C @ self.x_ref + self.D @ self.u_ref + self.Cd @ self.d == self.r + self.slack_tracking]
        else:
            self.constraints += [self.C @ self.x_ref + self.D @ self.u_ref + self.Cd @ self.d == self.r]
   

        # Main MPC horizon loop
        slack_variables = []  # To collect slack variables for other soft constraints
        for t in range(self.N):
            # Add the tracking cost
            self.cost += cp.quad_form(self.x[:, t] - self.x_ref, self.Q) + cp.quad_form(self.u[:, t] - self.u_ref, self.R)

            # System dynamics including disturbance if it exists
            self.constraints += [self.x[:, t + 1] == self.A @self.x[:, t] + self.B @ self.u[:, t] + self.Bd @ self.d]

            # Add input constraints
            for constraint in self.u_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.u[:, t] <= b]
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @self.u[:, t] <= b + np.ones(H.shape[0]) * slack]

            # Add state constraints
            for constraint in self.x_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.x[:, t] <= b]
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @self.x[:, t] <= b + np.ones(H.shape[0]) * slack]

            for constraint in self.y_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.C @ self.x[:, t] <= b]
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @ (self.C @ self.x[:, t] + self.D @ self.u[:,t])  <= b + np.ones(H.shape[0]) * slack]

        # Terminal cost
        self.cost += cp.quad_form(self.x[:, self.N] - self.x_ref, self.QT)

        # add exact terminal constraint 
        if not self.soft_tracking:
            self.constraints += [self.x[:, self.N] == self.x_ref]

        
        # Add slack penalties for other soft constraints
        for slack, penalty_weight in slack_variables:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.global_penalty_weight * penalty_weight * slack  # Linear penalty
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.global_penalty_weight * penalty_weight * cp.square(slack)  # Quadratic penalty
            else :
                raise ValueError("Invalid slack penalty type. Must be 'LINEAR' or 'SQUARE'.")
        
        if self.soft_tracking:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.tracking_penalty_weight * cp.norm(self.slack_tracking, 1)
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.tracking_penalty_weight * cp.sum_squares(self.slack_tracking)

        # add the timed-constraints
        for t in range(self.N+1):
            time = self.time_par + t * self.time_step
            # state time constraints (Only difference with standard set point tracking MPC)
            for constraint in self.time_state_constraints:
                H,b = constraint.to_polytope()
                activation_bit   = self.activation_parameters[constraint][t]
                self.constraints += [H @ cp.hstack((self.x[:, t], time)) <= b + (1 - activation_bit) * 1e6] # add time varying constraints

        # Create the problem instance
        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        

    def get_control_action(self, x0 : np.ndarray, reference : np.ndarray, disturbance : np.ndarray | None =None, **kwargs):
        """Compute the control action for a given state, reference, and optional disturbance."""
        
        t0 = kwargs['t0']
        self.set_reference(reference)
        self.set_time(t0)
        self.update_activation_parameters(t0) # selects active constraint

        if disturbance is not None and self.system.has_disturbance:
            self.set_disturbance(disturbance)
        _, u_pred = self.compute(x0)

        return np.atleast_1d(u_pred[:, 0])  # Return the entire vector for the first control input

    
    def get_state_and_control_trajectory(self, x0 : np.ndarray,t0:float, reference : np.ndarray, disturbance : np.ndarray | None =None)-> tuple[np.ndarray, np.ndarray,float]:

        self.set_reference(reference)
        self.set_time(t0)
        self.update_activation_parameters(t0) # selects active constraint

        if disturbance is not None and self.system.has_disturbance:
            self.set_disturbance(disturbance)
        x_pred, u_pred = self.compute(x0)
        terminal_time = self.time_par.value + self.time_step * self.N
        return x_pred, u_pred, terminal_time
        