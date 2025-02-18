import cvxpy as cp
import numpy as np


class MPC:
    def __init__(self, mpc_params):
        # Extract MPC parameters
        self.params = mpc_params
        self.A, self.B, self.C, _ = self.params.get_system_matrices()
        self.n, self.m = self.B.shape
        self.T = self.params.horizon
        self.Q = self.params.Q
        self.R = self.params.R
        self.QT = self.params.QT
        self.u_constraints = self.params.u_constraints
        self.x_constraints = self.params.x_constraints
        self.terminal_constraints = self.params.terminal_constraints
        self.global_penalty_weight = self.params.global_penalty_weight
        self.solver = self.params.solver
        self.slack_penalty = self.params.slack_penalty
        self.terminal_set = self.params.terminal_set
        self.dual_mode_controller = self.params.dual_mode_controller
        self.dual_mode_horizon = self.params.dual_mode_horizon

        # Define optimization variables
        self.x = cp.Variable((self.n, self.T + 1))
        self.u = cp.Variable((self.m, self.T))
        self.x0 = cp.Parameter(self.n)  # Initial state parameter
        
        # Set up the MPC problem
        self._setup_problem()

    def _setup_problem(self):
        self.cost = 0
        self.constraints = [self.x[:, 0] == self.x0]
        slack_variables = []  # To collect slack variables for soft constraints
        
        # Main MPC horizon loop
        for t in range(self.T):
            # Add the stage cost
            self.cost += cp.quad_form(self.x[:, t], self.Q) + cp.quad_form(self.u[:, t], self.R)
            
            # System dynamics
            self.constraints += [self.x[:, t + 1] == self.A @ self.x[:, t] + self.B @ self.u[:, t]]
            
            # Add input constraints
            for constraint in self.u_constraints:
                A, b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [A @ self.u[:, t] <= b]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [A @ self.u[:, t] <= b + np.ones(A.shape[0]) * slack]

            # Add state constraints 
            for constraint in self.x_constraints:
                A, b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [A @ self.x[:, t] <= b]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [A @ self.x[:, t] <= b + np.ones(A.shape[0]) * slack]

        # Dual mode implementation
        if self.dual_mode_controller is not None and self.dual_mode_horizon is not None:
            # Predict states using the dual mode controller beyond the main horizon
            x_dual = self.x[:, self.T]  # Initial state for the dual mode phase
            for t in range(self.dual_mode_horizon):
                # Compute control using the dual mode controller
                u_dual = -self.dual_mode_controller @ x_dual

                # Add state update for the dual mode
                x_next = cp.Variable(self.n)
                self.constraints += [x_next == self.A @ x_dual + self.B @ u_dual]
                x_dual = x_next

                # Add state and input constraints during dual mode
                for constraint in self.x_constraints:
                    A, b = constraint.to_polytope()
                    self.constraints += [A @ x_dual <= b]

                for constraint in self.u_constraints:
                    A, b = constraint.to_polytope()
                    self.constraints += [A @ u_dual <= b]

            # Apply terminal cost and constraints at the end of the dual mode horizon
            if self.QT is not None:
                self.cost += cp.quad_form(x_dual, self.QT)
            if self.terminal_set:
                A_terminal, b_terminal = self.terminal_set.A, self.terminal_set.b
                self.constraints += [A_terminal @ x_dual <= b_terminal]
        else:
            # Apply terminal cost and constraints at the end of the main horizon (if no dual mode is specified)
            if self.QT is not None:
                self.cost += cp.quad_form(self.x[:, self.T], self.QT)
            if self.terminal_set:
                A_terminal, b_terminal = self.terminal_set.A, self.terminal_set.b
                self.constraints += [A_terminal @ self.x[:, self.T] <= b_terminal]

        # Add slack penalties to the cost function
        for slack, penalty_weight in slack_variables:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.global_penalty_weight * penalty_weight * slack  # Linear penalty
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.global_penalty_weight * penalty_weight * cp.square(slack)  # Quadratic penalty

        # Create the problem instance
        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)

    def compute(self, x0):
        self.x0.value = x0
        # Use the solver specified in the parameters; if None, cvxpy will select the default solver
        self.problem.solve(solver=self.solver)
        return self.x.value, self.u.value

    def get_control_action(self, x0):
        _, u_pred = self.compute(x0)
        return np.atleast_1d(u_pred[0])[0]  # Return the first control input as a scalar


class TrackingMPC:
    """
    A class to implement Model Predictive Control (MPC) for tracking a reference trajectory.

    Attributes:
        params (MPCParameters): The parameters for the MPC.
        A (numpy.ndarray): The state transition matrix.
        B (numpy.ndarray): The control input matrix.
        C (numpy.ndarray): The output matrix.
        n (int): The number of states.
        m (int): The number of control inputs.
        T (int): The prediction horizon.
        Q (numpy.ndarray): The state weighting matrix.
        R (numpy.ndarray): The input weighting matrix.
        QT (numpy.ndarray, optional): The terminal state weighting matrix.
        terminal_set (Polytope, optional): The terminal set for the MPC.
        dual_mode_controller (optional, np.array): The dual mode controller.
        dual_mode_horizon (optional,int): The horizon for the dual mode controller.
        u_constraints (list): List of input constraints as `Constraint` objects.
        x_constraints (list): List of state constraints as `Constraint` objects.
        global_penalty_weight (float): The global penalty weight for the cost function.
        solver (optional): The solver to be used for optimization.
        slack_penalty (str): The type of penalty for slack variables.
    """

    def __init__(self, mpc_params):

        """
        Initializes the TrackingMPC with the given MPC parameters.

        Args:
            mpc_params (MPCParameters): The parameters for the MPC.
        """

        # Extract MPC parameters
        self.params = mpc_params
        self.A, self.B, self.C, _ = self.params.get_system_matrices()
        self.n, self.m = self.B.shape
        self.T = self.params.horizon
        self.Q = self.params.Q
        self.R = self.params.R
        self.QT = self.params.QT  # Terminal cost matrix
        self.terminal_set = self.params.terminal_set
        self.dual_mode_controller = self.params.dual_mode_controller
        self.dual_mode_horizon = self.params.dual_mode_horizon
        self.u_constraints = self.params.u_constraints
        self.x_constraints = self.params.x_constraints
        self.global_penalty_weight = self.params.global_penalty_weight
        self.solver = self.params.solver
        self.slack_penalty = self.params.slack_penalty

        # Tracking reference
        self.r = cp.Parameter(shape=(self.C.shape[0],))  # Ensure r is a 1D array with appropriate shape
        self.r.value = np.zeros(self.C.shape[0])  # Initialize with a default value (e.g., zero)

        # Disturbance matrices
        self.Bd = self.params.Bd if self.params.Bd is not None else np.zeros((self.n, 0))
        self.Cd = self.params.Cd if self.params.Cd is not None else np.zeros((self.C.shape[0], 0))
        self.nd = self.Bd.shape[1]  # Number of disturbances
        
        # Define decision variables
        self.x = cp.Variable((self.n, self.T + 1))
        self.u = cp.Variable((self.m, self.T))
        self.x_ref = cp.Variable(self.n)
        self.u_ref = cp.Variable(self.m)
        self.x0 = cp.Parameter(self.n)  # Initial state parameter

        # Option for soft tracking constraint from mpc_params
        self.soft_tracking = self.params.soft_tracking
        self.tracking_penalty_weight = self.params.tracking_penalty_weight
        
        # Define the slack variable if soft tracking is enabled
        if self.soft_tracking:
            self.slack_tracking = cp.Variable(self.C.shape[0])  # Slack variable
        else:
            self.slack_tracking = None

        # Only create the disturbance parameter if nd > 0
        if self.nd > 0:
            self.d = cp.Parameter(self.nd)  # Disturbance parameter
        else:
            self.d = None  # No disturbances            
            
        # Set up the MPC tracking problem
        self._setup_problem()

    def set_reference(self, reference):
        """Set the reference for tracking."""
        self.r.value = np.array(reference).reshape(self.C.shape[0],)  # Explicitly set to correct shape

    def set_disturbance(self, disturbance):
        """Set the disturbance parameter if disturbances are defined (known distrubance enetring the model)."""
        if self.d is not None:
            self.d.value = np.array(disturbance).reshape(self.nd,)

    def _setup_problem(self):
        self.cost = 0
        self.constraints = [self.x[:, 0] == self.x0]

        # Steady-state reference constraints with or without disturbance
        if self.d is not None:
            self.constraints += [self.A @ self.x_ref + self.B @ self.u_ref + self.Bd @ self.d == self.x_ref]
        else:
            self.constraints += [self.A @ self.x_ref + self.B @ self.u_ref == self.x_ref]

        # Tracking constraint with or without disturbance
        if self.soft_tracking:
            if self.d is not None:
                self.constraints += [self.C @ self.x_ref + self.Cd @ self.d == self.r + self.slack_tracking]
            else:
                self.constraints += [self.C @ self.x_ref == self.r + self.slack_tracking]
        else:
            if self.d is not None:
                self.constraints += [self.C @ self.x_ref + self.Cd @ self.d == self.r]
            else:
                self.constraints += [self.C @ self.x_ref == self.r]

        # Main MPC horizon loop
        slack_variables = []  # To collect slack variables for other soft constraints
        for t in range(self.T):
            # Add the tracking cost
            self.cost += cp.quad_form(self.x[:, t] - self.x_ref, self.Q) + cp.quad_form(self.u[:, t] - self.u_ref, self.R)

            # System dynamics including disturbance if it exists
            if self.d is not None:
                self.constraints += [self.x[:, t + 1] == self.A @ self.x[:, t] + self.B @ self.u[:, t] + self.Bd @ self.d]
            else:
                self.constraints += [self.x[:, t + 1] == self.A @ self.x[:, t] + self.B @ self.u[:, t]]

            # Add input constraints
            for constraint in self.u_constraints:
                A, b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [A @ self.u[:, t] <= b]
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [A @ self.u[:, t] <= b + np.ones(A.shape[0]) * slack]

            # Add state constraints
            for constraint in self.x_constraints:
                A, b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [A @ self.x[:, t] <= b]
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [A @ self.x[:, t] <= b + np.ones(A.shape[0]) * slack]

        # Terminal cost
        if self.QT is not None:
            self.cost += cp.quad_form(self.x[:, self.T] - self.x_ref, self.QT)

        # Terminal set constraint
        if self.terminal_set:
            A_terminal, b_terminal = self.terminal_set.A, self.terminal_set.b
            self.constraints += [A_terminal @ self.x[:, self.T] <= b_terminal]

        # Dual-mode implementation
        if self.dual_mode_controller is not None and self.dual_mode_horizon is not None:
            # Predict states using the dual-mode controller beyond the main horizon
            x_dual = self.x[:, self.T]  # Initial state for the dual mode phase
            for t in range(self.dual_mode_horizon):
                # Compute control using the dual-mode controller
                u_dual = -self.dual_mode_controller @ (x_dual - self.x_ref) + self.u_ref

                # Add state update for the dual mode
                x_next = cp.Variable(self.n)
                self.constraints += [x_next == self.A @ x_dual + self.B @ u_dual]
        
                # Add stage cost for the dual mode
                self.cost += cp.quad_form(x_dual - self.x_ref, self.Q) + cp.quad_form(u_dual - self.u_ref, self.R)
        
                # Update for the next iteration
                x_dual = x_next

                # Add state and input constraints during dual mode
                for constraint in self.x_constraints:
                    A, b = constraint.to_polytope()
                    self.constraints += [A @ x_dual <= b]

                for constraint in self.u_constraints:
                    A, b = constraint.to_polytope()
                    self.constraints += [A @ u_dual <= b]

        # Add slack penalties for other soft constraints
        for slack, penalty_weight in slack_variables:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.global_penalty_weight * penalty_weight * slack  # Linear penalty
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.global_penalty_weight * penalty_weight * cp.square(slack)  # Quadratic penalty

        # Add slack penalty for soft tracking if applicable
        if self.soft_tracking:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.global_penalty_weight * self.tracking_penalty_weight * cp.norm(self.slack_tracking, 1)  # Linear penalty
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.global_penalty_weight * self.tracking_penalty_weight * cp.norm(self.slack_tracking, 2)  # Quadratic penalty

        # Create the problem instance
        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        
        
   

    def get_control_action(self, x0, reference, disturbance=None):
        """Compute the control action for a given state, reference, and optional disturbance."""
        self.set_reference(reference)
        if disturbance is not None and self.d is not None:
            self.set_disturbance(disturbance)
        _, u_pred = self.compute(x0)
        return u_pred[:, 0]  # Return the entire vector for the first control input


    def compute(self, x0):
        self.x0.value = x0
        self.problem.solve(solver=self.solver)
        return self.x.value, self.u.value

