import control
import numpy as np
from InvariantSets import Polytope, invariant_set
from .constraints import Constraint

class MPCParameters:
    def __init__(self, system, horizon, Q, R, QT=None, global_penalty_weight=1.0, solver=None, slack_penalty='SQUARE'):
        # Store the state-space model
        self.system = system
        self.A, self.B, self.C, self.D = control.ssdata(system)

        # MPC parameters
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.QT = QT
        self.global_penalty_weight = global_penalty_weight
        self.solver = solver
        self.slack_penalty = slack_penalty  # Changed from slack_norm to slack_penalty

        # Constraints as lists of `Constraint` objects
        self.u_constraints = []
        self.x_constraints = []
        self.terminal_constraints = []
        self.terminal_set = None

        # Dual mode parameters
        self.dual_mode_controller = None
        self.dual_mode_horizon = None

        # Tracking parameters
        self.soft_tracking = False
        self.tracking_penalty_weight = 1  # Default penalty weight for soft tracking

        self.Bd = None
        self.Cd = None
        
        # Extract system matrices
        self.A, self.B, self.C, _ = self.get_system_matrices()


    def get_system_matrices(self):
        """Extract the A, B, C, D matrices from the system object."""
        return self.system.A, self.system.B, self.system.C, self.system.D
        
        
    def get_system_matrices(self):
        return self.A, self.B, self.C, self.D

    def add_input_magnitude_constraint(self, limit, input_index=None, is_hard=True, penalty_weight=1):
        """Add input magnitude constraint: -limit <= u <= limit."""
        if input_index is None:
            A = np.vstack([np.eye(self.B.shape[1]), -np.eye(self.B.shape[1])])
            b = np.array([limit] * self.B.shape[1] * 2)
        else:
            A = np.array([[1 if i == input_index else 0 for i in range(self.B.shape[1])]])
            A = np.vstack([A, -A])
            b = np.array([limit, limit])
        
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.u_constraints.append(constraint)

    def add_output_magnitude_constraint(self, limit, output_index=None, is_hard=True, penalty_weight=1):
        """Add magnitude constraint on a specific output: -limit <= y[output_index] <= limit."""
        if output_index is None:
            A = np.vstack([self.C, -self.C])
            b = np.array([limit] * self.C.shape[0] * 2)
        else:
            C_row = self.C[output_index, :].reshape(1, -1)
            A = np.vstack([C_row, -C_row])
            b = np.array([limit, limit])

        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.x_constraints.append(constraint)

    def add_state_magnitude_constraint(self, limit, state_index=None, is_hard=True, penalty_weight=1):
        """Add state magnitude constraint: -limit <= x[state_index] <= limit."""
        if state_index is None:
            A = np.vstack([np.eye(self.A.shape[0]), -np.eye(self.A.shape[0])])
            b = np.array([limit] * self.A.shape[0] * 2)
        else:
            A = np.array([[1 if i == state_index else 0 for i in range(self.A.shape[0])]])
            A = np.vstack([A, -A])
            b = np.array([limit, limit])
        
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.x_constraints.append(constraint)

    def add_general_state_constraints(self, Ax, bx, is_hard=True, penalty_weight=1):
        """Add general state constraints of the form Ax * x <= bx."""
        constraint = Constraint(Ax, bx, is_hard=is_hard, penalty_weight=penalty_weight)
        self.x_constraints.append(constraint)

    def add_terminal_ingredients(self, controller=None):
        """Add terminal ingredients using a controller (defaults to LQR)."""
        if controller is None:
            # Use LQR controller as the default
            controller, _, _ = control.dlqr(self.A, self.B, self.Q, self.R)

        # Compute terminal cost using the discrete Lyapunov equation
        A_cl = self.A - self.B @ controller
        P = control.dlyap(A_cl.T, self.Q + controller.T @ self.R @ controller)
        self.QT = P
        

        # Construct the terminal set using the constraints
        Ax_list, bx_list = [], []
        for constraint in self.x_constraints:
            A, b = constraint.to_polytope()
            Ax_list.append(A)
            bx_list.append(b)

        for constraint in self.u_constraints:
            A, b = constraint.to_polytope()            
            Ax_list.append(A@controller)
            bx_list.append(b)

        Px = Polytope(np.vstack(Ax_list), np.concatenate(bx_list))
        self.terminal_set = invariant_set(A_cl, Px)
                
    def add_dual_mode(self, controller=None, dual_mode_horizon=None):
        """Add a dual mode to the MPC with an optional controller and horizon."""
        if controller is None:
            # Use LQR controller as the default
            controller, _, _ = control.dlqr(self.A, self.B, self.Q, self.R)
            self.dual_mode_controller = controller
        else:
            self.dual_mode_controller = controller
        
        # Set the dual mode horizon
        self.dual_mode_horizon = dual_mode_horizon

    def soften_tracking_constraint(self, penalty_weight=1):
        """Enable softening of the tracking constraint with the specified penalty weight."""
        self.soft_tracking = True
        self.tracking_penalty_weight = penalty_weight
        
    def add_disturbances(self, Bd=None, Cd=None):
        """Add disturbance matrices to the system."""
        n = self.A.shape[0]  # Number of states
        p = self.C.shape[0]  # Number of outputs

        if Bd is None and Cd is None:
            # If both are None, default to Cd = eye(nd) and Bd = zeros(n, nd)
            nd = p  # Set number of disturbances equal to output dimension
            self.Bd = np.zeros((n, nd))
            self.Cd = np.eye(nd)
        else:
            # If one of the matrices is not provided, set it to a zero matrix of appropriate dimension
            if Bd is None:
                nd = Cd.shape[1]
                self.Bd = np.zeros((n, nd))
            else:
                self.Bd = Bd
                nd = Bd.shape[1]

            if Cd is None:
                self.Cd = np.zeros((p, nd))
            else:
                self.Cd = Cd

            # Validate the dimensions
            if self.Bd.shape != (n, nd):
                raise ValueError(f"Bd must have shape ({n}, {nd}) but has shape {self.Bd.shape}.")
            if self.Cd.shape != (p, nd):
                raise ValueError(f"Cd must have shape ({p}, {nd}) but has shape {self.Cd.shape}.")

    def check_offsetfree_conditions(self):
        """Check the conditions for offset-free MPC."""
        n = self.A.shape[0]
        p = self.C.shape[0]
        nd = self.Bd.shape[1]  # Number of disturbances

        # Condition 1: rank([A-I, Bd; C, Cd]) == n + nd
        A_I = self.A - np.eye(n)
        top_block_1 = np.hstack([A_I, self.Bd])
        bottom_block_1 = np.hstack([self.C, self.Cd])
        matrix_1 = np.vstack([top_block_1, bottom_block_1])
        rank_condition_1 = np.linalg.matrix_rank(matrix_1) == (n + nd)

        # Condition 2: rank([A-I, B; C, 0]) == n + p
        top_block_2 = np.hstack([A_I, self.B])
        bottom_block_2 = np.hstack([self.C, np.zeros((p, self.B.shape[1]))])
        matrix_2 = np.vstack([top_block_2, bottom_block_2])
        rank_condition_2 = np.linalg.matrix_rank(matrix_2) == (n + p)

        return rank_condition_1 and rank_condition_2

