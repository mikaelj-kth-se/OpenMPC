import numpy as np
import control
from openmpc.models.linear_system import LinearSystem

class KalmanFilter: 

    """
    Kalman filter class for output regulation for linear time-invariant systems under constant unkown input perturbations.

    Extended system takes the form

    x_{t+1} = A x_t + B u_t + Bd d_t + w_t
    d_{t+1} = d_t
    y_t     = C x_t + D u_t + Cd d_t + v_t

    where 

    A_ext = [A Bd]
            [0 I ]
    B_ext = [B]
            [0]
    C_ext = [C Cd]

    """
    def __init__(self, system : LinearSystem , Sigma_w : float | np.ndarray , Sigma_v : float | np.ndarray , is_stationary : bool =True):
        """
        Initialize the Kalman filter.
        
        :param system: Linear system model
        :type system: LinearSystem
        :param Sigma_w: Process noise covariance
        :type Sigma_w: float | np.ndarray
        :param Sigma_v: Measurement noise covariance
        :type Sigma_v: float | np.ndarray
        :param is_stationary: Flag to compute the stationary Kalman gain
        :type is_stationary: bool
        """


        self.system = system
        
        # Extract system matrices from MPC parameters
        self.A, self.B, self.C, self.D = self.system.get_system_matrices()
        self.Bd = self.system.Bd
        self.Cd = self.system.Cd
        self.nd = self.Bd.shape[1]  # Number of disturbances
        
        # Determine the dimensions of the extended state
        self.n  = self.A.shape[0]  # State dimension
        self.nx = self.n + self.nd  # Extended state dimension including disturbances

        # Construct the extended state-space matrices
        self.A_ext = np.block([
            [self.A, self.Bd],
            [np.zeros((self.nd, self.n)), np.eye(self.nd)]  # d_{t+1} = d_t
        ])
        self.B_ext = np.vstack([self.B, np.zeros((self.nd, self.B.shape[1]))])
        self.C_ext = np.hstack([self.C, self.Cd])

        # Initialize covariance matrices
        self.Sigma_w = Sigma_w  # Process noise covariance
        self.Sigma_v = Sigma_v  # Measurement noise covariance

        # Initialize the error covariance matrix (P)
        self.P = np.eye(self.nx)

        # Initialize the Kalman gain
        self.K = np.zeros((self.nx, self.C_ext.shape[0]))

        # Stationary Kalman gain flag
        self.is_stationary = is_stationary
        if self.is_stationary:
            # Compute the stationary Kalman gain using discrete-time algebraic Riccati equation (DARE)
            self.K = self._compute_stationary_gain()

        # State estimate
        self.x_est = np.zeros((self.nx, 1))  # Initial estimate [x; d]

    def _compute_stationary_gain(self):
        """Compute the stationary Kalman gain."""
        A_ext = self.A_ext
        C_ext = self.C_ext
        P = control.dare(A_ext.T, C_ext.T, self.Sigma_w, self.Sigma_v)[0]
        return P @ C_ext.T @ np.linalg.inv(C_ext @ P @ C_ext.T + self.Sigma_v)

    def set_state(self, x0, d0=None):
        """Set or reset the state and disturbance estimate."""
        self.x_est[:self.n, 0] = x0
        if d0 is not None:
            self.x_est[self.n:, 0] = d0
        else:
            self.x_est[self.n:, 0] = 0  # Initialize disturbance to zero if not provided

    def set_covariance_matrix(self, P0):
        """Set or reset the error covariance matrix."""
        self.P = P0

    def measurement_update(self, y, u):
        """Update the state estimate based on the new measurement."""
        # Predicted measurement (ensure it's a column vector)
        y_pred = self.C_ext @ self.x_est

        # Measurement residual (ensure it is a column vector)
        residual = (y.reshape(-1, 1) - y_pred)

        # Kalman gain (only update if not stationary)
        if not self.is_stationary:
            S = self.C_ext @ self.P @ self.C_ext.T + self.Sigma_v
            self.K = self.P @ self.C_ext.T @ np.linalg.inv(S)

        # Update state estimate
        self.x_est += self.K @ residual

        # Update error covariance matrix (if not stationary)
        if not self.is_stationary:
            self.P = (np.eye(self.nx) - self.K @ self.C_ext) @ self.P


    def prediction_update(self, u):
        """Predict the next state based on the system dynamics and control input."""
        # Ensure control input u is a column vector
        u = np.array(u).reshape(-1, 1)

        # Predict the next state
        self.x_est = self.A_ext @ self.x_est + self.B_ext @ u

        # Update the error covariance matrix (if not stationary)
        if not self.is_stationary:
            self.P = self.A_ext @ self.P @ self.A_ext.T + self.Sigma_w

    def get_estimated_state(self):
        """Return the estimated state vector x."""
        return self.x_est[:self.n]

    def get_estimated_disturbance(self):
        """Return the estimated disturbance vector d."""
        return self.x_est[self.n:]
