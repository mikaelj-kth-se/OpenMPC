import numpy as np
import control
from openmpc.models.linear_system import LinearSystem
from openmpc.filters.parameters import KFParameters



class KF: 

    """
    Kalman filter class for output regulation for linear time-invariant systems under constant unkown input perturbations.
    
    Extended system takes the form
    
    .. math::
        \begin{aligned}
        x_{t+1} = A x_t + B u_t + Bd d_t + wx_t\\
        d_{t+1} = d_t + wd_t\\
        y_t     = C x_t + D u_t + Cd d_t + v_t\\
        \end{aligned}

    where 

    A_ext = [A Bd]
            [0 I ]
    B_ext = [B]
            [0]
    C_ext = [C Cd]

    """
    def __init__(self, parameters    : KFParameters, 
                       is_stationary : bool =True):
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

        self.system = parameters.system
        self.params = parameters
        
        # Extract system matrices 
        self.A = self.system.A
        self.B = self.system.B
        self.C = self.system.C
        self.D = self.system.D

        self.Bd = self.system.Bd
        self.Cd = self.system.Cd
        
        # extract relevant dimensions
        self.nx = self.system.size_state
        self.nd = self.system.size_disturbance
        self.nu = self.system.size_input
        self.ny = self.system.size_output

        # Initialize covariance matrices
        self.Sigma_w  = parameters.Sigma_w  # Process noise covariance
        self.Sigma_v  = parameters.Sigma_v  # Measurement noise covariance
        self.Sigma_wd = parameters.Sigma_wd  # Disturbance noise covariance

        # Initialize state covariance estimate and state estimate
        self.P_est = parameters.P0
        self.x_est = parameters.x0

        self.K = np.zeros((self.nx,self.ny)) # Kalman Gain


        # Expand filter in case disturbance has to be estimated
        if parameters.has_distrubance_filter:

            d0_est  = parameters.d0
            P0d_est = parameters.P0d

            # Construct the extended state-space matrices
            self.A     = np.block([ [self.A                     , self.Bd],
                                    [np.zeros((self.nd, self.nx)), np.eye(self.nd)]])
            self.B     = np.vstack([self.B, np.zeros((self.nd, self.nu))])
            self.C     = np.hstack([self.C, self.Cd])
            self.x_est = np.hstack([self.x_est, d0_est])
            
            self.P_est = np.block([[self.P_est, np.zeros((self.nx, self.nd))],
                                   [np.zeros((self.nd, self.nx)), P0d_est]])
            
            self.Sigma_w = np.block([[self.Sigma_w, np.zeros((self.nx, self.nd))],
                                    [np.zeros((self.nd, self.nx)), self.Sigma_wd]])
            
            self.K = np.zeros((self.nx + self.nd ,self.ny)) # Kalman Gain

        # Stationary Kalman gain flag
        self.is_stationary = is_stationary
        if self.is_stationary:
            # Compute the stationary Kalman gain using discrete-time algebraic Riccati equation (DARE)
            self.K = self._compute_stationary_gain()


    def _compute_stationary_gain(self):
        """Compute the stationary Kalman gain."""

        print(self.A.T)
        print(self.C.T)
        print(self.Sigma_w)
        print(self.Sigma_v)

        P = control.dare(self.A.T, self.C.T, self.Sigma_w, self.Sigma_v)[0]
        return P @ self.C.T @ np.linalg.inv(self.C @ P @ self.C.T + self.Sigma_v)


    def measurement_update(self, y : np.ndarray, u: np.ndarray | None = None):
        """
        Update the state estimate based on the new measurement.

        :param y: Measurement vector
        :type y: np.ndarray
        :param u: Control input
        :type u: np.ndarray
        """
        
        # Ensure measurement y is a column vector
        if u is not None:
            try:
                u = np.array(u).reshape(self.nu,)
            except:
                raise ValueError(f"u must be reshapable into size {self.nu}")
        else:
            u = np.zeros(self.nu)

        # Predicted measurement 
        y_pred = self.C @ self.x_est + self.D @ u
        y      = np.array(y).reshape(self.ny,)

        # Measurement residual 
        residual = (y - y_pred)

        # Kalman gain (only update if not stationary)
        if not self.is_stationary:
            S      = self.C @ self.P_est @ self.C.T + self.Sigma_v
            self.K = self.P_est @ self.C.T @ np.linalg.inv(S)

        # Update state estimate
        self.x_est += self.K @ residual

        # Update error covariance matrix (if not stationary)
        if not self.is_stationary:
            self.P_est = (np.eye(self.nx) - self.K @ self.C) @ self.P_est


    def prediction_update(self, u : np.ndarray | None = None, d : np.ndarray | None = None):
        """
        Predict the next state based on the system dynamics and control input.
        :param u: Control input
        :type u: np.ndarray
        :return: Estimated state vector x
        :rtype: np.ndarray
        """
        # Ensure control input u is a column vector
        if u is not None:
            try:
                u = np.array(u).reshape(self.nu,)
            except:
                raise ValueError(f"u must be reshapable into size {self.nu}")
        else:
            u = np.zeros(self.nu)
        
        if d is not None:
            try:
                d = np.array(d).reshape(self.nd,)
            except:
                raise ValueError(f"d must be reshapable into size {self.nd}")
        else:
            d = np.zeros(self.nd)
        
        if self.params.has_distrubance_filter:
            # Predict the next state
            self.x_est = self.A @ self.x_est + self.B @ u 
        else :
            # Predict the next state
            self.x_est = self.A @ self.x_est + self.B @ u + self.Bd @ d

        # Update the error covariance matrix (if not stationary)
        if not self.is_stationary:
            self.P_est = self.A @ self.P_est @ self.A.T + self.Sigma_w

    
    def get_state_estimate(self):
        """Return the estimated state vector x."""
        return self.x_est[:self.nx]

    def get_disturbance_estimate(self):
        """Return the estimated disturbance vector d."""
        if self.params.has_distrubance_filter:
            return self.x_est[self.nx:]
        else:
            raise ValueError("Disturbance estimation is not enabled in the Kalman filter parameters. Please enable it in the parameters.")
    

    # def set_state(self, x0 : np.ndarray, d0 : np.ndarray| None = None):
    #     """Set or reset the state and disturbance estimate.
        
        
    #     :param x0: Initial state estimate for the Kalman filter.
    #     :type x0: np.ndarray
    #     :param d0: Initial disturbance estimate for the Kalman filter.
    #     :type d0: np.ndarray
    #     """
        
    #     if self.params.has_distrubance_filter:
    #         if d0 is None:
    #             d0 = np.zeros((self.nd, 1))
    #         else :
    #             try:
    #                 d0 = np.array(d0).reshape(-1,self.nd)
    #             except:
    #                 raise ValueError(f"d0 must be reshapable into size {self.nd}")
        
    #     try :
    #         x0 = np.array(x0).reshape(-1,self.nx)
    #     except :
    #         raise ValueError(f"x0 must be reshapable into size {self.nx}")
        
    #     self.x_est = np.hstack([x0.flatten(), d0.flatten()])
        

    # def set_covariance_matrix(self, P0):
    #     """Set or reset the error covariance matrix.
        
    #     :param P0: Initial error covariance matrix.
    #     :type P0: np.ndarray
    #     """

    #     if P0.shape != self.P_est.shape:
    #         raise ValueError(f"P0 must have shape ({self.P_est.shape})")
        
    #     self.P_est = P0