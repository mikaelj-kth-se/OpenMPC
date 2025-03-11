from openmpc.models import NonlinearSystem, LinearSystem
import numpy as np


class KFParameters:
    """
    Defines set of parameters to be fed into  general Kalman filter!
    """

    def __init__(self, system: NonlinearSystem | LinearSystem,
                       Sigma_w   : np.ndarray | float,
                       Sigma_v   : np.ndarray | float,
                       P0   : np.ndarray | float,
                       x0   : np.ndarray,):


        """
        Initialize the Kalman filter parameters.
        
        :param system: System model over which the kalman filter is applied
        :type system: NonlinearSystem | LinearSystem
        :param Sigma_w: Process noise covariance. If float, the assume Sigma_w = eye(size_x) * Sigma_w
        :type Sigma_w: np.ndarray | float
        :param Sigma_v: Measurement noise covariance. If float, the assume Sigma_v = eye(size_y) * Sigma_v
        :type Sigma_v: np.ndarray | float
        :param x0: Initial state estimate for the Kalman filter.
        :type x0: np.ndarray
        :param P0: Initial error covariance matrix.
        :type P0: np.ndarray
        """



        self.system = system
        self.Sigma_w = Sigma_w
        self.Sigma_v = Sigma_v
        self.x0 = x0.flatten()
        self.P0 = P0

        self.Sigma_wd = np.zeros((system.size_disturbance, system.size_disturbance))
        self.d0       = np.zeros(system.size_disturbance)
        self.P0d      = np.zeros((system.size_disturbance, system.size_disturbance))
        
        self.has_distrubance_filter = False

        # checks 
        if isinstance(Sigma_w, float):
            self.Sigma_w = np.eye(system.size_state) * Sigma_w
        elif isinstance(Sigma_w, np.ndarray):
            if Sigma_w.shape != (system.size_state, system.size_state) :
                raise ValueError(f"Sigma_w must be of size ({system.size_state}, {system.size_state})")
        
        if isinstance(Sigma_v, float):
            self.Sigma_v = np.eye(system.size_output) * Sigma_v
        elif isinstance(Sigma_v, np.ndarray):
            if Sigma_v.shape != (system.size_output, system.size_output) :
                raise ValueError(f"Sigma_v must be of size ({system.size_output}, {system.size_output})")
        
        if isinstance(P0, float):
            self.P0 = np.eye(system.size_state) * P0
        elif isinstance(P0, np.ndarray):
            if P0.shape != (system.size_state, system.size_state) :
                raise ValueError(f"P0 must be of size ({system.size_state}, {system.size_state})")
        
        if len(x0) != system.size_state :
            raise ValueError(f"x0 must be of size {system.size_state}")
        
        
    def add_constant_disturbance_prediction(self,Sigma_wd  : np.ndarray | float, d0 : np.ndarray, P0d : np.ndarray | float) :
        """
        Add disturbace prediction to the Kalman filter parameters.

        :param Sigma_wd: Process noise covariance for the disturbance. If float, the assume Sigma_wd = eye(size_d) * Sigma_wd
        :type Sigma_wd: np.ndarray | float
        :param d0: Initial disturbance estimate for the Kalman filter.
        :type d0: np.ndarray
    
        """
        self.d0 = d0.flatten()

        if isinstance(Sigma_wd, float):
            self.Sigma_wd = np.eye(self.system.size_disturbance) * Sigma_wd
        elif isinstance(Sigma_wd, np.ndarray):
            if Sigma_wd.shape != (self.system.size_disturbance, self.system.size_disturbance) :
                raise ValueError(f"Sigma_wd must be of size ({self.system.size_disturbance}, {self.system.size_disturbance})")

        if isinstance(P0d, float):
            self.P0d = np.eye(self.system.size_disturbance) * P0d
        elif isinstance(P0d, np.ndarray):
            if P0d.shape != (self.system.size_disturbance, self.system.size_disturbance) :
                raise ValueError(f"P0d must be of size ({self.system.size_disturbance}, {self.system.size_disturbance})")


        if len(d0) != self.system.size_disturbance :
            raise ValueError(f"d0 must be of size {self.system.size_disturbance}")
        
        self.has_distrubance_filter = True


        