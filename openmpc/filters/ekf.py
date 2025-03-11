import casadi as ca
import numpy as np
from openmpc.models.nonlinear_system import NonlinearSystem
from openmpc.filters.parameters import KFParameters

class EKF:
    def __init__(self, parameters : KFParameters):
        """
        Initialize the Extended Kalman filter.

        :param parameters: Kalman filter parameters.
        :type parameters: KFParameters
        """

        self.system : NonlinearSystem = parameters.system
        self.params = parameters
        
        # estract dimensions
        self.nx = self.system.size_state
        self.nd = self.system.size_disturbance
        self.nu = self.system.size_input
        self.ny = self.system.size_output

        # extract symbolic states for differentiation
        self.symbolic_states = self.system.states
        self.symbolic_inputs = self.system.inputs
        self.symbolic_disturbances = self.system.disturbances

        # define linearized system matrices
        self.A  = ca.Function('A', [self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances], [ca.jacobian(self.system.updfcn_expr, self.symbolic_states)])
        self.Bd = ca.Function('Bd', [self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances], [ca.jacobian(self.system.updfcn_expr, self.symbolic_disturbances)])
        
        # note that we do not linearize the output function
        self.C  = ca.Function('C', [self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances], [ca.jacobian(self.system.outfcn_expr, self.symbolic_states)])
        self.Cd = ca.Function('Cd', [self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances], [ca.jacobian(self.system.outfcn_expr, self.symbolic_disturbances)])
        self.h  = ca.Function('h', [self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances], [self.system.outfcn_expr])


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

            A = ca.vertcat(ca.horzcat(self.A(self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances), self.Bd(self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances)),
                            ca.horzcat(ca.DM.zeros(self.nd, self.nx)                                                , ca.DM.eye(self.nd)))
            
            self.A = ca.Function('A', [self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances], [A])

            C = ca.horzcat(self.C(self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances), self.Cd(self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances))
            
            self.C = ca.Function('C', [self.symbolic_states, self.symbolic_inputs, self.symbolic_disturbances], [C])

            self.x_est = np.hstack([self.x_est, d0_est]) # now the state estimate includes also the disturbance estimate
            
            self.P_est = np.block([[self.P_est, np.zeros((self.nx, self.nd))],
                                   [np.zeros((self.nd, self.nx)), P0d_est]])
            
            self.Sigma_w = np.block([[self.Sigma_w, np.zeros((self.nx, self.nd))],
                                    [np.zeros((self.nd, self.nx)), self.Sigma_wd]])
            
            self.K = np.zeros((self.nx + self.nd ,self.ny)) # Kalman Gain


    def prediction_update(self, u : np.ndarray | None = None, d: np.ndarray | None = None):
        """
        Perform the prediction step of the EKF. If the input is not provided, zero is assumed. 
        Same approach is considered for the disturbance.
        Note that if the disturbance filter is active, then the provided disturbance is ignored
        and the current disturbacne estimate is used instead.

        :param u: Control input.
        :type u: np.ndarray, optional
        :param d: Disturbance input.
        :type d: np.ndarray, optional
        """

        if u is None:
            u = np.zeros(self.nu)
        else:
            try :
                u = np.array(u).reshape(self.nu)
            except:
                raise ValueError(f"u must be of size {self.nu} or reshapable into it")
            
        if d is None:
            d = np.zeros(self.nd)
        else:
            try :
                d = np.array(d).reshape(self.nd)
            except:
                raise ValueError(f"d must be of size {self.nd} or reshapable into it")


        if self.params.has_distrubance_filter:
            # if the filter for the distrurbance is active then use the prediction model to update the state

            state_estimate = self.x_est[:self.nx]
            disturbance_estimate = self.x_est[self.nx:]
            A_est = self.A(state_estimate, u, disturbance_estimate).full()
            
            x_next = self.system.updfcn(state_estimate, u, disturbance_estimate).full().flatten()
            d_next = disturbance_estimate
            
            # update estimate of the extended state (recall disturbance dynamics is just d_t+1 = d_t)
            self.x_est = np.hstack((x_next,d_next))  # use nonlinear model to update the state
            self.P_est = A_est @ self.P_est @ A_est.T + self.Sigma_w           # use linearized model to update the covariance

        else:
            # If the disturbance filter is not active then use the provided input to update the state (meaning that the disturbance is know the same as the input is known)
            A_est = self.A(self.x_est, u, d).full()
            self.x_est = self.system.updfcn(self.x_est, u, d).full().flatten() # use nonlinear model to update the state
            self.P_est = A_est @ self.P_est @ A_est.T + self.Sigma_w           # use linearized model to update the covariance

    def measurement_update(self, y : np.ndarray , u : np.ndarray | None =None, d: np.ndarray | None = None):
        """
        Perform the measurement update step of the EKF. If the input is not provided, zero is assumed.
        Same approach is considered for the disturbance.
        Note that if the disturbance filter is active, then the provided disturbance is ignored
        and the current disturbacne estimate is used instead.

        :param y: Measured output.
        :type y: np.ndarray
        :param u: Control input.
        :type u: np.ndarray, optional
        :param d: Disturbance input.
        :type d: np.ndarray, optional
        """
        if u is None:
            u = np.zeros(self.nu)
        else:
            try :
                u = np.array(u).reshape(self.nu)
            except:
                raise ValueError(f"u must be of size {self.nu} or reshapable into it")
            
        if d is None:
            d = np.zeros(self.nd)
        else:
            try :
                d = np.array(d).reshape(self.nd)
            except:
                raise ValueError(f"d must be of size {self.nd} or reshapable into it")

        
        if self.params.has_distrubance_filter:
            # if the filter for the distrurbance is active then use the prediction model to update the state
            state_estimate = self.x_est[:self.nx]
            disturbance_estimate = self.x_est[self.nx:]

            C_est = self.C(state_estimate, u, disturbance_estimate).full()
            y_est = self.h(state_estimate, u, disturbance_estimate).full().flatten()
        
        else:
            # If the disturbance filter is not active then use the provided input to update the state (meaning that the disturbance is know the same as the input is known)
            state_estimate = self.x_est
            C_est = self.C(state_estimate, u, d).full()
            y_est = self.h(state_estimate, u, d).full().flatten()
     

        S = C_est @ self.P_est @ C_est.T + self.Sigma_v
        K = self.P_est @ C_est.T @ np.linalg.inv(S)

        self.x_est = self.x_est + K @ (y - y_est)

        I = np.eye(self.P_est.shape[0])
        self.P_est = (I - K @ C_est) @ self.P_est

    def get_state_estimate(self):
        """Return the estimated state vector x."""
        return self.x_est[:self.nx]

    def get_disturbance_estimate(self):
        """Return the estimated disturbance vector d.
        
        :return: Estimated disturbance vector.
        :rtype: np.ndarray
        """
        if self.params.has_distrubance_filter:
            return self.x_est[self.nx:]
        else:
            raise ValueError("Disturbance estimation is not enabled in the Kalman filter parameters. Please enable it in the parameters.")

    def get_covariance_estimate(self):
        """
        Get the current state covariance estimate.

        :return: Current state covariance estimate.
        :rtype: np.ndarray
        """
        return self.P_est

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
    #     """
    #     Set or reset the error covariance matrix.
        
    #     :param P0: Initial error covariance matrix.
    #     :type P0: np.ndarray
    #     """

    #     if P0.shape != self.P_est.shape:
    #         raise ValueError(f"P0 must have shape ({self.P_est.shape})")
        
    #     self.P_est = P0  
