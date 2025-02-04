import casadi as ca
import numpy as np
from .integrators import RK
from .nonlinear_system import NonlinearSystem

class EKF:
    def __init__(self, ekfParameters):
        """
        Initialize the EKF with the given parameters.

        Parameters:
        ekfParameters (dict): Contains the nonlinear system, noise covariances,
                              initial state, and initial covariance.
        """
        self.nlsys = ekfParameters['predictionModel']
        self.Q = ekfParameters['Q']
        self.R = ekfParameters['R']
        #self.x_est = ca.DM(ekfParameters['x0'])
        self.x_est = ekfParameters['x0']
        self.P_est = ekfParameters['P0']
        self.dt = self.nlsys.dt

        # Create the RK4 integrator for the prediction step if the system is continuous
        if not self.nlsys.is_discrete:
            self.integrator = RK(self.nlsys.updfcn_expr, self.dt, self.nlsys.states, self.nlsys.inputs, self.nlsys.disturbances, order=4)

        # Create the Jacobian functions for the prediction and measurement models
        x = self.nlsys.states
        u = self.nlsys.inputs if self.nlsys.inputs is not None else ca.MX.sym('u', 0)
        d = self.nlsys.disturbances if self.nlsys.disturbances is not None else ca.MX.sym('d', 0)

        if self.nlsys.inputs is not None and self.nlsys.disturbances is not None:
            self.A = ca.Function('A', [x, u, d], [ca.jacobian(self.nlsys.updfcn_expr, x)])
            self.B = ca.Function('B', [x, u, d], [ca.jacobian(self.nlsys.updfcn_expr, u)])
        elif self.nlsys.inputs is not None:
            self.A = ca.Function('A', [x, u], [ca.jacobian(self.nlsys.updfcn_expr, x)])
            self.B = ca.Function('B', [x, u], [ca.jacobian(self.nlsys.updfcn_expr, u)])
        else:
            self.A = ca.Function('A', [x], [ca.jacobian(self.nlsys.updfcn_expr, x)])
            self.B = ca.Function('B', [x], [ca.jacobian(self.nlsys.updfcn_expr, u)])

        if self.nlsys.outfcn_expr is not None:
            if self.nlsys.inputs is not None and self.nlsys.disturbances is not None:
                self.C = ca.Function('C', [x, u, d], [ca.jacobian(self.nlsys.outfcn_expr, x)])
                self.h = ca.Function('h', [x, u, d], [self.nlsys.outfcn_expr])
            elif self.nlsys.inputs is not None:
                self.C = ca.Function('C', [x, u], [ca.jacobian(self.nlsys.outfcn_expr, x)])
                self.h = ca.Function('h', [x, u], [self.nlsys.outfcn_expr])
            else:
                self.C = ca.Function('C', [x], [ca.jacobian(self.nlsys.outfcn_expr, x)])
                self.h = ca.Function('h', [x], [self.nlsys.outfcn_expr])
        else:
            self.C = ca.Function('C', [x], [ca.jacobian(x, x)])
            self.h = ca.Function('h', [x], [x])

    def prediction_update(self, u=None, d=None):
        """
        Perform the prediction step of the EKF.

        Parameters:
        u (np.array, optional): Control input.
        d (np.array, optional): Disturbance input.

        Returns:
        None
        """
        if u is None:
            u = np.zeros(self.nlsys.m)
        if d is None:
            d = np.zeros(self.nlsys.nd)
        
        u = ca.DM(u)
        d = ca.DM(d)

        if self.nlsys.is_discrete:
            if self.nlsys.inputs is not None and self.nlsys.disturbances is not None:
                self.x_est = self.nlsys.updfcn(self.x_est, u, d).full().flatten()
            elif self.nlsys.inputs is not None:
                self.x_est = self.nlsys.updfcn(self.x_est, u).full().flatten()
            else:
                self.x_est = self.nlsys.updfcn(self.x_est).full().flatten()
        else:
            if self.nlsys.inputs is not None and self.nlsys.disturbances is not None:
                self.x_est = self.integrator(self.x_est, u, d).full().flatten()
            elif self.nlsys.inputs is not None:
                self.x_est = self.integrator(self.x_est, u).full().flatten()
            else:
                self.x_est = self.integrator(self.x_est).full().flatten()

        if self.nlsys.inputs is not None and self.nlsys.disturbances is not None:
            A_est = self.A(self.x_est, u, d).full()
            B_est = self.B(self.x_est, u, d).full()
        elif self.nlsys.inputs is not None:
            A_est = self.A(self.x_est, u).full()
            B_est = self.B(self.x_est, u).full()
        else:
            A_est = self.A(self.x_est).full()
            B_est = self.B(self.x_est).full()

        self.P_est = A_est @ self.P_est @ A_est.T + self.Q

    def measurement_update(self, y_meas, u=None, d=None):
        """
        Perform the update step of the EKF.

        Parameters:
        y_meas (np.array): Measurement vector.
        u (np.array, optional): Control input (needed for the measurement model).
        d (np.array, optional): Disturbance input.

        Returns:
        None
        """
        if u is None:
            u = np.zeros(self.nlsys.m)
        if d is None:
            d = np.zeros(self.nlsys.nd)
        
        u = ca.DM(u)
        d = ca.DM(d)

        if self.nlsys.inputs is not None and self.nlsys.disturbances is not None:
            C_est = self.C(self.x_est, u, d).full()
            y_est = self.h(self.x_est, u, d).full().flatten()
        elif self.nlsys.inputs is not None:
            C_est = self.C(self.x_est, u).full()
            y_est = self.h(self.x_est, u).full().flatten()
        else:
            C_est = self.C(self.x_est).full()
            y_est = self.h(self.x_est).full().flatten()

        S = C_est @ self.P_est @ C_est.T + self.R
        K = self.P_est @ C_est.T @ np.linalg.inv(S)

        self.x_est = self.x_est + K @ (y_meas - y_est)

        I = np.eye(self.P_est.shape[0])
        self.P_est = (I - K @ C_est) @ self.P_est

    def get_state(self):
        """
        Get the current state estimate.

        Returns:
        np.array: Current state estimate.
        """
        return self.x_est.flatten()

    def get_covariance(self):
        """
        Get the current state covariance estimate.

        Returns:
        np.array: Current state covariance estimate.
        """
        return self.P_est

    def set_state(self, x_est):
        """
        Set the current state estimate.

        Parameters:
        x_est (np.array): New state estimate.

        Returns:
        None
        """
        self.x_est = x_est

    def set_covariance(self, P_est):
        """
        Set the current state covariance estimate.

        Parameters:
        P_est (np.array): New state covariance estimate.

        Returns:
        None
        """
        self.P_est = P_est  

def create_estimator_model(nlsys):
    """
    Create an estimator model from a given NonlinearSystem model.
    
    Parameters:
    nlsys (NonlinearSystem): The original nonlinear system.
    
    Returns:
    NonlinearSystem: An augmented nonlinear system for the estimator.
    """
    # Extract states, inputs, and disturbances from the original system
    x = nlsys.states
    u = nlsys.inputs
    d = nlsys.disturbances
   
    # Define the new state which includes the original states and disturbances
    new_x = ca.vertcat(x, d)
        
    # Define the new update function based on whether the system is discrete-time or continuous-time
    if nlsys.is_discrete:
        # For discrete-time, the disturbance state should just hold its value
        d_next = d
    else:
        # For continuous-time, the disturbance dynamics are zero
        d_next = ca.MX.zeros(d.shape[0])
    
    
    # Combine original state dynamics with disturbance dynamics
    new_updfun_expr = ca.vertcat(nlsys.updfcn(x,u,d), d_next)
    new_outfun_expr = nlsys.outfcn(x,u,d)
    print(new_updfun_expr)
    print(new_outfun_expr)
    
    # Create the new NonlinearSystem object for the estimator
    estimator_nlsys = NonlinearSystem(
        updfcn=new_updfun_expr,
        states=new_x,
        inputs=u,
        disturbances=None,  # Disturbances are now part of the state
        outfcn=new_outfun_expr,
        dt=nlsys.dt
    )
    
    return estimator_nlsys
