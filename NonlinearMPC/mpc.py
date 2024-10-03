import casadi as ca
import numpy as np
from .integrators import RK
import control as ctrl

class MPC:
    def __init__(self, mpcProblemData):
        """
        Initialize the MPC controller with the given problem data.

        Parameters:
        mpcProblemData (dict): Contains the prediction horizon (N), sampling time (dt),
                               cost matrices Q, R, Q_N, prediction model, control limits,
                               state limits, output function, output limits, 
                               baseController (feedback gain L), and slack penalty weight.
        """
        self.mpcProblemData = mpcProblemData
        self.previous_X = None
        self.previous_V = None
        self._formulate_planning_problem()

    def _formulate_planning_problem(self):        
        """
        Formulate the optimization problem and create the function for optimal control.
        """
        mpcProblemData = self.mpcProblemData

        # Extract MPC problem data
        N = mpcProblemData['N']
        dt = mpcProblemData['dt']
        Q = mpcProblemData['Q']
        R = mpcProblemData['R']
        Q_N = mpcProblemData['Q_N']
        umin = mpcProblemData.get('umin', None)
        umax = mpcProblemData.get('umax', None)
        xmin = mpcProblemData.get('xmin', None)
        xmax = mpcProblemData.get('xmax', None)
        ymin = mpcProblemData.get('ymin', None)
        ymax = mpcProblemData.get('ymax', None)
        slackPenaltyWeight = mpcProblemData.get('slackPenaltyWeight', 1e6)
        Ndm = mpcProblemData.get('dualModeHorizon', 0)
        Ldm = mpcProblemData.get('dualModeController', 0)
        L = mpcProblemData.get('baseController',0)

        # Extract dimensions
        predictionModel = mpcProblemData['predictionModel']
        n = predictionModel.n
        m = predictionModel.m        
        f = predictionModel.updfcn
        g = predictionModel.outfcn
        
        # ---- decision variables ---------
        #X = ca.MX.sym('X', n, N + Ndm + 1)  # state trajectory
        #V = ca.MX.sym('V', m, N)  # new control signal to optimize
        #s = ca.MX.sym('s')  # slack variable

        opti = ca.Opti()
        # Create an Opti instance


        # ---- decision variables and parameters ---------
        X = opti.variable(n, N + Ndm + 1)  # state trajectory
        V = opti.variable(m, N)  # new control signal to optimize
        s = opti.variable()  # slack variable
        x0 = opti.parameter(n)
        #opti.set_value(x0, np.zeros(n))

        # ---- objective ---------
        cost = 0
        for k in range(N):
            x_k = X[:, k]
            v_k = V[:, k]
            u_k = -L @ x_k + v_k
            cost += ca.mtimes([x_k.T, Q, x_k]) + ca.mtimes([u_k.T, R, u_k])

        # --- dual mode horizon cost ---
        for k in range(Ndm):
            x_k = X[:, N+k]
            u_k = -Ldm@x_k
            cost += ca.mtimes([x_k.T, Q, x_k]) + ca.mtimes([u_k.T, R, u_k])
        
        # Final cost
        x_N = X[:, -1]
        cost += ca.mtimes([x_N.T, Q_N, x_N])

        # Add slack penalty
        cost += slackPenaltyWeight * s**2

        opti.minimize(cost)

        # ---- dynamic constraints --------
        opti.subject_to(X[:, 0] == x0)  # Initial state constraint
        for k in range(N):  # loop over control intervals
            x_current = X[:, k]
            v_current = V[:, k]
            u_current = -L @ x_current + v_current
            opti.subject_to(X[:, k + 1] == f(x_current, u_current))  

        for k in range(Ndm):
            x_current = X[:, N+k]
            u_current = -Ldm@x_current
            opti.subject_to(X[:, N+k+1] == f(x_current, u_current))
        
        # Control constraints
        if umin is not None and umax is not None:
            U = -L @ X[:, 0:N] + V
            for i in range(m):
                opti.subject_to(opti.bounded(umin[i], U[i, :], umax[i]))

        # State constraints
        if xmin is not None and xmax is not None:
            for i in range(n):
                opti.subject_to(opti.bounded(xmin[i], X[i, :], xmax[i]))

        # Output constraints
        if ymin is not None and ymax is not None:
            for k in range(N + Ndm + 1):
                x_k = X[:, k]
                if k == 0:
                    u_k = -L @ x0 + V[:, 0]
                elif k < N:
                    u_k = -L @ x_k + V[:, k - 1]
                else:
                    u_k = -Ldm @ x_k
                for i in range(y_k.size1()):
                    opti.subject_to(opti.bounded(ymin[i] - s, g(x_k, u_k), ymax[i] + s))
                    
        

        # set solver options
        opts = {'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-3,
            'expand': 1,
            'error_on_fail':1,                              # to guarantee transparency if solver fails
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-8,
            'ipopt.warm_start_mult_bound_push': 1e-8,
            'ipopt.mu_init': 1e-5,
            'ipopt.bound_relax_factor': 1e-9}

        opti.solver('ipopt', opts)
        
        # Now, convert the Opti object to a function
        self.planningOptimizer=opti.to_function('MPCPlanner', [x0, X, V], [X, V], ['initial_state', 'x_guess', 'v_guess'], ['x_opt', 'u_opt'])


        
    def compute_predicted_optimal_controls(self, x0):
        """
        Compute the optimal control and state trajectories for the given initial state.

        Parameters:
        x0 (np.array): Initial state.

        Returns:
        (np.array, np.array): Optimal control trajectory and optimal state trajectory.
        """

        # ---- prepare initial values for solver, then call solver function ---

        N = self.mpcProblemData['N']
        Ndm = self.mpcProblemData.get('dualModeHorizon', 0)
        Ldm = self.mpcProblemData.get('dualModeController', 0)
        L = self.mpcProblemData.get('baseController',0)
        f = self.mpcProblemData['predictionModel'].updfcn
        n = self.mpcProblemData['predictionModel'].n
        m = self.mpcProblemData['predictionModel'].m
        
        
        if self.previous_X is not None and self.previous_V is not None:
            # Use the previous solution as the initial guess
            X_initial = np.hstack((self.previous_X[:, 1:], np.zeros((n, 1))))
            X_initial[:,0]=x0
            xx = self.previous_X[:, -1]
            if Ndm>0:
                uu = -Ldm @ xx
            else:
                uu = -L @ xx    
            X_initial[:, -1] = f(xx, uu).full().flatten()    
            V_initial = np.hstack((self.previous_V[:, 1:], np.zeros((m, 1))))
        else:
            # Initial guess for V: zero matrix
            V_initial = np.zeros((m, N))

            # Initial guess for X: linear interpolation from x0 to the origin
            X_initial = np.zeros((n, N + Ndm+ 1))
            for i in range(n):
                X_initial[i, :] = np.linspace(x0[i], 0, N + Ndm + 1)

        try:
            (x_opt, v_opt)=self.planningOptimizer(x0, X_initial, V_initial)
        except Exception as e:
            print(f"Solver failed: {e}")
            

        # Extract the control and state trajectories
        #v_opt = sol.value(V).reshape(m, N)
        u_opt = -L @ x_opt[:, 0:N] + v_opt
        #x_opt = sol.value(X)
                
        # Store the solution for the next time step
        self.previous_X = x_opt
        self.previous_V = v_opt

        return u_opt, x_opt

    def get_control_action(self, x):
        """
        MPC controller that plans optimal controls and states over the prediction horizon
        and returns the first control action in the predicted optimal sequence.

        Parameters:
        x (np.array): Current state.

        Returns:
        np.array: The first control action in the predicted optimal sequence.
        """
        # Plan optimal controls and states over the next N samples
        uPred, xPred = self.compute_predicted_optimal_controls(x)

        # Apply the first control action in the predicted optimal sequence
        return uPred[:, 0]


class trackingMPC:
    def __init__(self, mpcProblemData):
        """
        Initialize the MPC controller with the given problem data.

        Parameters:
        mpcProblemData (dict): Contains the prediction horizon (N), sampling time (dt),
                               cost matrices Q, R, Q_N, prediction model, control limits,
                               state limits, output limits, baseController (feedback gain L), and slack penalty weight.
        """
        self.mpcProblemData = mpcProblemData
        self.previous_X = None
        self.previous_V = None
        self._formulate_planning_problem()


    def _formulate_planning_problem(self):        
        """
        Formulate the optimization problem and create the function for optimal control.
        """
        mpcProblemData = self.mpcProblemData

        # Extract MPC problem data
        N = mpcProblemData['N']
        dt = mpcProblemData['dt']
        Q = mpcProblemData['Q']
        R = mpcProblemData['R']
        Q_N = mpcProblemData['Q_N']
        umin = mpcProblemData.get('umin', None)
        umax = mpcProblemData.get('umax', None)
        xmin = mpcProblemData.get('xmin', None)
        xmax = mpcProblemData.get('xmax', None)
        ymin = mpcProblemData.get('ymin', None)
        ymax = mpcProblemData.get('ymax', None)
        slackPenaltyWeight = mpcProblemData.get('slackPenaltyWeight', 1e6)
        Ndm = mpcProblemData.get('dualModeHorizon', 0)
        Ldm = mpcProblemData.get('dualModeController', 0)
        L = mpcProblemData.get('baseController',0)

        # Extract dimensions
        predictionModel = mpcProblemData['predictionModel']
        n = predictionModel.n
        m = predictionModel.m  
        nd= predictionModel.nd
       
        f = predictionModel.updfcn
        g = predictionModel.outfcn

        opti = ca.Opti()
        # Create an Opti instance


        # ---- decision variables and parameters ---------
        X = opti.variable(n, N + Ndm + 1)  # state trajectory
        V = opti.variable(m, N)  # new control signal to optimize
        s = opti.variable()  # slack variable
        x0 = opti.parameter(n)
        xref= opti.parameter(n)
        uref= opti.parameter(m)
        d= opti.parameter(nd)
        
        #opti.set_value(x0, np.zeros(n))
        # ---- objective ---------
        cost = 0
        for k in range(N):
            x_k = X[:, k]
            v_k = V[:, k]
            u_k = -L @ (x_k - xref) + uref + v_k
            cost += ca.mtimes([(x_k - xref).T, Q, (x_k - xref)]) + ca.mtimes([(u_k - uref).T, R, (u_k - uref)])

        # Dual mode horizon cost
        for k in range(Ndm):
            x_k = X[:, N + k]
            u_k = -Ldm @ (x_k - xref) + uref 
            cost += ca.mtimes([(x_k - xref).T, Q, (x_k - xref)]) + ca.mtimes([(u_k - uref).T, R, (u_k - uref)])

        # Final cost
        x_T = X[:, -1]
        cost += ca.mtimes([(x_T - xref).T, Q_N, (x_T - xref)])

        # Slack variable penalty
        cost += slackPenaltyWeight * s**2

        opti.minimize(cost)

        
        # Dynamic constraints
        for k in range(N):  # Loop over control intervals
            x_current = X[:, k]
            v_current = V[:, k]
            u_current = -L @ (x_current - xref) + uref + v_current
            opti.subject_to(X[:, k + 1] == f(x_current, u_current, d))  

        for k in range(Ndm):
            x_current = X[:, N + k]
            u_current = -Ldm @ (x_current - xref) + uref
            opti.subject_to(X[:, N + k + 1] == f(x_current, u_current, d))

        # Control constraints
        if umin is not None and umax is not None:
            U = -L @ (X[:, 0:N]-xref) + uref + V
            for i in range(m):
                opti.subject_to(opti.bounded(umin[i], U[i, :], umax[i]))

        # State constraints
        if xmin is not None and xmax is not None:
            for i in range(n):
                opti.subject_to(opti.bounded(xmin[i], X[i, :], xmax[i]))

        # Output constraints
        if ymin is not None and ymax is not None:
            for k in range(N + Ndm + 1):
                x_k = X[:, k]
                if k == 0:
                    u_k = -L @ (x0-xref) + uref + V[:, 0]
                elif k < N:
                    u_k = -L @ (x_k-xref) + uref + V[:, k - 1]
                else:
                    u_k = -Ldm @ (x_k-xref) + uref
                for i in range(y_k.size1()):
                    opti.subject_to(opti.bounded(ymin[i] - s, g(x_k, u_k, d), ymax[i] + s))

        # Boundary conditions
        opti.subject_to(X[:, 0] == x0)  # Initial state constraint
        

        # set solver options
        opts = {'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-3,
            'expand': 1,
            'error_on_fail':1,                              # to guarantee transparency if solver fails
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-8,
            'ipopt.warm_start_mult_bound_push': 1e-8,
            'ipopt.mu_init': 1e-5,
            'ipopt.bound_relax_factor': 1e-9}

        opti.solver('ipopt', opts)
        
        # Now, convert the Opti object to a function
        self.planningOptimizer=opti.to_function('MPCPlanner', [x0, xref, uref, d, X, V], [X, V], ['initial_state', 'x_ref', 'v_ref', 'd', 'x_guess', 'v_guess'], ['x_opt', 'v_opt'])
        
        
    def compute_predicted_optimal_controls(self, x0, yref, d=None):
        """
        Compute the optimal control and state trajectories for the given initial state.

        Parameters:
        x0 (np.array): Initial state.
        yref (np.array): Reference output.
        d (np.array): Disturbance (optional).

        Returns:
        (np.array, np.array): Optimal control trajectory and optimal state trajectory.
        """

        
        N = self.mpcProblemData['N']
        Ndm = self.mpcProblemData.get('dualModeHorizon', 0)
        Ldm = self.mpcProblemData.get('dualModeController', 0)
        L = self.mpcProblemData.get('baseController',0)
        f = self.mpcProblemData['predictionModel'].updfcn
        n = self.mpcProblemData['predictionModel'].n
        m = self.mpcProblemData['predictionModel'].m
        
        # Determine reference point
        (xref, uref) = self.mpcProblemData['predictionModel'].get_target_point(yref, d) 
        
        if self.previous_X is not None and self.previous_V is not None:
            # Use the previous solution as the initial guess
            # Use the previous solution as the initial guess
            X_initial = np.hstack((self.previous_X[:, 1:], np.zeros((n, 1))))
            X_initial[:, 0] = x0
            
            xx = self.previous_X[:, -1]
            if Ndm>0:
                uu = -Ldm @ (xx-xref) + uref
            else:
                uu = -L @ (xx - xref) + uref
                
            X_initial[:, -1] = f(xx, uu, d).full().flatten()
            V_initial = np.hstack((self.previous_V[:, 1:], np.zeros((m, 1))))
            
        else:
            X_initial = np.zeros((n, N + Ndm + 1))
            X_initial[:, 0] = x0
            for k in range(N + Ndm):
                X_initial[:, k + 1] = f(X_initial[:, k], uref, d).full().flatten()

            # Initialize V_initial to make u = uref with the proposed state trajectory
            V_initial = L @ (X_initial[:, :N] - xref.reshape(-1, 1))

        try:
            (x_opt, v_opt)=self.planningOptimizer(x0, xref, uref, d, X_initial, V_initial)
        except Exception as e:
            print(f"Solver failed: {e}")

        # Extract the control and state trajectories
        #v_opt = sol.value(V).reshape(m, N)
        #x_opt = sol.value(X)
        u_opt = -L @ (x_opt[:, 0:N] - ca.repmat(xref, 1, N)) + uref + v_opt
                
        # Store the solution for the next time step
        self.previous_X = x_opt
        self.previous_V = v_opt

        return u_opt, x_opt


    def get_control_action(self, x, yref, d=None):
        """
        MPC controller that plans optimal controls and states over the prediction horizon
        and returns the first control action in the predicted optimal sequence.

        Parameters:
        x (np.array): Current state.
        yref (np.array): Reference output.
        d (np.array): Disturbance (optional).

        Returns:
        np.array: The first control action in the predicted optimal sequence.
        """
        # Plan optimal controls and states over the next N samples
        uPred, xPred = self.compute_predicted_optimal_controls(x, yref, d)

        # Apply the first control action in the predicted optimal sequence
        return uPred[:, 0]

