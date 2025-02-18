import casadi as ca
import numpy as np
import control as ctrl
from .integrators import RK

class NonlinearSystem:
    def __init__(self, updfcn, states, inputs=None, disturbances=None, outfcn=None, dt=0.0, **kwargs):
        self.states = states  # System states
        self.inputs = inputs  # System inputs
        self.disturbances = disturbances  # System disturbances
        self.updfcn_expr = updfcn  # ODE update function expression
        self.outfcn_expr = outfcn  # Output function expression
        self.dt = dt  # Time step
        self.is_discrete = dt > 0.0
        self.parameters = kwargs  # Additional parameters

        # Create the update function based on the presence of inputs and disturbances
        if inputs is not None and disturbances is not None:
            self.updfcn = ca.Function('updfcn', [states, inputs, disturbances], [updfcn])
        elif inputs is not None:
            self.updfcn = ca.Function('updfcn', [states, inputs], [updfcn])
        else:
            self.updfcn = ca.Function('updfcn', [states], [updfcn])

        # Create the output function if provided
        if outfcn is not None:
            if inputs is not None and disturbances is not None:
                self.outfcn = ca.Function('outfcn', [states, inputs, disturbances], [outfcn])
            elif inputs is not None:
                self.outfcn = ca.Function('outfcn', [states, inputs], [outfcn])
            else:
                self.outfcn = ca.Function('outfcn', [states], [outfcn])
        else:
            self.outfcn = None

    @property
    def n(self):
        """Number of states in the system."""
        return self.states.size1()

    @property
    def m(self):
        """Number of controlled inputs to the system."""
        if self.inputs is not None:
            return self.inputs.size1()
        return 0

    @property
    def nd(self):
        """Number of disturbance inputs to the system."""
        if self.disturbances is not None:
            return self.disturbances.size1()
        return 0

    @property
    def p(self):
        """Number of outputs of the system."""
        if self.outfcn is not None:
            return self.outfcn_expr.size1()
        return self.states.size1()

    def c2d(self, dt, integrator='RK4'):
        """Convert a continuous-time model to a discrete-time model using numerical integration."""
        if self.is_discrete:
            raise ValueError("The system is already discrete-time.")
        
        if integrator == 'Euler':
            updfcn_discrete_expr = forward_euler(self, dt, steps=1)
        elif integrator in ['RK1', 'RK2', 'RK3', 'RK4']:
            order = int(integrator[-1]) if integrator != 'RK1' else 1
            updfcn_discrete_expr = RK(self, dt, order=order)
        else:
            raise ValueError(f"Integrator '{integrator}' is not supported yet.")        

        return NonlinearSystem(
            updfcn=updfcn_discrete_expr,
            outfcn=self.outfcn_expr,
            inputs=self.inputs,
            states=self.states,
            disturbances=self.disturbances,
            dt=dt
        )

    def simulate(self, x0=None, u=None, t=None, d=None):
        """Simulate the NonlinearSystem using appropriate integrator based on system type.

        Parameters:
        x0 (np.array, optional): Initial state vector. Defaults to zero vector of appropriate dimension.
        u (np.array, optional): Input signal. Defaults to zero.
        t (float or np.array, optional): Time steps for simulation. Defaults to 0 to 10 with nlsys.dt steps.
        d (np.array, optional): Disturbance signal. Defaults to zero.

        Returns:
        np.array: Simulated state trajectory.
        """
        if x0 is None:
            x0 = ca.DM.zeros(self.n)
        else:
            x0 = ca.DM(x0)

        if isinstance(t, (int, float)):
            t = np.arange(0, t + (self.dt if self.is_discrete else 0.1), (self.dt if self.is_discrete else 0.1))
        elif isinstance(t, np.ndarray):
            dt = np.diff(t)
            if self.is_discrete and not np.allclose(dt, self.dt):
                raise ValueError("Time vector t must have consistent steps equal to nlsys.dt")
        else:
            t = np.arange(0, 10 + (self.dt if self.is_discrete else 0.1), (self.dt if self.is_discrete else 0.1))

        N_steps = len(t)

        if u is None:
            u = ca.DM.zeros((self.m, N_steps))
        elif u.ndim == 1:
            u = ca.repmat(ca.DM(u), 1, N_steps)
        else:
            u = ca.DM(u)

        if d is None:
            d = ca.DM.zeros((self.nd, N_steps))
        elif d.ndim == 1:
            d = ca.repmat(ca.DM(d), 1, N_steps)
        else:
            d = ca.DM(d)

        X = [x0]

        if self.is_discrete:
            for k in range(N_steps - 1):
                if self.nd > 0:
                    X.append(self.updfcn(X[-1], u[:, k], d[:, k]))
                elif self.m > 0:
                    X.append(self.updfcn(X[-1], u[:, k]))
                else:
                    X.append(self.updfcn(X[-1]))
        else:
            for i in range(N_steps - 1):
                dt = t[i + 1] - t[i]
                opts = {'tf': dt}
                if self.nd > 0:
                    integrator = ca.integrator('integrator', 'cvodes', {'x': self.states, 'p': ca.vertcat(self.inputs, self.disturbances), 'ode': self.updfcn_expr}, opts)
                    res = integrator(x0=X[-1], p=ca.vertcat(u[:, i], d[:, i]))
                elif self.m > 0:
                    integrator = ca.integrator('integrator', 'cvodes', {'x': self.states, 'p': self.inputs, 'ode': self.updfcn_expr}, opts)
                    res = integrator(x0=X[-1], p=u[:, i])
                else:
                    integrator = ca.integrator('integrator', 'cvodes', {'x': self.states, 'ode': self.updfcn_expr}, opts)
                    res = integrator(x0=X[-1])
                X.append(res['xf'])

        return np.array(ca.horzcat(*X).full())

    def linearize(self, targetPoint=None):
        """
        Linearize the NonlinearSystem around a target point.

        Parameters:
        targetPoint (tuple, optional): Target point (x, u, d) around which to linearize. Defaults to zero vectors of appropriate size.

        Returns:
        control.StateSpace: Linearized state-space model.
        """
        x = self.states
        u = self.inputs if self.inputs is not None else ca.MX.zeros(0)
        d = self.disturbances if self.disturbances is not None else ca.MX.zeros(0)

        if targetPoint is None:
            x0 = np.zeros(self.n)
            u0 = np.zeros(self.m)
            d0 = np.zeros(self.nd)
        else:
            x0, u0, d0 = targetPoint
            x0 = np.reshape(x0, (self.n, 1))
            u0 = np.reshape(u0, (self.m, 1))
            d0 = np.reshape(d0, (self.nd, 1))
        
        # Compute Jacobians for A, B, E matrices
        A = ca.jacobian(self.updfcn_expr, x)
        B = ca.jacobian(self.updfcn_expr, u)
        E = ca.jacobian(self.updfcn_expr, d)

        A_func = ca.Function('A_func', [x, u, d], [A])
        B_func = ca.Function('B_func', [x, u, d], [B])
        E_func = ca.Function('E_func', [x, u, d], [E])

        A_matrix = np.array(A_func(x0, u0, d0).full())
        B_matrix = np.array(B_func(x0, u0, d0).full())
        E_matrix = np.array(E_func(x0, u0, d0).full())        

        if self.outfcn is not None:
            C = ca.jacobian(self.outfcn_expr, x)
            D = ca.jacobian(self.outfcn_expr, u)
            C_func = ca.Function('C_func', [x, u, d], [C])
            D_func = ca.Function('D_func', [x, u, d], [D])

            C_matrix = np.array(C_func(x0, u0, d0).full())
            D_matrix = np.array(D_func(x0, u0, d0).full())
        else:
            C_matrix = np.eye(self.n)
            D_matrix = np.zeros((self.n, self.m))

        if self.is_discrete:
            sys = ctrl.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix, self.dt)
        else:
            sys = ctrl.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

        return sys

    def get_target_point(self, yref, d=None):
        """
        Get target values for the given reference output yref.

        Parameters:
        yref (array-like): The reference output values.
        d (array-like, optional): Nominal disturbance values. If not provided, set to a zero vector of the same dimension as 'd' in the prediction model.

        Returns:
        tuple: xref and uref that satisfy the equilibrium conditions.
        """
        # Define symbols
        x = self.states
        u = self.inputs
        d_model = self.disturbances

        # If d is not provided, set it to a zero vector of the same dimension as 'd' in the prediction model
        if d is None and d_model is not None:
            d = np.zeros(d_model.size1())

        # Define the update function
        if d_model is not None:
            f = ca.Function('f', [x, u, d_model], [self.updfcn_expr])
        else:
            f = ca.Function('f', [x, u], [self.updfcn_expr])

        # Define the output equation if provided
        if self.outfcn is not None:
            if d_model is not None:
                g = ca.Function('g', [x, u, d_model], [self.outfcn_expr])
            else:
                g = ca.Function('g', [x, u], [self.outfcn_expr])

        # Determine the dimensions of x and u
        n_x = x.size1()
        n_u = u.size1()

        # Define symbolic variables for xref and uref
        xref = ca.MX.sym('xref', n_x)
        uref = ca.MX.sym('uref', n_u)

        # Define the nonlinear system of equations for equilibrium conditions
        if self.is_discrete:
            if d_model is not None:
                eq1 = f(xref, uref, d) - xref
            else:
                eq1 = f(xref, uref) - xref
        else:
            if d_model is not None:
                eq1 = f(xref, uref, d)
            else:
                eq1 = f(xref, uref)

        if self.outfcn is not None:
            if d_model is not None:
                eq2 = g(xref, uref, d) - yref
            else:
                eq2 = g(xref, uref) - yref
            eqs = ca.vertcat(eq1, eq2)
        else:
            eqs = eq1

        # Define the NLP problem
        nlp = {'x': ca.vertcat(xref, uref), 'f': ca.sumsqr(uref), 'g': eqs}

        # Set solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve the NLP
        sol = solver(lbg=0, ubg=0)

        xref_sol = sol['x'][:n_x]
        uref_sol = sol['x'][n_x:n_x+n_u]

        return np.array(xref_sol.full()).flatten(), np.array(uref_sol.full()).flatten()
    
    def compute_lqr_controller(self, Q, R, targetPoint=None):
        """
        Compute the infinite-horizon LQR controller for the given prediction model.

        Parameters:
        Q (np.array): State cost matrix.
        R (np.array): Control cost matrix.
        targetPoint (tuple): Optional. A tuple (xref, uref, d) around which to evaluate the linearization.
                             Default is (zero vector of appropriate dimensions, zero vector of appropriate dimensions, zero vector of the size of 'd').

        Returns:
        (np.array, np.array, control.StateSpace): LQR gain matrix, solution to the discrete-time Riccati equation, and the discrete linearized system.
        """
        # Determine the linearization point
        if targetPoint is None:
            xref = np.zeros(self.n)
            uref = np.zeros(self.m)
            dnom = np.zeros(self.nd)
        else:
            xref, uref, dnom = targetPoint

        # Linearize the system around the target point
        sys_discrete = self.linearize((xref, uref, dnom))
        
        # Extract the system matrices from the discrete-time state-space model
        Ad = np.array(sys_discrete.A)
        Bd = np.array(sys_discrete.B)
        
        # Compute the infinite-horizon LQR controller using dlqr
        L, P, _ = ctrl.dlqr(Ad, Bd, Q, R)
        
        return L, P, sys_discrete
