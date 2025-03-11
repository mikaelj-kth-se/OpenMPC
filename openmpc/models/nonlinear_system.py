import casadi as ca
import numpy as np
import control as ctrl
from openmpc.support.integrators import RK,forward_euler
from openmpc.models.model import Model


class NonlinearSystem(Model):

    def __init__(self, updfcn, states, inputs=None, disturbances=None, outfcn=None, dt=0.1):

        """
        Create a discrete NonlinearSystem object. 

        .. math::

            x_{k+1} = f(x_k, u_k, d_k)\\
            y_k = g(x_k, u_k, d_k)

        :param updfcn: The update function f(x, u, d) that defines the system dynamics.
        :type updfcn: casadi.MX,cas
        :param states: The system states.
        :type states: casadi.MX,casadi.SX
        :param inputs: The system inputs.
        :type inputs: casadi.MX,cas
        :param disturbances: The system disturbances.
        :type disturbances: casadi.MX,casadi.SX
        :param outfcn: The output function g(x, u, d).
        :type outfcn: casadi.MX,cas
        :param dt: The time step.
        :type dt: float
        """


        self.states       = states  # System states
        self.inputs       = inputs  # System inputs
        self.disturbances = disturbances  # System disturbances
        self.updfcn_expr  = updfcn  # ODE update function expression
        self.outfcn_expr  = outfcn  # Output function expression
        self._dt           = dt      # Time step

        self.has_disturbance = True if disturbances is not None else False
        self.has_inputs      = True if inputs is not None else False

        if dt <=0:
            raise ValueError("The time step must be positive.")
        
        # create fake inputs and distrurbances if necessary (by default the input and output will be of dimension 1 even if their impact is zero)
        # this allows for a readable implementation.
        if disturbances is None:
            self.disturbances = ca.MX.sym('d')
        
        if inputs is None:
            self.inputs = ca.MX.sym('u')


        # the model contains always x,u,d even if the dynamics does not depend on all of them (e.g there is a symbolic perturbation even if the system does not have any).
        # this make the code more readable and easier to implement.
        self.updfcn = ca.Function('updfcn', [self.states, self.inputs, self.disturbances], [updfcn])
        if outfcn is not None:
            self.outfcn = ca.Function('outfcn', [self.states, self.inputs, self.disturbances], [self.outfcn_expr]) 
        else:
            self.outfcn = ca.Function('outfcn', [self.states, self.inputs, self.disturbances], [self.states]) # identity mapping
            self.outfcn_expr = self.states

        super().__init__()


    @property
    def size_disturbance(self):
        """Number of disturbance inputs to the system."""
        return self.disturbances.size1()
       
    @property
    def size_input(self):
        """Number of inputs to the system."""
        return self.inputs.size1()
        
    @property
    def size_output(self):
        """Number of outputs of the system."""
        return self.outfcn_expr.size1()
    
    @property
    def size_state(self):
        return self.states.size1()
    
    @property
    def dt(self):
        return self._dt
    
    
    @staticmethod
    def _discretize_dynamics(  updfcn, states, inputs=None, disturbances=None, dt =0.1, integrator='RK4',**kwargs):
        """
        Convert a continuous-time model to a discrete-time model using numerical integration.
        This is a static method that can be called without creating an instance of the class.
        It is useful to create a discrete-time model from a continuous-time model to then create an instance of the class NonlinearSystem.
        
        """
        
        if integrator == 'Euler':
            steps = kwargs.get('steps',1)
            updfcn_discrete_expr = forward_euler(updfcn, states, inputs, disturbances, dt = dt, steps=steps)
        elif integrator in ['RK1', 'RK2', 'RK3', 'RK4']:
            order = int(integrator[-1]) if integrator != 'RK1' else 1
            updfcn_discrete_expr = RK(updfcn, states, inputs, disturbances, dt=dt, order=order)
        else:
            raise ValueError(f"Integrator '{integrator}' is not supported yet.")        

        return updfcn_discrete_expr
    
    
    @staticmethod
    def c2d(updfcn, states, inputs=None, disturbances=None, outfcn=None, dt=0.1, integrator='RK4', **kwargs):
        """
        Create a discrete-time NonlinearSystem from a continuous-time
        model using numerical integration.

        Args:
            updfcn (casadi.MX,casadi.SX): The update expression f(x, u, d) that defines the system dynamics.
            states (casadi.MX,casadi.SX): The system states.
            inputs (casadi.MX,casadi.SX, optional): The system inputs.
            disturbances (casadi.MX,casadi.SX, optional): The system disturbances.
            outfcn (casadi.MX,casadi.SX, optional): The output function g(x, u, d).
            dt (float, optional): The time step. Defaults to 0.1.
            integrator (str, optional): The integrator type. Defaults to 'RK4'.

        Returns:
            NonlinearSystem: The discrete-time system.
        """
        
        updfcn_discrete_expr = NonlinearSystem._discretize_dynamics(updfcn, states, inputs, disturbances, dt, integrator,**kwargs)
        
        return NonlinearSystem(
            updfcn =       updfcn_discrete_expr,
            outfcn =       outfcn,
            inputs =       inputs,
            states =       states,
            disturbances = disturbances,
            dt           = dt
        )
    
    
    def discrete_dynamics(self, x, u = None, d= None):
        """
        Discrete dynamics

        .. math::
            x_{k+1} = f(x_k,u_k,d_k)

        :param x: The current state.
        :type x: np.ndarray
        :param u: The current input.
        :type u: np.ndarray, optional
        :param d: The current disturbance.
        :type d: np.ndarray, optional
        :return: The next state.
        :rtype: np.ndarray
        """

        x   = super()._check_and_normalise_state(x)
        u,d = super()._check_and_normalise_inputs(u,d)

        # Create the update function based on the presence of inputs and disturbances
        x_next = self.updfcn(x, u, d)

        return x_next.full().flatten()
    
    def output(self, x, u = None, d = None):
        """
        Output function

        .. math::
            y_k = g(x_k,u_k,d_k)

        :param x: The current state.
        :type x: np.ndarray
        :param u: The current input.
        :type u: np.ndarray, optional
        :param d: The current disturbance.
        :type d: np.ndarray, optional
        :return: The output.
        :rtype: np.ndarray
        """

        x   = super()._check_and_normalise_state(x)
        u,d = super()._check_and_normalise_inputs(u,d)

        y = self.outfcn(x, u, d)

        return y.full().flatten()

    def simulate(self, x0, u = None, d = None, steps=10):
        """
        Simulate the discrete time system for a number of steps. Input should be an array of size (steps, size_input) and disturbance should be an array of size (steps, disturbance_size)
        where steps is the number of steps to simulate.

        :param x0: The initial state.
        :type x0: np.ndarray
        :param u: The input signal.
        :type u: np.ndarray, optional
        :param d: The disturbance signal.
        :type d: np.ndarray, optional
        :param steps: The number of steps to simulate.
        :type steps: int
        :return: The state trajectory and output trajectory.
        :rtype: np.ndarray, np.ndarray
        """
        

        # check the input and disturbance signals
        u,d = super().check_and_normalise_input_signals(u,d)

        if u is None and d is None:
            d = np.zeros((self.size_disturbance, steps))
            u = np.zeros((self.size_input, steps))

        
        steps = u.shape[1]
        x_trj = np.empty((self.size_state, steps+1))
        y_trj = np.empty((self.size_output, steps))
        x_trj[:, 0] = x0


        for k in range(steps):

            x_next = self.discrete_dynamics(x_trj[:, k] , u[:, k], d[:, k])
            y      = self.output(x_trj[:, k] , u[:, k], d[:, k])


            x_trj[:,k+1] = x_next
            y_trj[:,k]   = y

        
        return x_trj, y_trj 


    
    
    def linearize(self, x_ref : np.ndarray | None = None, u_ref : np.ndarray | None = None, d_ref : np.ndarray | None = None):
        """
        Linearize the NonlinearSystem around a target point with the form (xref, uref, dref).

        x_{k+1} = A*(x_k - xref) + B*(u_k - uref) + Bd*(d_k - dref)
        y_k = C*(x_k - xref) + D*(u_k - uref) + Dd*(d_k - dref)

        :param x_ref: The reference state vector. If not provided, set to a zero vector of the same dimension as 'x' in the prediction model.
        :type x_ref: np.ndarray, optional
        :param u_ref: The reference input vector. If not provided, set to a zero vector of the same dimension as 'u' in the prediction model.
        :type u_ref: np.ndarray, optional
        :param d_ref: The reference disturbance vector. If not provided, set to a zero vector of the same dimension as 'd' in the prediction model.
        :type d_ref: np.ndarray, optional
        :return: The system matrices A, B, C, D, Bd, Dd.
        :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        
        """

        if x_ref is None:
            x_ref = np.zeros(self.size_state,)
        if u_ref is None:
            u_ref = np.zeros(self.size_input,)
        if d_ref is None:
            d_ref = np.zeros(self.size_disturbance,)

        x0 = x_ref.reshape(self.size_state,)
        u0 = u_ref.reshape(self.size_input,)
        d0 = d_ref.reshape(self.size_disturbance,)

        # Compute Jacobians for A, B, E matrices
        A = ca.jacobian(self.updfcn_expr, self.states       )
        B = ca.jacobian(self.updfcn_expr, self.inputs       )
        Bd = ca.jacobian(self.updfcn_expr, self.disturbances)


        A_func = ca.Function('A_func'  , [self.states, self.inputs, self.disturbances], [A])
        B_func = ca.Function('B_func'  , [self.states, self.inputs, self.disturbances], [B])
        Bd_func = ca.Function('Bd_func', [self.states, self.inputs, self.disturbances], [Bd])

        A_matrix  = A_func(x0, u0, d0).full()
        B_matrix  = B_func(x0, u0, d0).full()
        Bd_matrix = Bd_func(x0, u0, d0).full()      

        
        C  = ca.jacobian(self.outfcn_expr, self.states       )
        D  = ca.jacobian(self.outfcn_expr, self.inputs       )
        Dd = ca.jacobian(self.outfcn_expr,  self.disturbances)

        C_func  = ca.Function('C_func' , [self.states, self.inputs, self.disturbances], [C])
        D_func  = ca.Function('D_func' , [self.states, self.inputs, self.disturbances], [D])
        Dd_func = ca.Function('Dd_func', [self.states, self.inputs, self.disturbances], [Dd])

        C_matrix  = C_func(x0, u0, d0).full()
        D_matrix  = D_func(x0, u0, d0).full()
        Dd_matrix = Dd_func(x0, u0, d0).full()


        return A_matrix, B_matrix, C_matrix, D_matrix, Bd_matrix, Dd_matrix

    def get_target_point(self, yref : np.ndarray, d : np.ndarray | None = None):
        """
        Given a reference output valye y_ref we want to find the the eqilubrium pair (xref, uref) that satisfies the equilibrium conditions:

        x_ref = f(x_ref, u_ref, d)
        y_ref = g(x_ref, u_ref, d)
        

        :param yref: The reference output value.
        :type yref: np.ndarray
        :param d: The disturbance signal. If not provided, set to a zero vector of the same dimension as 'd' in the prediction model.
        :type d: np.ndarray, optional
        :return: The equilibrium state x_ref and input u_ref.
        :rtype: np.ndarray, np.ndarray
        """
        # Define symbols
        x       = self.states
        u       = self.inputs
        d_model = self.disturbances

        if d is None:
            d = np.zeros(self.size_disturbance)
        else:
            try :
                d = d.reshape(self.size_disturbance,)
            except:
                raise ValueError(f"Disturbance signal size mismatch the system disturbance. Expected size is array with dimension compatible with {self.size_disturbance}, given size is {d.shape}.")

       
        f = ca.Function('f', [x, u, d_model], [self.updfcn_expr])
        g = ca.Function('g', [x, u, d_model], [self.outfcn_expr])
            
        # Determine the dimensions of x and u
        n_x = self.size_state
        n_u = self.size_input

        # Define symbolic variables for xref and uref
        xref = ca.MX.sym('xref', n_x)
        uref = ca.MX.sym('uref', n_u)

        # Define the nonlinear system of equations for equilibrium conditions
        eq1 = f(xref, uref, d) - xref
        eq2 = g(xref, uref, d) - yref
        eqs = ca.vertcat(eq1, eq2)


        # Define the NLP problem
        nlp = {'x': ca.vertcat(xref, uref), 'f': ca.sumsqr(uref), 'g': eqs}

        # Set solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve the NLP
        sol = solver(lbg=0, ubg=0)

        xref_sol = sol['x'][:n_x]
        uref_sol = sol['x'][n_x:n_x+n_u]
        
        x_ref = xref_sol.full().flatten()
        u_ref = uref_sol.full().flatten()


        return x_ref, u_ref
    
    def compute_lqr_controller(self, Q : np.ndarray, 
                                     R : np.ndarray, 
                                     x_ref : np.ndarray | None = None, 
                                     u_ref : np.ndarray | None = None, 
                                     d_ref : np.ndarray | None = None):
        """
        Compute the infinite-horizon LQR controller for the given prediction model.

        :param Q: The state cost matrix.
        :type Q: np.ndarray
        :param R: The input cost matrix.
        :type R: np.ndarray
        :param x_ref: The reference state vector. If not provided, set to a zero vector of the same dimension as 'x' in the prediction model.
        :type x_ref: np.ndarray, optional
        :param u_ref: The reference input vector. If not provided, set to a zero vector of the same dimension as 'u' in the prediction model.
        :type u_ref: np.ndarray, optional
        :param d_ref: The reference disturbance vector. If not provided, set to a zero vector of the same dimension as 'd' in the prediction model.
        :type d_ref: np.ndarray, optional
        
        :return: A tuple (L,P) where L is the feedback gain matrix and P is the algebraic Riccati equation solution.
        :rtype: tuple(np.ndarray, np.ndarray)
        """


        if x_ref is None:
            x_ref = np.zeros(self.size_state)
        if u_ref is None:
            u_ref = np.zeros(self.size_input)
        if d_ref is None:
            d_ref = np.zeros(self.size_disturbance)

        x_ref = x_ref.reshape(self.size_state,)
        u_ref = u_ref.reshape(self.size_input,)
        d_ref = d_ref.reshape(self.size_disturbance,)

        # Linearize the system around the target point
        A,B,C,D,Bd,Dd = self.linearize(x_ref, u_ref, d_ref)
        
        # Compute the infinite-horizon LQR controller using dlqr
        L, P, _ =  ctrl.dlqr(A, B, Q, R)
        
        return L, P
    

    def __str__(self):
        
        string_out = super().__str__() + "\n"

        string_out += "x_k+1 = f(x_k, u_k, d_k)\n"
        string_out += "y_k = g(x_k, u_k, d_k)\n\n\n"

        string_out += ("Parameters: \n")
        string_out += ("----------------------------------------\n")
        string_out += (f"dt                    : {self.dt}\n")
        string_out += (f"state dimension       : {self.size_state}\n")
        string_out += (f"input dimension       : {self.size_input}\n")
        string_out += (f"disturbance dimension : {self.size_disturbance}\n")
        string_out += (f"output dimension      : {self.size_output}\n")

        string_out += ("is autonomous          : {}\n".format(not self.has_inputs))
        string_out += ("has disturbances       : {}\n".format(self.has_disturbance))

        return string_out