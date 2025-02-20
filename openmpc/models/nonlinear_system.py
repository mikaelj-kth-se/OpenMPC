import casadi as ca
import numpy as np
import control as ctrl
from openmpc.support.integrators import RK,forward_euler
from openmpc.models.model import Model


class NonlinearSystem(Model):

    def __init__(self, updfcn, states, inputs=None, disturbances=None, outfcn=None, dt=0.1):

        """
        Create a discrete NonlinearSystem object. 

        x_{k+1} = f(x_k, u_k, d_k)
        y_k = g(x_k, u_k, d_k)

        Parameters:
            updfcn (casadi.MX,casadi.SX): The update expression f(x, u, d) that defines the system dynamics.
            states (casadi.MX,casadi.SX): The system states.
            inputs (casadi.MX,casadi.SX, optional): The system inputs.
            disturbances (casadi.MX,casadi.SX, optional): The system disturbances.
            outfcn (casadi.MX,casadi.SX, optional): The output function g(x, u, d).
            dt (float, optional): The time step. Defaults to 0.0.
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
        self.updfcn = ca.Function('updfcn', [states, inputs, disturbances], [updfcn])
        if outfcn is not None:
            self.outfcn = ca.Function('outfcn', [states, inputs, disturbances], [outfcn]) 
        else:
            self.outfcn = ca.Function('outfcn', [states, inputs, disturbances], [states]) # identity mapping
            self.outfcn_expr = states

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
    def _discretize_dynamics(  updfcn, states, inputs=None, disturbances=None, dt =0.1, integrator='RK4'):
        """
        Convert a continuous-time model to a discrete-time model using numerical integration.
        This is a static method that can be called without creating an instance of the class.
        It is useful to create a discrete-time model from a continuous-time model to then create an instance of the class NonlinearSystem.
        
        """
        
        if integrator == 'Euler':
            updfcn_discrete_expr = forward_euler(updfcn, states, inputs, disturbances, dt = dt, steps=1)
        elif integrator in ['RK1', 'RK2', 'RK3', 'RK4']:
            order = int(integrator[-1]) if integrator != 'RK1' else 1
            updfcn_discrete_expr = RK(updfcn, states, inputs, disturbances, dt=dt, order=order)
        else:
            raise ValueError(f"Integrator '{integrator}' is not supported yet.")        

        return updfcn_discrete_expr
    
    
    @staticmethod
    def c2d(updfcn, states, inputs=None, disturbances=None, outfcn=None, dt=0.1, integrator='RK4'):
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
        
        updfcn_discrete_expr = NonlinearSystem._discretize_dynamics(updfcn, states, inputs, disturbances, dt, integrator)
        
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
        x_{k+1} = f(x_k,u_k,d_k)

        Args:
            x (np.array): state
            u (np.array): input
            d (np.array): disturbance

        Returns:
            x_next (np.array): next state
    
        """

        if u is None:
            u = np.zeros(self.size_input)
        if d is None:
            d = np.zeros(self.size_disturbance)

        # Create the update function based on the presence of inputs and disturbances
        x_next = self.updfcn(x, u, d)

        return np.asarray(x_next).flatten()


    def simulate(self, x0, u = None, d = None, steps=10):
        """
        Simulate the discrete time system for a number of steps. Input should be an array of size (steps, size_input) and disturbance should be an array of size (steps, disturbance_size)
        where steps is the number of steps to simulate.

            Args :
                x0 (np.array): initial state
                u (np.array): input array of size (size_input,stps).
                d (np.array): disturbance of size (disturbance_size,steps)
                steps (int): number of steps to simulation steps (only used if the input and disturbance are None)

            Returns:
                x (np.array): state trajectory
                y (np.array): output trajectory
        """
        

        # check the input and disturbance signals
        x0,u,d = super().check_and_regularize_input_triplet(x0,u,d,allow_matrix_input=True)

        if u is None and d is None:
            d = np.zeros((self.size_disturbance, steps))
            u = np.zeros((self.size_input, steps))

        
        steps = u.shape[1]
        x_trj = np.empty((self.size_state, steps+1))
        y_trj = np.empty((self.size_output, steps))
        x_trj[:, 0] = x0


        for k in range(steps):

            x_next = self.updfcn(x_trj[:, k] , u[:, k], d[:, k])
            y      = self.outfcn(x_trj[:, k] , u[:, k], d[:, k])


            x_trj[:,k+1] = np.asarray(x_next)
            y_trj[:,k]   = np.asarray(y)

        
        return x_trj, y_trj 


    
    
    def linearize(self, targetPoint=None):
        """
        Linearize the NonlinearSystem around a target point with the form (xref, uref, dref).

        x_{k+1} = A*(x_k - xref) + B*(u_k - uref) + Bd*(d_k - dref)
        y_k = C*(x_k - xref) + D*(u_k - uref) + Dd*(d_k - dref)

        Parameters:
        targetPoint (tuple, optional): Target point (x, u, d) around which to linearize. Defaults to zero vectors of appropriate size.

        Returns:
        
        """
        x = self.states
        u = self.inputs if self.inputs is not None else ca.MX.zeros(0)
        d = self.disturbances if self.disturbances is not None else ca.MX.zeros(0)

        if targetPoint is None:
            x0 = np.zeros(self.size_state)
            u0 = np.zeros(self.size_input)
            d0 = np.zeros(self.size_disturbance)
        else:
            x0, u0, d0 = targetPoint
            x0 = np.reshape(x0, (self.size_state, 1))
            u0 = np.reshape(u0, (self.size_input, 1))
            d0 = np.reshape(d0, (self.size_disturbance, 1))
        
        # Compute Jacobians for A, B, E matrices
        A = ca.jacobian(self.updfcn_expr, x)
        B = ca.jacobian(self.updfcn_expr, u)
        Bd = ca.jacobian(self.updfcn_expr, d)

        A_func = ca.Function('A_func', [x, u, d], [A])
        B_func = ca.Function('B_func', [x, u, d], [B])
        Bd_func = ca.Function('Bd_func', [x, u, d], [Bd])

        A_matrix = np.array(A_func(x0, u0, d0).full())
        B_matrix = np.array(B_func(x0, u0, d0).full())
        Bd_matrix = np.array(Bd_func(x0, u0, d0).full())        

        
        C = ca.jacobian(self.outfcn_expr, x)
        D = ca.jacobian(self.outfcn_expr, u)
        Dd = ca.jacobian(self.outfcn_expr, d)

        C_func = ca.Function('C_func', [x, u, d], [C])
        D_func = ca.Function('D_func', [x, u, d], [D])
        Dd_func = ca.Function('Dd_func', [x, u, d], [Dd])

        C_matrix = np.array(C_func(x0, u0, d0).full())
        D_matrix = np.array(D_func(x0, u0, d0).full())
        Dd_matrix = np.array(Dd_func(x0, u0, d0).full())

        return A,B,C,D,Bd,Dd

    def get_target_point(self, yref, d=None):
        """
        Given a reference output valye y_ref we want to find the the eqilubrium pair (xref, uref) that satisfies the equilibrium conditions:

        x_ref = f(x_ref, u_ref, d)
        y_ref = g(x_ref, u_ref, d)
        

        Parameters:
        yref (array-like): The reference output values.
        d (array-like, optional): Nominal disturbance values. If not provided, set to a zero vector of the same dimension as 'd' in the prediction model.

        Returns:
        tuple: xref and uref that satisfy the equilibrium conditions.
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
            xref = np.zeros(self.size_state)
            uref = np.zeros(self.size_input)
            dnom = np.zeros(self.size_disturbance)
        else:
            xref, uref, dnom = targetPoint
            
            try :
                xref = xref.reshape(self.size_state,)
                uref = uref.reshape(self.size_input,)
                dnom = dnom.reshape(self.size_disturbance,)
            except:
                raise ValueError(f"Something is wrong with the target point. Expected size is array with dimension compatible with {self.size_state}, {self.size_input}, {self.size_disturbance}, given sizes are {xref.shape}, {uref.shape}, {dnom.shape}.")

        # Linearize the system around the target point
        sys_discrete = self.linearize((xref, uref, dnom))
        
        # Extract the system matrices from the discrete-time state-space model
        Ad = np.array(sys_discrete.A)
        Bd = np.array(sys_discrete.B)
        
        # Compute the infinite-horizon LQR controller using dlqr
        L, P, _ = ctrl.dlqr(Ad, Bd, Q, R)
        
        return L, P, sys_discrete
    

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