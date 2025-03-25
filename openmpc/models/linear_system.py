import control
import numpy as np
import matplotlib.pyplot as plt
from  .model import Model


class LinearSystem(Model) :
    """
    Simple base class to define a linear system in state-space form. The system is assumed to be discrete-time.
    You can use the class method `c2d` obtain a discerete system from continuous time matrices if required.

    Model of the system 

    .. math::
        x_{k+1} = A x_k + B u_k + Bd d_k

        y_k = C x_k + D u_k + Cd d_k
    
    """

    def __init__(self, A  : np.ndarray, 
                       B  : np.ndarray | None  = None, 
                       C  : np.ndarray | None  = None, 
                       D  : np.ndarray | None  = None,
                       Bd : np.ndarray | None  = None, 
                       Cd : np.ndarray | None  = None,
                       dt : float = 1):
        """
        Initialize linear system

        :param A: State matrix
        :type A: np.ndarray
        :param B: Input matrix
        :type B: np.ndarray
        :param C: Output matrix
        :type C: np.ndarray
        :param D: Feedforward matrix
        :type D: np.ndarray
        :param Bd: Disturbance input matrix
        :type Bd: np.ndarray
        :param Cd: Disturbance output matrix
        :type Cd: np.ndarray
        :param dt: Sampling time
        :type dt: float
        """

        self._A = A

        n = A.shape[0]

        if B is None:
            self._B = np.zeros((n, 1))
            self.has_input = False
        else:
            self._B = B
            self.has_input = True

        if self._B.shape[0] != n:
            raise ValueError("The number of rows in B must match the number of rows in A")
        
        m = self._B.shape[1]

        # take all the state without feed-forward in case No oputput is given
        if C is None:
            self._C = np.eye(n)
        else:
            self._C = C
            if self._C.shape[1] != n:
                raise ValueError("The number of columns in C must match the number of rows in A (state dimension)")

        if D is None:
            self._D = np.zeros((self._C.shape[0], m))
        else:
            self._D = D
            if self._D.shape[1] != m:
                raise ValueError("The number of columns in D must match the number of columns in B (input dimension)")
        
        # distrurbances assumed to be one dimension with zero effect on the output

        if Bd is None and Cd is None:
            self.has_disturbance = False
        else :
            self.has_disturbance = True

        if Bd is None:
            self._Bd = np.zeros((self._A.shape[0], self._B.shape[1]))
        else :
            self._Bd = Bd
            if self._Bd.shape[0] != n:
                raise ValueError("The number of rows in Bd must match the number of rows in A")
        if Cd is None:
            self._Cd = np.zeros((self._C.shape[0], self._B.shape[1]))  
        else :
            self._Cd = Cd
            if self._Cd.shape[1] != m:
                raise ValueError("The number of columns in Cd must match the number of columns in D")

        self._dt = dt

        super().__init__()

    
    @property
    def A(self):
        """
        State matrix
        """
        return self._A
    @property
    def B(self):
        """
        Input matrix
        """
        return self._B
    @property
    def C(self):
        """
        Output matrix
        """
        return self._C
    @property
    def D(self):
        return self._D
    @property
    def Bd(self):
        return self._Bd
    @property
    def Cd(self):
        return self._Cd

    @property
    def size_disturbance(self):
        return self._Bd.shape[1]

    @property
    def size_input(self):
        return self._B.shape[1]
    @property
    def size_output(self):
        return self._C.shape[0]
    
    @property
    def size_state(self):
        return self._A.shape[0]
    @property
    def dt(self):
        return self._dt
    

    def set_disturbance_interface(self, Bd, Cd):
        """
        Set the disturbance interface matrices.

        :param Bd: Disturbance input matrix
        :type Bd: np.ndarray
        :param Cd: Disturbance output matrix
        :type Cd: np.ndarray
        """
        if Bd.shape[0] != self._A.shape[0]:
            raise ValueError("The number of rows in Bd must match the number of rows in A")
        if Cd.shape[0] != self._C.shape[0]:
            raise ValueError("The number of rows in Cd must match the number of rows in C")
        
        self._Bd = Bd
        self._Cd = Cd

        self.has_disturbance = True


        

    def discrete_dynamics(self, x, u = None, d = None):

        """
        Discrete dynamics
        x_{k+1} = f(x_k,u_k,d_k)

        :param x: state vector
        :type x: np.ndarray
        :param u: input vector
        :type u: np.ndarray
        :param d: disturbance vector
        :type d: np.ndarray
        :return: next state vector
        :rtype: np.ndarray
        """

        if x is None :
            raise ValueError("State cannot be None.")
        else :
            try :
                x = x.reshape(self.size_state,) # flattening
            except:
                raise ValueError(f"State size mismatch the system state. Expected size is array with dimension compatible with {self.size_state}, given size is {x.shape}.")
    
        if u is not None:
            try:
                u = u.reshape(self.size_input,)
            except:
                raise ValueError(f"Input size mismatch the system input. Expected size is array with dimension compatible with {self.size_input}, given size is {u.shape}.")
        else:
            u = np.zeros(self.size_input)
        
        if d is not None:
            try:
                d = d.reshape(self.size_disturbance,)
            except:
                raise ValueError(f"Disturbance size mismatch the system disturbance. Expected size is array with dimension compatible with {self.size_disturbance}, given size is {d.shape}.")
        else:
            d = np.zeros(self.size_disturbance)
        
        x_next = self._A @ x + self._B @ u + self._Bd @ d

        return x_next
    
    def output(self, x, u = None, d = None):
        """
        Output function
        y = g(x)

        :param x: state vector
        :type x: np.ndarray
        :param u: input vector
        :type u: np.ndarray
        :param d: disturbance vector
        :type d: np.ndarray
        :return: output vector
        :rtype: np.ndarray
        """

        x = super()._check_and_normalise_state(x)
        u,d = super()._check_and_normalise_inputs(u,d)

        y = self._C @ x + self._D @ u + self._Cd @ d

        return y.flatten()
    
    def simulate(self, x0, u = None , d=None, steps = 10 ) :
        """
        Simulate the discrete time system for a number of steps. Input should be an array of size (size_input,steps) and disturbance should be an array of size (disturbance_size,steps)
        where steps is the number of steps to simulate. Note that the function will attempt to reshape your array to the given form (size_input,steps). Hence the output will not be the length you expact if you have an array with 
        typed the wrong dimensions.

        :param x0: Initial state vector
        :type x0: np.ndarray
        :param u: Input signal
        :type u: np.ndarray
        :param d: Disturbance signal
        :type d: np.ndarray
        :param steps: Number of steps to simulate
        :type steps: int
        
        :return x_trj: State trajectory of shape (size_state, steps+1)
        :rtype x_trj: np.ndarray
        :return y_trj: Output trajectory of shape (size_output, steps)
        :rtype y_trj: np.ndarray
        """

        
        # check the input and disturbance signals and regualises them. None inputs are left unchanged
        x0  = super()._check_and_normalise_state(x0)
        u,d = super().check_and_normalise_input_signals(u,d)

        if u is None and d is None:
            d = np.zeros((self.size_disturbance, steps))
            u = np.zeros((self.size_input, steps))
            
        
        steps = u.shape[1]
        x_trj = np.empty((self.size_state, steps+1))
        y_trj = np.empty((self.size_output, steps))
        x_trj[:, 0] = x0

        for k in range(steps):

            x_next = self._A @ x_trj[:, k] + self._B @ u[:, k] + self._Bd @ d[:, k]
            y = self._C @ x_trj[:, k] + self._D @ u[:, k] + self._Cd @ d[:, k]

            x_trj[:,k+1] = x_next
            y_trj[:,k]   = y

        
        return x_trj, y_trj 
    

    def closed_loop_simulate(self, x0, L, d= None, steps=10):
        """
        
        Simulate the LinearSystem in closed-loop with an L gain (a.k.a A-BL).

        :param x0: Initial state vector
        :type x0: np.ndarray
        :param L: Gain matrix
        :type L: np.ndarray
        :param d: Disturbance signal
        :type d: np.ndarray
        :param steps: Number of steps to simulate
        :type steps: int

        :returns x_trj: State trajectory of shape (size_state, steps+1)
        :rtype x_trj: np.ndarray
        :returns y_trj: Output trajectory of shape (size_output, steps)
        :rtype y_trj: np.ndarray
        """

        x0    = x0.reshape(self.size_state, )

        if L.shape != (self.size_input, self.size_state):
            raise ValueError("L gain matrix does not match the system dimensions. Expected shape is ({}, {}) while given is {}".format(self.size_input, self.size_state, L.shape))

        if d is not None:
            try:
                d = d.reshape(self.size_disturbance, steps)
            except:
                raise ValueError("Disturbance signal size mismatch the system disturbance. Expected size is array with dimension compatible with ({},{}), given size is {}".format(self.size_disturbance, steps, d.shape))
        else:
            d = np.zeros((self.size_disturbance, steps))

        

        y_trj = np.empty((self.size_output, steps))
        x_trj = np.empty((self.size_state, steps+1))
        x_trj[:, 0] = x0.flatten() # initial state condition

        for k in range(steps):
            
            x_next = (self._A - self._B@L) @ x_trj[:, k]
            y = (self._C - self._D @L)@x_trj[:, k]

            x_trj[:,k+1] = x_next
            y_trj[:,k]   = y

        return x_trj, y_trj 


    def get_system_matrices(self):
        return self.A, self.B, self.C, self.D

    
    @staticmethod
    def c2d(A_cont, B_cont, C_cont, D_cont,dt):
        """
        Convert continuous-time state-space model to discrete-time.
        
        :param A_cont: Continuous-time state matrix
        :type A_cont: np.ndarray
        :param B_cont: Continuous-time input matrix
        :type B_cont: np.ndarray
        :param C_cont: Continuous-time output matrix
        :type C_cont: np.ndarray
        :param D_cont: Continuous-time feedforward matrix
        :type D_cont: np.ndarray
        :param dt: Sampling time
        :type dt: float
        
        :returns: Discrete-time state-space model
        :rtype: LinearSystem
        """

        # Create the state-space system
        sys_cont = control.ss(A_cont, B_cont, C_cont, D_cont)

        # Sample the system with a sampling time h = 0.25 seconds
        h = dt
        sys_disc = control.c2d(sys_cont, h)

        # Extract the discrete-time matrices
        Adist, Bdist, Cdist, Ddist = control.ssdata(sys_disc)
        
        return LinearSystem(Adist, Bdist, Cdist, Ddist)

    
    def get_lqr_controller(self,Q,R):
        """
        Solve the discrete-time algebraic Riccati equation (DARE) for the given system.

        :param Q: The state weighting matrix.
        :type Q: np.ndarray
        :param R: The input weighting matrix.
        :type R: np.ndarray

        :returns: Optimal gain of the LQR solution.
        :rtype: np.ndarray
        """
        L, _, _ = control.dlqr(self.A, self.B, Q, R) 
       
        return L 
    
    def get_lqr_cost_matrix(self,Q,R):
        """
        Solve the discrete-time algebraic Riccati equation (DARE) for the given system.

        :param Q: The state weighting matrix.
        :type Q: np.ndarray
        :param R: The input weighting matrix.
        :type R: np.ndarray

        :returns: symmetric postive semidefinite matrix to the discrete-time algebraic Riccati equation.
        :rtype: np.ndarray
        """
        _, P, _ = control.dlqr(self.A, self.B, Q, R) 
       
        return P
    
    
    
    def plot_trajectories(self, x_trj, y_trj, title):
        """
            Barebone function to plot state and output trajectories.

        Args:
            x_trj (np.array): State trajectory.
            y_trj (np.array): Output trajectory.
            title (str): The title of the plot.
        """
        
        plt.figure(figsize=(12, 6))

        # Plot state trajectory
        plt.subplot(2, 1, 1)
        plt.plot(x_trj.T)
        plt.title(f'{title} - State Trajectories')
        plt.xlabel('Time Step')
        plt.ylabel('State Values')
        plt.legend([f"State {i+1}" for i in range(self.size_state)])

        # Plot output trajectory
        plt.subplot(2, 1, 2)
        plt.plot(y_trj.T)
        plt.title(f'{title} - Output Trajectories')
        plt.xlabel('Time Step')
        plt.ylabel('Output Values')
        plt.legend([f"Output {i+1}" for i in range(self.size_output)])

        plt.tight_layout()
        plt.show()
        

    def __str__(self):
        
        string_out = super().__str__() + "\n"

        string_out += "x_(k+1) = A x_k + B u_k + Bd d_k\n"
        string_out += "y_k     = C x_k + D u_k + Cd d_k\n\n\n"
        string_out += ("Parameters: \n")
        string_out += (f"dt                    : {self.dt} (note: default is 1)\n")
        string_out += (f"state dimension       : {self.size_state}\n")
        string_out += (f"input dimension       : {self.size_input}\n")
        string_out += (f"disturbance dimension : {self.size_disturbance}\n")
        string_out += (f"output dimension      : {self.size_output}\n")

        string_out += ("is autonomous          : {}\n".format(not self.has_input))
        string_out += ("has disturbances       : {}\n".format(self.has_disturbance))

        string_out += ("System matrices: \n")
        string_out += "A = \n{}\n".format(self.A)
        string_out += "B = \n{}\n".format(self.B)
        string_out += "Bd = \n{}\n".format(self._Bd)
        string_out += "C = \n{}\n".format(self.C)
        string_out += "D = \n{}\n".format(self.D)
        string_out += "Cd = \n{}\n".format(self._Cd)

        return string_out
        




def compute_lqr_controller(A, B, Q, R):
    """
    Solve the discrete-time algebraic Riccati equation (DARE) for the given system.

    :param A: The state matrix.
    :type A: np.ndarray
    :param B: The input matrix.
    :type B: np.ndarray
    :param Q: The state weighting matrix.
    :type Q: np.ndarray
    :param R: The input weighting matrix.
    :type R: np.ndarray

    :returns: A tuple containing:
        - The optimal gain of the LQR solution. (np.ndarray)
        - The symmetric positive semidefinite solution to the discrete-time algebraic Riccati equation. (np.ndarray)
    :rtype: tuple[np.ndarray, np.ndarray]
    
    """

    L, P, E = control.dlqr(A, B, Q, R)

    return L,P



if __name__ == "__main__" :

    sys = LinearSystem(np.array([[1, 1.2], [0, 0.4]]), np.array([[1], [1]]))
    print(sys.A)
    print(sys.B)
    print(sys.C)
    print(sys.D)