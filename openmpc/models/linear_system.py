import control
import numpy as np
import matplotlib.pyplot as plt
from  .model import Model


class LinearSystem(Model) :
    """
    Simple base class to define a linear system in state-space form. The system is assumed to be discrete-time.
    You can use the class method `c2d` obtain a discerete system from continuous time matrices if required.

    Model of the system 

    x_{k+1} = A x_k + B u_k + Bd d_k
        y_k = C x_k + D u_k + Cd d_k
    
    """

    def __init__(self, A, B=None, C = None, D = None,Bd = None, Cd = None, dt = 1):
        """
        Initialize linear system

        Args:
            A (numpy.ndarray): The state matrix.
            B (numpy.ndarray): The input matrix.
            C (numpy.ndarray): The output matrix.
            D  (numpy.ndarray): The feedforward matrix.
            Bd (numpy.ndarray): The disturbance input matrix.
            Cd (numpy.ndarray): The disturbance output matrix.
            dt (float): The sampling time.
        """

        self._A = A

        if B is None:
            self._B = np.zeros((A.shape[0], 1))
            self.has_input = False
        else:
            self._B = B
            self.has_input = True

        if self._B.shape[0] != self._A.shape[0]:
            raise ValueError("The number of rows in B must match the number of rows in A")
        
        # take all the state without feed-forward in case No oputput is given
        if C is None:
            self._C = np.eye(A.shape[0])
        else:
            self._C = C

        if D is None:
            self._D = np.zeros((A.shape[0], B.shape[1]))
        else:
            self._D = D
        
        # distrurbances assumed to be one dimension with zero effect on the output

        if Bd is None and Cd is None:
            self.has_disturbance = False
        else :
            self.has_disturbance = True

        if Bd is None:
            self._Bd = np.zeros((A.shape[0], B.shape[1]))
        if Cd is None:
            self._Cd = np.zeros((C.shape[0], B.shape[1]))  

        self._dt = dt

        super().__init__()

    
    @property
    def A(self):
        return self._A
    @property
    def B(self):
        return self._B
    @property
    def C(self):
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


        

    def discrete_dynamics(self, x, u = None, d = None):

        """
        Discrete dynamics
        x_{k+1} = f(x_k,u_k,d_k)

    

        Args:
            x (np.array): state
            u (np.array): input
            d (np.array): disturbance

        Returns:
            x_next (np.array): next state as a row vector
        """

        x = super()._check_and_normalise_state(x) 
        u,x = super()._check_and_normalise_inputs(u,d) # sets to zero none inputs
       
        x_next = self._A @ x + self._B @ u + self._Bd @ d

        return x_next.flatten()
    
    
    
    def simulate(self, x0, u = None , d=None, steps = 10 ) :
        """
            Simulate the discrete time system for a number of steps. Input should be an array of size (size_input,steps) and disturbance should be an array of size (disturbance_size,steps)
            where steps is the number of steps to simulate. Note that the function will attempt to reshape your array to the given form (size_input,steps). Hence the output will not be the length you expact if you have an array with 
            typed the wrong dimensions.

            Args :
                x0 (np.array): initial state
                u (np.array): input array of size (size_input,stps).
                d (np.array): disturbance of size (disturbance_size,steps)
                steps (int): number of steps to simulation steps (only used if the input and disturbance are None)

            Returns:
                x (np.array): state trajectory
                y (np.array): output trajectory
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

        Args:
            x0    (np.array): Initial state vector. Defaults to zero vector of appropriate dimension.
            L     (np.array): L gain matrix.
            steps (int)     : Number of steps to simulate.

        Returns:
            x_trj (np.array): Simulated state trajectory.
            y_trj (np.array): Simulated output trajectory.

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
        
        Args:
            A (numpy.ndarray): The state matrix.
            B (numpy.ndarray): The input matrix.
            C (numpy.ndarray): The output matrix.
            D (numpy.ndarray): The feedforward matrix.
            dt (float): The sampling time.

        Returns:
            discrete_system : LinearSystems 

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

        Args:
            Q (numpy.ndarray): The state weighting matrix.
            R (numpy.ndarray): The input weighting matrix.

        Returns:
            L (numpy.ndarray): Optimal gain of the LQR solution.
            P (numpy.ndarray):  symmetric postive semidefinite matrix to the discrete-time algebraic Riccati equation.
            E (numpy.ndarray): Eigenvalues of the closed-loop system (A-BL)
        """
        L, _, _ = control.dlqr(self.A, self.B, Q, R) 
       
        return L 
    
    def get_lqr_cost_matrix(self,Q,R):
        """
        Solve the discrete-time algebraic Riccati equation (DARE) for the given system.

        Args:
            Q (numpy.ndarray): The state weighting matrix.
            R (numpy.ndarray): The input weighting matrix.

        Returns:
            P (numpy.ndarray):  symmetric postive semidefinite matrix to the discrete-time algebraic Riccati equation.
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

    Args:
        A (numpy.ndarray): The state matrix.
        B (numpy.ndarray): The input matrix.
        Q (numpy.ndarray): The state weighting matrix.
        R (numpy.ndarray): The input weighting matrix.

    Returns:
        L (numpy.ndarray): Optimal gain of the LQR solution.
        P (numpy.ndarray):  symmetric postive semidefinite matrix to the discrete-time algebraic Riccati equation.
        E (numpy.ndarray): Eigenvalues of the closed-loop system (A-BL)
    """

    L, P, E = control.dlqr(A, B, Q, R)

    return L,P,E



if __name__ == "__main__" :

    sys = LinearSystem(np.array([[1, 1.2], [0, 0.4]]), np.array([[1], [1]]))
    print(sys.A)
    print(sys.B)
    print(sys.C)
    print(sys.D)