import control
import numpy as np


class LinearSystem :
    """
    Simple base class to define a linear system in state-space form. The system is assumed to be discrete-time.
    You can use the class method `c2d` obtain a discerete system from continuous time matrices
    """

    def __init__(self, A, B, C = None, D = None):
        """
        Initialize linear system

        Args:
            Ad (numpy.ndarray): The state matrix.
            Bd (numpy.ndarray): The input matrix.
            Cd (numpy.ndarray): The output matrix.
            Dd (numpy.ndarray): The feedforward matrix.
        """

        self._A = A
        self._B = B
        
        if C is None:
            self._C = np.eye(A.shape[0])
        if D is None:
            self._D = np.zeros((A.shape[0], B.shape[1]))

        self._C = C
        self._D = D
    
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
    def size_in(self):
        return self._B.shape[1]
    @property
    def size_out(self):
        return self._C.shape[0]
    @property
    def size_state(self):
        return self._A.shape[0]


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
        h = 0.25
        sys_disc = control.c2d(sys_cont, h)

        # Extract the discrete-time matrices
        Adist, Bdist, Cdist, Ddist = control.ssdata(sys_disc)
        
        return LinearSystem(Adist, Bdist, Cdist, Ddist)


    @staticmethod
    def get_lqr_solution(A, B, Q, R):
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

    def lqr_solution(self,Q,R):
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

        return self.get_lqr_solution(self._A, self._B, Q, R)

    def open_loop_simulation(self, x0, u=None):
        """Simulate the LinearSystem.

        Parameters:
            x0 (np.array): Initial state vector. Defaults to zero vector of appropriate dimension.
            u (np.array, optional): Input signal. should be an array of dimension (n_input, steps). The system will be simulated for step+1 points . Defaults to zero.

        Returns:
            x_trj (np.array): Simulated state trajectory.
            y_trj (np.array): Simulated output trajectory.
        """

        x0    = np.array(x0).reshpe(self.size_state,1)
        steps = 10
        
        if u is None: # default ten steps input
            u = np.zeros((self.size_in, steps))
        else : 
            if u.shape[0] != self.size_in:
                raise ValueError("input signal does not match system input size, expected size is {}".format(self.size_in))
            else :
                steps = u.shape[1]


        x_trj = np.empty((self.size_state, steps+1))
        x_trj[:, 0] = x0.flatten() # initial state condition

        for k in range(steps):

            x_next = self._A @ x_trj[:, k] + self._B @ u[:, k] 
            y_next = self._C @ x_trj[:, k] + self._D @ u[:, k]

            x_trj[k+1] = x_next.flatten()
            y_trj[k]   = y_next.flatten()

        return x_trj, y_trj 
    

    def closed_loop_simulate(self, x0, L, steps=10):
        """Simulate the LinearSystem in closed-loop with an L gain (a.k.a A-BL).

        Parameters:
            x0 (np.array): Initial state vector. Defaults to zero vector of appropriate dimension.
            L (np.array): L gain matrix.
            steps (int): Number of steps to simulate.

        Returns:
            x_trj (np.array): Simulated state trajectory.
            y_trj (np.array): Simulated output trajectory.
        """

        x0    = np.array(x0).reshpe(self.size_state,1)
        steps = 10

        if L.shape != (self.size_in, self.size_state):
            raise ValueError("L gain matrix does not match the system dimensions. Expected shape is ({}, {})".format(self.size_in, self.size_state))

        u_trj = np.empty((self.size_in, steps-1))
        x_trj = np.empty((self.size_state, steps))
        x_trj[:, 0] = x0.flatten() # initial state condition

        for k in range(steps - 1):

            x_next = (self._A - self._B@L) @ x_trj[:, k]
            y_next = (self._C + self._D @L) x_trj[:, k]

            x_trj[k+1] = x_next.flatten()
            y_trj[k]   = y_next.flatten()

        return x_trj, y_trj 
        
        
