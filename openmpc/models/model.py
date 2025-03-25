import numpy as np
from abc import ABC, abstractmethod


class Model(ABC):
    _counter = 0

    def __init__(self,name : str = None):
        
        if name is not None:
            self._name = name
        else:
            self._name = "Model" + str(Model._counter)
            Model._counter += 1

    @property
    def name(self):
        return self._name
    
    @property
    @abstractmethod
    def dt(self):
        """discrete time step of the model"""
        return NotImplementedError
    
    @property
    @abstractmethod
    def size_input(self):
        return NotImplementedError
    
    @property
    @abstractmethod
    def size_disturbance(self):
        return NotImplementedError
        
    @property
    @abstractmethod
    def size_output(self):
        return NotImplementedError
        
    @property
    @abstractmethod
    def size_state(self):
        return NotImplementedError
            

    @abstractmethod
    def discrete_dynamics(self, x : np.ndarray , u : np.ndarray | None = None, d : np.ndarray | None =None):
        """
        Discrete dynamics :math:`x_{k+1} = f(x_k,u_k,d_k)`

        :param x: state at time k
        :type x: np.ndarray
        :param u: input at time k
        :type u: np.ndarray
        :param d: disturbance at time k
        :type d: np.ndarray
        :return: state at time k+1
        :rtype: np.ndarray
        """

        raise NotImplementedError
    
    @abstractmethod
    def output(self, x:np.ndarray , u : np.ndarray| None = None, d : np.ndarray | None = None):
        """
        Output function :math:`y_k = g(x_k)`

        :param x: State
        :type x: np.ndarray
        :param u: Input
        :type u: np.ndarray
        :param d: Disturbance
        :type d: np.ndarray
        :return: Output
        :rtype: np.ndarray
        """
        raise NotImplementedError
    
    @abstractmethod
    def simulate(self, x0 : np.ndarray, u : np.ndarray | None = None , d : np.ndarray | None = None, steps : int = 10): 
        """
        Simulate the discrete time system for a number of steps. Input should be an array of size (steps, size_input) and disturbance should be an array of size (steps, disturbance_size)
        where steps is the number of steps to simulate.

        :param x0: initial state
        :type x0: np.ndarray
        :param u: input signal
        :type u: np.ndarray
        :param d: disturbance signal
        :type d: np.ndarray
        :param steps: number of steps to simulate
        :type steps: int
        :return: state trajectory
        :rtype: np.ndarray
        """
        raise NotImplementedError
    

    def _check_and_normalise_state(self,x) :
        
        if x is None :
            raise ValueError("State cannot be None.")
        else :
            try :
                x = x.reshape(self.size_state,) # flattening
            except:
                raise ValueError(f"State size mismatch the system state. Expected size is array with dimension compatible with {self.size_state}, given size is {x.shape}.")
        return x
    
    def _check_and_normalise_inputs(self,u,d):
        """
        Check and normalise the input and disturbance signals. If u or d is None, then it is set to zeros of appropriate dimensions. If u or d is an array, then its size is checked and reshaped to the expected size.

        :param u: The input signal to be checked and normalized.
        :type u: np.ndarray or None
        :param d: The disturbance signal to be checked and normalized.
        :type d: np.ndarray or None
        :return: The normalized input and disturbance signals.
        :rtype: tuple of np.ndarray (u, d)
        """
        
        if u is not None:
            try :
                u = u.reshape(self.size_input,)
            except:
                raise ValueError(f"Input signal size mismatch the system input. Expected size is array with dimension compatible with {self.size_input}, given size is {u.shape}.")
        else :
            u = np.zeros(self.size_input)

        if d is not None:
            try :
                d = d.reshape(self.size_disturbance,)
            except:
                raise ValueError(f"Disturbance signal size mismatch the system disturbance. Expected size is array with dimension compatible with {self.size_disturbance}, given size is {d.shape}.")
        else :
            d = np.zeros(self.size_disturbance)
        return u,d


    def check_and_normalise_input_signals(self, u, d ):
        """
        Check and regularize the input triplet (u, d) to the model.

        If d is not None and u is None, then u is set to zero with same length of the signal d. 
        If u is not None and d is None, then d is set to zero with the same length of the signal u.
        If u and d are not None, then they should have the same number of columns, otherwise an error is raised.
        If u and d are None, they are returned as none.

        :param u: The input signal to be checked and regularized.
        :type u: np.ndarray or None
        :param d: The disturbance signal to be checked and regularized.
        :type d: np.ndarray or None
        :return: The regularized input and disturbance signals.
        :rtype: tuple of np.ndarray (u, d)
        """
        

        if u is not None:
            try :
                u = u.reshape(self.size_input,-1)
            except:
                raise ValueError(f"Input signal size mismatch the system input. Expected size is array with dimension compatible with {self.size_input}, given size is {u.shape}.")

        if d is not None:
            try :
                d = d.reshape(self.size_disturbance,-1)
            except:
                raise ValueError(f"Disturbance signal size mismatch the system disturbance. Expected size is array with dimension compatible with {self.size_disturbance}, given size is {d.shape}.")
        
        
        if u is None and d is not None:
            u = np.zeros((self.size_disturbance,u.shape[1]))
        elif u is not None and d is None:
            d = np.zeros((self.size_input,d.shape[0]))
        elif u is not None and d is not None:
            if u.shape[1] != d.shape[1]:
                raise ValueError(f"Input and disturbance signals should have the same number of columns. Given input signal has {u.shape[1]} columns and disturbance signal has {d.shape[1]} columns. Note that your input signal has {u.shape[0]} dimensions and disturbance signal has {d.shape[0]} dimension.")
            
        return u,d


    def __str__(self):
        return self.__class__.__name__ + " : " + self.name
    