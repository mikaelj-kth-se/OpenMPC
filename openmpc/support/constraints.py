import numpy as np

class Constraint:
    """
    Simple class to represent polyhedral constraints of the form Hx <= b.
    
    """
    def __init__(self, H : np.ndarray , b : np.ndarray, is_hard : bool =True, penalty_weight : float | None = None):

        """
        Constructor for the Constraint class.

        Define constraints of the form Hx <= b. Penality is given for soft constraints in MPC optimization.
        
        :param H: The matrix H in the constraint Hx <= b.
        :type H: np.ndarray
        :param b: The vector b in the constraint Hx <= b.
        :type b: np.ndarray
        :param is_hard: A boolean indicating whether the constraint is hard or soft.
        :type is_hard: bool
        :param penalty_weight: The penalty weight for the constraint. Only used if is_hard is False.
        :type penalty_weight: float

        """
        
        
        self.H       = H
        self.b       = b.flatten()
        self.is_hard = is_hard
        self.penalty_weight = penalty_weight  # Only used if is_hard is False
        
        if self.penalty_weight is not None:
            if self.penalty_weight < 0:
                raise ValueError("penalty_weight must be non-negative.")

        if len(b) != H.shape[0]:
            raise ValueError("Number of rows in A must match the length of b. A has shape {} and b has length {}.".format(H.shape, len(b)))

    def to_polytope(self):
        """Returns the polytope representation."""
        return self.H, self.b




class TimedConstraint(Constraint):
    """
    Class to represent timed constraints of the form Hx <= b with time dependency.
    
    """
    def __init__(self, H : np.ndarray , b : np.ndarray, start:float, end:float , is_hard : bool =True, penalty_weight : float | None = None):
        
        
        super().__init__(H, b, is_hard, penalty_weight)
        self.start = start
        self.end   = end

        if self.start > self.end: # the equal condition is accepted (corresponding to a singleton)
            raise ValueError("start must be less than end.")