import numpy as np

class Constraint:
    def __init__(self, A : np.ndarray , b : np.ndarray, is_hard : bool =True, penalty_weight : float | None = None):
        
        
        self.A       = A
        self.b       = b.flatten()
        self.is_hard = is_hard
        self.penalty_weight = penalty_weight  # Only used if is_hard is False

        if self.penalty_weight < 0:
            raise ValueError("penalty_weight must be non-negative.")

        if len(b) != A.shape[0]:
            raise ValueError("Number of rows in A must match the length of b. A has shape {} and b has length {}.".format(A.shape, len(b)))

    def to_polytope(self):
        """Returns the polytope representation."""
        return self.A, self.b
