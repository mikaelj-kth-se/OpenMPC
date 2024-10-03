class Constraint:
    def __init__(self, A, b, is_hard=True, penalty_weight=None):
        self.A = A
        self.b = b
        self.is_hard = is_hard
        self.penalty_weight = penalty_weight  # Only used if is_hard is False

    def to_polytope(self):
        """Returns the polytope representation."""
        return self.A, self.b
