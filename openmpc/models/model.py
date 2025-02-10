import casadi as ca


class Model(object):

    def __init__(self, dt: float):
        """
        Base class for all models.

        :param dt: sampling time
        :type dt: float
        """
        self.dt = dt
        self.x = None
        self.u = None
        self.d = None

    def continuous_dynamics(self, x, u, d):
        raise NotImplementedError

    def discrete_dynamics(self, x, u, d):
        raise NotImplementedError

    def integrator(self):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__
