import casadi as cs
import numpy as np
from openmpc.models import NonlinearSystem
from openmpc.support.integrators import RK


def Qmat(q):
    qx, qy, qz = q[0], q[1], q[2]
    qw = cs.sqrt(1 - qx ** 2 - qy ** 2 - qz ** 2)
    return cs.vertcat(cs.horzcat(qw, -qz, qy),
                      cs.horzcat(qz, qw, -qx),
                      cs.horzcat(-qy, qx, qw))


def q_to_rot_mat(q):
    qx, qy, qz = q[0], q[1], q[2]
    qw = cs.sqrt(1 - qx ** 2 - qy ** 2 - qz ** 2)

    rot_mat = cs.vertcat(
        cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
        cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))
    return rot_mat


def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    return cs.mtimes(rot_mat, v)



# set up states & controls
p      = cs.MX.sym('p', 3)  # position
v      = cs.MX.sym('v', 3)  # velocity
q      = cs.MX.sym('q', 3)  # quaternion vector
w      = cs.MX.sym('w', 3)  # angular velocity

x = cs.vertcat(p, v, q, w)
u = cs.MX.sym('u', 6)  # control input
d = cs.MX.sym('d', 6)  # disturbance

# parameters
torque_arm_length = 0.12
mass = 16.8
inertia = cs.diag(cs.DM([[0.258, 0.258, 0.258]]))

# Grab states used in the propagation
v = x[3:6]
q = x[6:9]
w = x[9:12]

F = u[:3]
tau = u[3:]

d_trans = d[:3]
d_rot = d[3:]

# dynamics
xdot = cs.vertcat(v,
                    v_dot_q(F, q)/mass + d_trans,
                    1 / 2 * Qmat(q) @ w,
                    np.linalg.inv(inertia) @ (tau - cs.cross(w, inertia @ w)) + d_rot
                    )





class Atmos2D(NonlinearSystem):

    def __init__(self, dt: float):
        """
        Linearizable 3D model of a freeflyer with vector-only quaternion dynamics.

        Follows:
        Vyas, Shubham, Bilal Wehbe, and Shivesh Kumar. "Quaternion based LQR for
        Free-Floating Robots without Gravity." CEAS EuroGNC 2022. 2022.

        :param dt: sampling time
        :type dt: float
        """

        # set up states & controls
        p      = cs.MX.sym('p', 3)  # position
        v      = cs.MX.sym('v', 3)  # velocity
        q      = cs.MX.sym('q', 3)  # quaternion vector
        w      = cs.MX.sym('w', 3)  # angular velocity

        self.x = cs.vertcat(p, v, q, w)
        self.u = cs.MX.sym('u', 6)  # control input
        self.d = cs.MX.sym('d', 6)  # disturbance

        # parameters
        self.torque_arm_length = 0.12
        self.mass = 16.8
        self.inertia = cs.diag(cs.DM([[0.258, 0.258, 0.258]]))


        x_next = RK(self.continuous_dynamics(), self.x, self.u, self.d, dt, order=4)
        super().__init__(x_next, self.x, self.u, self.d, dt = dt)

        

    def normalizeQuaternion(self, x):
        # Get quat vector
        q = x[6:9]

        # Get full quat
        q0 = cs.sqrt(1 - q[0] ** 2 - q[1] ** 2 - q[2] ** 2)
        q = cs.vertcat(q0, q)

        # Normalize
        q = q / cs.norm_2(q)

        # Return vector in-place of x
        # x[6:9] = q[1:]
        x[6] = q[1]
        x[7] = q[2]
        x[8] = q[3]
        return x


    def continuous_dynamics(self, x=None, u=None, d=None):

        if x is None:
            x = self.x
        if u is None:
            u = self.u
        if d is None:
            d = self.d

        # Grab states used in the propagation
        v = x[3:6]
        q = x[6:9]
        w = x[9:12]

        F = u[:3]
        tau = u[3:]

        d_trans = d[:3]
        d_rot = d[3:]

        # dynamics
        xdot = cs.vertcat(v,
                          v_dot_q(F, q)/self.mass + d_trans,
                          1 / 2 * Qmat(q) @ w,
                          np.linalg.inv(self.inertia) @ (tau - cs.cross(w, self.inertia @ w)) + d_rot
                          )
        return xdot
    


if __name__ == '__main__':
    model = Atmos2D(0.01)
    print(model)