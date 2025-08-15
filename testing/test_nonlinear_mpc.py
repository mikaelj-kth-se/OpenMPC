import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from openmpc.models import LinearSystem
from openmpc.mpc import MPC, MPCProblem



# create system matrix

v_ref  = 5
Lp     = 0.18 *v_ref

Iz = 6286
lr = 1.9
lf = 1.27
m  = 1823
Cf = 42000
Cr = 62000


c1 = -(Cf + Cr)
c2 = -(Cf * lf - Cr * lr)
c3 = -(Cf * lf**2 + Cr * lr**2) 


r1 = [v_ref, 0    ,-1           , -Lp              ,  0]
r2 = [0,     0    ,    0        ,  -1              ,  0.]
r3 = [0,     0    , c1/(m*v_ref), c2/(m*v_ref**2)-1.,   Cf/(m*v_ref)]
r4 = [0,     0    , c2/Iz       , c3/(Iz*v_ref)    ,   Cf*lf/(Iz)]
r5 = [0,     0    , 0          , 0                ,  0]

A = np.array([r1, r2, r3, r4, r5])

b1 = [0.,0.]
b2 = [0., 0.]
b3 = [0., 0.]
b4 = [0., 1/Iz]
b5 = [1., 0.]
B = np.array([b1, b2, b3, b4, b5])


print(A)
print(np.real(np.linalg.eig(A)[0]))
system = LinearSystem.c2d(A,B, dt = 0.1)

print(np.real(np.linalg.eig(system.A)[0]))

Q = np.diag([100, 100, 1, 1, 1])
R = np.diag([10, 10])

L = system.get_lqr_controller(Q, R)

x0 = np.array([0.2, -0.1, 0, 0, 0])
n_steps = 100
x = np.zeros((n_steps, system.size_state))
u = np.zeros((n_steps, system.size_input))
x[0] = x0
for i in range(n_steps - 1):
    u[i] = -L @ x[i] 
    x[i + 1] = system.A @ x[i] + system.B @ u[i]


fig, ax = plt.subplots(5,1)
ax[0].plot(x[:, 0])
ax[0].set_xlabel('Time step')
ax[0].set_ylabel(r' $e_{d}$ m')
ax[0].grid()


ax[1].plot(x[:, 1])
ax[1].set_xlabel('Time step')
ax[1].set_ylabel(r' $e_{\psi}$ rad')
ax[1].grid()

ax[2].plot(x[:, 2])
ax[2].set_xlabel('Time step')
ax[2].set_ylabel(r' $v_{y}$ m/s')
ax[2].grid()

ax[3].plot(x[:, 3])
ax[3].set_xlabel('Time step')
ax[3].set_ylabel(r' $\dot{\phi}$ rad/s')
ax[3].grid()

ax[4].plot(x[:, 4])
ax[4].set_xlabel('Time step')
ax[4].set_ylabel(r' $\delta$ rad')
ax[4].grid()





fig, ax = plt.subplots(2,1)
ax[0].plot(u[:, 0])
ax[0].set_xlabel('Time step')
ax[0].set_ylabel(r' $u_{1}$ rad')
ax[0].grid()
ax[1].plot(u[:, 1])
ax[1].set_xlabel('Time step')
ax[1].set_ylabel(r' $u_{2}$ $N \cdot m$')
ax[1].grid()



# Define MPC controller
T = 10  # Prediction horizon

# Create MPC parameters object
mpc_params = MPCProblem(system= system, horizon=T, Q=Q, R=R, QT=np.zeros((5, 5)))


# Add output magnitude constraint on the pitch angle (±20°)
mpc_params.add_output_magnitude_constraint(limit=np.deg2rad(20), output_index=4, is_hard=True)

# extract LQR controller
L = system.get_lqr_controller(Q,R)

# # Add dual-mode with LQR controller
# mpc_params.add_dual_mode(horizon=5, controller=L)

# Create the MPC object
mpc = MPC(mpc_params)


# Simulation settings
num_steps = 20  # Number of simulation steps to cover 5 seconds
x_sim = np.zeros((5, num_steps + 1))  # Store state trajectory
u_sim = np.zeros(num_steps)  # Store control inputs
x_sim[:, 0] = x0  # Set initial state

# Simulate the system
for t in range(num_steps):
    # Get the current state
    current_state = x_sim[:, t]

    # Compute the control action using MPC
    u_t = mpc.get_control_action(current_state)
    u_sim[t] = u_t  # Store the control input
    print(u_t)
    # Apply the control input to the discrete-time system
    x_next = system.A @ current_state + system.B @ u_t.flatten()
    x_sim[:, t + 1] = x_next



plt.tight_layout()
plt.show()
