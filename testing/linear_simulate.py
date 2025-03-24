from openmpc.models import LinearSystem
import numpy as np
import matplotlib.pyplot as plt

# Define the continuous-time state-space model
A_cont = np.array([
    [-1.2822, 0, 0.98, 0],
    [0, 0, 1, 0],
    [-5.4293, 0, -1.8366, 0],
    [-128.2, 128.2, 0, 0]
])
B_cont = np.array([[-0.3], [0], [-17], [0]])
C_cont = np.array([[0, 1, 0, 0], [0, 0, 0, 1]])
D_cont = np.zeros((2, 1))

h = 0.25
# Discretize the continuous-time model using zero-order hold
system = LinearSystem.c2d(A_cont, B_cont, C_cont, D_cont, h)

# Define the initial state
x0 = np.array([-0.001, 0.0012, 0, 0.3])  # Initial state: descent of 10 meters

# Define MPC parameters
Q = np.eye(4)  # State penalty matrix
R = np.array([[10]])  # Input penalty matrix
T = 10  # Prediction horizon



# Test printing 
print(system)
# test system matrices 
print(system.A)
print(system.B)
print(system.C)
print(system.D)

# Test simulation
u = np.ones((1, T))*0.3
d = np.zeros((1, T))

x_trj,y_trj = system.simulate(x0, u, d)

# Plot the state trajectory
plt.figure()
plt.plot(np.arange(T + 1), x_trj.T)
plt.xlabel('Time step')
plt.ylabel('State')
plt.legend(['$x_1$', '$x_2$', '$x_3$', '$x_4$'])
plt.title('State trajectory')
plt.grid()

# Plot the output trajectory
plt.figure()
plt.plot(np.arange(T), y_trj.T)
plt.xlabel('Time step')
plt.ylabel('Output')
plt.legend(['$y_1$', '$y_2$'])
plt.title('Output trajectory')
plt.grid()



# closed look simulation
# Define the initial state
x0 = np.array([-0.001, 0.0012, 0, 0.3])  # Initial state: descent of 10 meters
L = np.array([[0.1,-0.1,0.1,.3]])  # Kalman gain

# Define the disturbance
d = np.ones((1, T))*0.1
x_trj,y_trj = system.closed_loop_simulate(x0,L, d= d)

# Plot the state trajectory
plt.figure()
plt.plot(np.arange(T + 1), x_trj.T)
plt.xlabel('Time step')
plt.ylabel('State')
plt.legend(['$x_1$', '$x_2$', '$x_3$', '$x_4$'])
plt.title('State trajectory')
plt.grid()

# Plot the output trajectory
plt.figure()
plt.plot(np.arange(T), y_trj.T)
plt.xlabel('Time step')
plt.ylabel('Output')
plt.legend(['$y_1$', '$y_2$'])
plt.title('Output trajectory')
plt.grid()










plt.show()






