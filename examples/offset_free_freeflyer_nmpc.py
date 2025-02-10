import numpy as np
import matplotlib.pyplot as plt
from openmpc.NonlinearMPC import NonlinearSystem, trackingMPC, EKF, create_estimator_model
from openmpc.models.atmos_2d import Atmos2D
import time as tim

samplingTime = 0.1
model = Atmos2D(dt=samplingTime)
rhs = model.continuous_dynamics()
states = model.x
inputs = model.u
disturbances = model.d

# Create the NonlinearSystem object
system = NonlinearSystem(updfcn=rhs, states=states, inputs=inputs, disturbances=disturbances, outfcn=states)

# Define discrete-time prediction model
predictionModel = system.c2d(samplingTime)

Q = np.diag([10, 10, 10, 1, 1, 1, 10, 10, 10, 1, 1, 1])
R = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Compute the LQR controller and the corresponding Riccati solution
dnom = np.array([0.00, 0.00, 0.0])
yref = np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0])
(xref, uref) = predictionModel.get_target_point(yref, dnom)
L, P, _ = predictionModel.compute_lqr_controller(Q, R, (xref, uref, dnom))

mpcProblemData = {
    'N': 10,
    'dt': samplingTime,  # sampling time
    'Q': Q,
    'R': R,
    'Q_N': P,
    'predictionModel': predictionModel,
    'umin': np.array([-1.5, -1.5, -1.5, -1.5*0.12, -1.5*0.12, -1.5*0.12]),  # Control limits
    'umax': np.array([1.5, 1.5, 1.5, 1.5*0.12, 1.5*0.12, 1.5*0.12]),  # Control limits
    'slackPenaltyWeight': 1e6,  # Slack penalty weight
    'baseController': L,
    'dualModeHorizon': 5,  # Dual mode horizon
    'dualModeController': L
}

# Initialize the MPC controller
mpc = trackingMPC(mpcProblemData)


simulationModel = NonlinearSystem(updfcn=rhs, states=states, inputs=inputs, disturbances=disturbances)

# Create the simulation model
simulationModel = simulationModel.c2d(samplingTime)

# Estimator
estimatorModel = create_estimator_model(predictionModel)

# Define the initial state and covariance estimates
xe0 = np.array([0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Initial state [\hat{x], \hat{d}]
P0 = 1000 * np.eye(15)  # Initial state covariance

# Define the process noise and measurement noise covariance matrices
Qnoise = np.diag([0.001]*15)  # Process noise covariance
Rnoise = np.diag([0.01]*12)  # Measurement noise covariance

# Pack the EKF parameters into a struct
ekfParameters = {
    'predictionModel': estimatorModel,
    'Q': Qnoise,
    'R': Rnoise,
    'x0': xe0,
    'P0': P0,
    'dt': samplingTime  # Time step
}
ekf = EKF(ekfParameters)

# Time
dt = samplingTime  # Time step (hours)

# Simulation time
Tsim = 20  # Final time (hours)
time = np.arange(0, Tsim, dt)


# Initial state
x0 = xe0[0:12]
d = np.array([0.001, 0.005, 0.0])

# Initialize state and control trajectories for simulation
x_sim = [x0]
u_sim = []

x_hat = xe0
yref = np.array([0.0, 0.0, 0, 0, 0, 0, 0.15, 0.2, 0.1, 0, 0, 0])
n = 12

for k in range(len(time)):
    x_current = x_sim[-1]
    x_hat = ekf.get_state()

    if time[k] > 10:
        d = np.array([0.005, 0.01, 0.0])
        yref = np.array([0.0, 0.1, 0, 0, 0, 0, 0.0, 0.0, 0.2, 0, 0, 0])

    try:
        u_current = mpc.get_control_action(x_hat[0:n], yref, x_hat[n:])
    except Exception as e:
        print(f"Error at simulation step {k}: {e}")
        break

    # Integrate the state using the simulation model
    x_next = simulationModel.updfcn(x_current, u_current, d).full().flatten()

    # Normalize the quaternion
    x_next = model.normalizeQuaternion(x_next)

    # Simulate a measurement
    y_meas = x_next

    # Update the EKF with the new measurement
    ekf.prediction_update(u_current)
    ekf.measurement_update(y_meas, u_current)

    # Store the control action and the next state
    u_sim.append(u_current)
    x_sim.append(x_next)

    # Print status every hour of simulation time
    if (k + 1) % int(1 / dt) == 0:
        sim_time = (k + 1) * dt
        print(f"Simulation time: {sim_time:.2f} hours. Current system state: {x_next}")

# Convert simulation results to numpy arrays for easier manipulation
x_sim_array = np.array(x_sim)
u_sim_array = np.array(u_sim)

# Ensure the simulation result arrays are numpy arrays for easier manipulation
# x_sim_array = np.array(x_sim)
# u_sim_array = np.array(u_sim)

# Time grid for plotting
t_sim = time

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot position
ax1.set_title('Position Control')
ax1.plot(t_sim, x_sim_array[:-1, 0], label='Position x', color='#AA3939', linewidth=2)
ax1.plot(t_sim, x_sim_array[:-1, 1], label='Position y', color='#004791', linewidth=2)
ax1.plot(t_sim, x_sim_array[:-1, 2], label='Position z', color='#007D34', linewidth=2)

# Plot inputs on the second subplot
ax2.step(t_sim, u_sim_array[:, 0], label='u1', color='#AA3939', linewidth=2)
ax2.step(t_sim, u_sim_array[:, 1], label='u2', color='#004791', linewidth=2)
ax2.step(t_sim, u_sim_array[:, 2], label='u3', color='#007D34', linewidth=2)

# Toggle legend
ax1.legend()
ax2.legend()


# New figure for attitude plots
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Plot attitude
ax[0].set_title('Attitude Control')
ax[0].plot(t_sim, x_sim_array[:-1, 6], label='q1', color='#AA3939', linewidth=2)
ax[0].plot(t_sim, x_sim_array[:-1, 7], label='q2', color='#004791', linewidth=2)
ax[0].plot(t_sim, x_sim_array[:-1, 8], label='q3', color='#007D34', linewidth=2)

# Plot torque inputs
ax[1].step(t_sim, u_sim_array[:, 3], label='u4', color='#AA3939', linewidth=2)
ax[1].step(t_sim, u_sim_array[:, 4], label='u5', color='#004791', linewidth=2)
ax[1].step(t_sim, u_sim_array[:, 5], label='u6', color='#007D34', linewidth=2)

# Toggle legend
ax[0].legend()
ax[1].legend()

plt.show()