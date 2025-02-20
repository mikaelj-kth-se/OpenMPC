
from openmpc.models import LinearSystem, NonlinearSystem
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca


# Define the model parameters
C_hot = 5000  # Heat capacity of hot fluid (J/K)
C_cold = 5000  # Heat capacity of cold fluid (J/K)
c_p_hot = 2000  # Specific heat capacity of hot fluid (J/kg·K)
c_p_cold = 4184  # Specific heat capacity of cold fluid (J/kg·K)
U = 500  # Overall heat transfer coefficient (W/m²*K)
A = 50  # Heat transfer area (m²)

# Define the system states and control inputs
T_hot = ca.MX.sym('T_hot')
T_hot_in = ca.MX.sym('T_hot_in')
T_cold = ca.MX.sym('T_cold')
T_cold_in = ca.MX.sym('T_cold_in')
m_dot_hot = ca.MX.sym('m_dot_hot')
m_dot_cold = ca.MX.sym('m_dot_cold')

# Define prediction model
samplingTime = 0.1

dT_hot = (1 / C_hot) * (m_dot_hot * c_p_hot * (T_hot_in - T_hot) - U * A * (T_hot - T_cold))
dT_cold = (1 / C_cold) * (m_dot_cold * c_p_cold * (T_cold_in - T_cold) + U * A * (T_hot - T_cold))

ode          = ca.vertcat(dT_hot, dT_cold)
states       = ca.vertcat(T_hot, T_cold)
inputs       = ca.vertcat(m_dot_hot, m_dot_cold)
disturbances = ca.vertcat(T_hot_in, T_cold_in)


nonlinear_system = NonlinearSystem.c2d(ode, states, inputs, disturbances, dt=samplingTime)
print(nonlinear_system)



# # Create the NonlinearSystem object
# heatExchangerSystem = NonlinearSystem(updfcn=rhs, states=states, inputs=inputs, disturbances=disturbances, outfcn=states)

