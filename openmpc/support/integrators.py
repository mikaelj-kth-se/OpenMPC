import casadi as ca
import numpy as np


def RK(updfcn, states, inputs = None ,disturbances = None, dt = 1, order=4):
    """
    Create a Runge-Kutta expression for the given ODE. Note that the expression in the output will depend only on the given 
    states, inputs, and disturbances. So if the input is not given , then the output expression will not contain any input. 
    The same holds for the disturbances. The state must always be given.

    :param updfcn: The function that defines the ODE.
    :type updfcn: casadi.MX,casadi.SX
    :param states: The states of the system.
    :type states: casadi.MX,casadi.SX
    :param inputs: The inputs of the system (default is None).
    :type inputs: casadi.MX,casadi.SX
    :param disturbances: The disturbances of the system (default is None).
    :type disturbances: casadi.MX,casadi.SX
    :param dt: Time step for the integrator.
    :type dt: float
    :param order: Order of the Runge-Kutta method (default is 4).
    :type order: int
    :return: CasADi expression for one integration step.
    :rtype: casadi.MX
    """
    x = states
    u = inputs if inputs is not None else ca.MX.sym('u')
    d = disturbances if disturbances is not None else ca.MX.sym('d')

    updfcn = ca.Function('updfcn', [x, u, d], [updfcn])

    def rk_step(x0, u_val, d_val, h):
       
        k1 = updfcn(x0, u_val, d_val)

        if order == 1:
            return x0 + h * k1
        
        elif order == 2:
            k2 = updfcn(x0 + 0.5 * h * k1, u_val, d_val)
            return x0 + h * k2
        
        elif order == 3:
            k2 = updfcn(x0 + 0.5 * h * k1, u_val, d_val)
            k3 = updfcn(x0 - h * k1 + 2 * h * k2, u_val, d_val)
            
            return x0 + (h / 6) * (k1 + 4 * k2 + k3)
        
        elif order == 4:
           
            k2 = updfcn(x0 + 0.5 * h * k1, u_val, d_val)
            k3 = updfcn(x0 + 0.5 * h * k2, u_val, d_val)
            k4 = updfcn(x0 + h * k3, u_val, d_val)

            return x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError("Unsupported order. Please choose 1, 2, 3, or 4.")

    if disturbances is None and inputs is None:
        rk_step_expr = rk_step(x, ca.DM.zeros(1), ca.DM.zeros(1), dt)
    elif disturbances is None:
        rk_step_expr = rk_step(x, u, ca.DM.zeros(1), dt)
    elif inputs is None:
        rk_step_expr = rk_step(x, ca.DM.zeros(1), d, dt)
    else:
        rk_step_expr = rk_step(x, u, d, dt)

    return rk_step_expr

def forward_euler(updfcn, states, inputs = None ,disturbances = None, dt = 1, steps=1):
    """
    Create a forward-Euler expression for the given ODE. Note that the expression in the output will depend only on the given 
    states, inputs, and disturbances. So if the input is not given , then the output expression will not contain any input. 
    The same holds for the disturbances. The state must always be given.

    :param updfcn: The function that defines the ODE.
    :type updfcn: casadi.MX,casadi.SX
    :param states: The states of the system.
    :type states: casadi.MX,casadi.SX
    :param inputs: The inputs of the system (default is None).
    :type inputs: casadi.MX,casadi.SX
    :param disturbances: The disturbances of the system (default is None).
    :type disturbances: casadi.MX,casadi.SX
    :param dt: Time step for the integrator.
    :type dt: float
    :param steps: Number of integration steps (default is 1).
    :type steps: int
    :return: CasADi expression for one integration step using euler method.
    :rtype: casadi.MX
    """
    x = states
    u = inputs if inputs is not None else ca.MX.sym('u')
    d = disturbances if disturbances is not None else ca.MX.sym('d')

    updfcn = ca.Function('updfcn', [x, u, d], [updfcn])

    h = dt / steps

    def euler_step(x0, u_val, d_val, h):
        x_current = x0
        for _ in range(steps):
            x_current = x_current + h * updfcn(x_current, u_val, d_val)
        return x_current
    

    if disturbances is None and inputs is None:
        euler_step_expr = euler_step(x, ca.DM.zeros(1), ca.DM.zeros(1), h)
    elif disturbances is None:
        euler_step_expr = euler_step(x, u, ca.DM.zeros(1), h)
    elif inputs is None:
        euler_step_expr = euler_step(x, ca.DM.zeros(1), d, h)
    else:
        euler_step_expr = euler_step(x, u, d, h)

    return euler_step_expr


def Integrator(updfcn, states, inputs = None ,disturbances = None, integratorType='RK4', dt=0.1, **kwargs):

    if integratorType.startswith('RK'):
        order = int(integratorType[-1]) if len(integratorType) > 2 else 4
        step_expr = RK(updfcn, states, inputs ,disturbances, dt, order=order)
    elif integratorType == 'Euler':
        steps = kwargs.get('steps', 1)
        step_expr = forward_euler(updfcn, states, inputs, disturbances , dt, steps=steps)
    else:
        raise ValueError("Unsupported integrator type. Please choose 'RK', 'RK2', 'RK3', 'RK4', or 'Euler'.")

    x = states
    u = inputs if inputs is not None else ca.MX.sym('u', 1)
    d = disturbances if disturbances is not None else ca.MX.sym('d', 1)


    integrator_func = ca.Function('integrator_func', [x,u,d], [step_expr])
        
    def integrator(x0, u_val=None, d_val=None):
        if u_val is None:
            u_val = ca.DM.zeros(1)
        if d_val is None:
            d_val = ca.DM.zeros(1)

        return np.asarray(integrator_func(x0, u_val, d_val)).flatten()

    return integrator


# def simulate_system(integrator, initial_state, N_steps, u_val=None, d_val=None):
#     """
#     Simulate the system using a given integrator.

#     Parameters:
#     integrator (Function): CasADi function that performs one integration step.
#     initial_state (np.array): Initial state of the system.
#     N_steps (int): Number of simulation steps.
#     u_val (np.array or None): Control input signal. Can be:
#         - None (default): No control input, assumed to be zero.
#         - 1D array (shape: (m,)): Constant control input across the simulation horizon.
#         - 2D array (shape: (m, N_steps)): Time-varying control input, with each column representing the control input at each step.
#     d_val (np.array or None): Disturbance signal. Can be:
#         - None (default): No disturbance, assumed to be zero.
#         - 1D array (shape: (nd,)): Constant disturbance across the simulation horizon.
#         - 2D array (shape: (nd, N_steps)): Time-varying disturbance, with each column representing the disturbance at each step.

#     Returns:
#     np.array: Simulated state trajectory (shape: (n, N_steps+1)).
#     """
#     x_sim = [initial_state]

#     # Ensure u_val and d_val are in correct format
#     if u_val is None:
#         u_val = np.zeros((1, N_steps))
#     elif u_val.ndim == 1:
#         u_val = np.tile(u_val[:, np.newaxis], (1, N_steps))
#     if d_val is None:
#         d_val = np.zeros((1, N_steps))
#     elif d_val.ndim == 1:
#         d_val = np.tile(d_val[:, np.newaxis], (1, N_steps))

#     num_args = len(inspect.signature(integrator).parameters)

#     for k in range(N_steps):
#         x_current = x_sim[-1]
#         u_current = u_val[:, k]
#         d_current = d_val[:, k]
#         if num_args == 3:
#             x_next = integrator(x_current, u_current, d_current).full().flatten()
#         elif num_args == 2:
#             x_next = integrator(x_current, u_current).full().flatten()
#         else:
#             x_next = integrator(x_current).full().flatten()
#         x_sim.append(x_next)
#     return np.array(x_sim).T
