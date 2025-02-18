import casadi as ca
import numpy as np
import inspect

def RK(nlsys, dt, order=4):
    """
    Create a Runge-Kutta expression for the given ODE.

    Parameters:
    nlsys (NonlinearSystem): The nonlinear system to integrate.
    dt (float): Time step for the integrator.
    order (int): Order of the Runge-Kutta method (default is 4).

    Returns:
    casadi.MX: CasADi expression for one integration step.
    """
    x = nlsys.states
    u = nlsys.inputs if nlsys.inputs is not None else ca.MX()
    d = nlsys.disturbances if nlsys.disturbances is not None else ca.MX()

    def rk_step(x0, u_val, d_val, h):
        if nlsys.inputs is not None and nlsys.disturbances is not None:
            k1 = nlsys.updfcn(x0, u_val, d_val)
        elif nlsys.inputs is not None:
            k1 = nlsys.updfcn(x0, u_val)
        else:
            k1 = nlsys.updfcn(x0)
        if order == 1:
            return x0 + h * k1
        elif order == 2:
            if nlsys.inputs is not None and nlsys.disturbances is not None:
                k2 = nlsys.updfcn(x0 + 0.5 * h * k1, u_val, d_val)
            elif nlsys.inputs is not None:
                k2 = nlsys.updfcn(x0 + 0.5 * h * k1, u_val)
            else:
                k2 = nlsys.updfcn(x0 + 0.5 * h * k1)
            return x0 + h * k2
        elif order == 3:
            if nlsys.inputs is not None and nlsys.disturbances is not None:
                k2 = nlsys.updfcn(x0 + 0.5 * h * k1, u_val, d_val)
                k3 = nlsys.updfcn(x0 - h * k1 + 2 * h * k2, u_val, d_val)
            elif nlsys.inputs is not None:
                k2 = nlsys.updfcn(x0 + 0.5 * h * k1, u_val)
                k3 = nlsys.updfcn(x0 - h * k1 + 2 * h * k2, u_val)
            else:
                k2 = nlsys.updfcn(x0 + 0.5 * h * k1)
                k3 = nlsys.updfcn(x0 - h * k1 + 2 * h * k2)
            return x0 + (h / 6) * (k1 + 4 * k2 + k3)
        elif order == 4:
            if nlsys.inputs is not None and nlsys.disturbances is not None:
                k2 = nlsys.updfcn(x0 + 0.5 * h * k1, u_val, d_val)
                k3 = nlsys.updfcn(x0 + 0.5 * h * k2, u_val, d_val)
                k4 = nlsys.updfcn(x0 + h * k3, u_val, d_val)
            elif nlsys.inputs is not None:
                k2 = nlsys.updfcn(x0 + 0.5 * h * k1, u_val)
                k3 = nlsys.updfcn(x0 + 0.5 * h * k2, u_val)
                k4 = nlsys.updfcn(x0 + h * k3, u_val)
            else:
                k2 = nlsys.updfcn(x0 + 0.5 * h * k1)
                k3 = nlsys.updfcn(x0 + 0.5 * h * k2)
                k4 = nlsys.updfcn(x0 + h * k3)
            return x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError("Unsupported order. Please choose 1, 2, 3, or 4.")

    rk_step_expr = rk_step(x, u, d, dt)
    return rk_step_expr

def forward_euler(nlsys, dt, steps=1):
    """
    Create a forward-Euler expression for the given ODE.

    Parameters:
    nlsys (NonlinearSystem): The nonlinear system to integrate.
    dt (float): Total integration interval.
    steps (int): Number of forward-Euler steps to perform within the interval dt.

    Returns:
    casadi.MX: CasADi expression for one integration step.
    """
    x = nlsys.states
    u = nlsys.inputs if nlsys.inputs is not None else ca.MX()
    d = nlsys.disturbances if nlsys.disturbances is not None else ca.MX()
    h = dt / steps

    def euler_step(x0, u_val, d_val, h):
        x_current = x0
        for _ in range(steps):
            if nlsys.inputs is not None and nlsys.disturbances is not None:
                x_current = x_current + h * nlsys.updfcn(x_current, u_val, d_val)
            elif nlsys.inputs is not None:
                x_current = x_current + h * nlsys.updfcn(x_current, u_val)
            else:
                x_current = x_current + h * nlsys.updfcn(x_current)
        return x_current

    euler_step_expr = euler_step(x, u, d, h)
    return euler_step_expr

def Integrator(nlsys, integratorType='RK4', dt=0.1, **kwargs):
    if integratorType.startswith('RK'):
        order = int(integratorType[-1]) if len(integratorType) > 2 else 4
        step_expr = RK(nlsys, dt, order=order)
    elif integratorType == 'Euler':
        steps = kwargs.get('steps', 1)
        step_expr = forward_euler(nlsys, dt, steps=steps)
    else:
        raise ValueError("Unsupported integrator type. Please choose 'RK', 'RK2', 'RK3', 'RK4', or 'Euler'.")

    if nlsys.inputs is not None and nlsys.disturbances is not None:
        integrator_func = ca.Function('integrator_func', [nlsys.states, nlsys.inputs, nlsys.disturbances], [step_expr])
        def integrator(x0, u_val=None, d_val=None):
            if u_val is None:
                u_val = ca.DM.zeros(nlsys.inputs.size1())
            if d_val is None:
                d_val = ca.DM.zeros(nlsys.disturbances.size1())
            return integrator_func(x0, u_val, d_val)
    elif nlsys.inputs is not None:
        integrator_func = ca.Function('integrator_func', [nlsys.states, nlsys.inputs], [step_expr])
        def integrator(x0, u_val=None):
            if u_val is None:
                u_val = ca.DM.zeros(nlsys.inputs.size1())
            return integrator_func(x0, u_val)
    else:
        integrator_func = ca.Function('integrator_func', [nlsys.states], [step_expr])
        def integrator(x0):
            return integrator_func(x0)

    return integrator



def simulate_system(integrator, initial_state, N_steps, u_val=None, d_val=None):
    """
    Simulate the system using a given integrator.

    Parameters:
    integrator (Function): CasADi function that performs one integration step.
    initial_state (np.array): Initial state of the system.
    N_steps (int): Number of simulation steps.
    u_val (np.array or None): Control input signal. Can be:
        - None (default): No control input, assumed to be zero.
        - 1D array (shape: (m,)): Constant control input across the simulation horizon.
        - 2D array (shape: (m, N_steps)): Time-varying control input, with each column representing the control input at each step.
    d_val (np.array or None): Disturbance signal. Can be:
        - None (default): No disturbance, assumed to be zero.
        - 1D array (shape: (nd,)): Constant disturbance across the simulation horizon.
        - 2D array (shape: (nd, N_steps)): Time-varying disturbance, with each column representing the disturbance at each step.

    Returns:
    np.array: Simulated state trajectory (shape: (n, N_steps+1)).
    """
    x_sim = [initial_state]

    # Ensure u_val and d_val are in correct format
    if u_val is None:
        u_val = np.zeros((1, N_steps))
    elif u_val.ndim == 1:
        u_val = np.tile(u_val[:, np.newaxis], (1, N_steps))
    if d_val is None:
        d_val = np.zeros((1, N_steps))
    elif d_val.ndim == 1:
        d_val = np.tile(d_val[:, np.newaxis], (1, N_steps))

    num_args = len(inspect.signature(integrator).parameters)

    for k in range(N_steps):
        x_current = x_sim[-1]
        u_current = u_val[:, k]
        d_current = d_val[:, k]
        if num_args == 3:
            x_next = integrator(x_current, u_current, d_current).full().flatten()
        elif num_args == 2:
            x_next = integrator(x_current, u_current).full().flatten()
        else:
            x_next = integrator(x_current).full().flatten()
        x_sim.append(x_next)
    return np.array(x_sim).T
