import numpy as np
import control as ctrl  
from .polytope import Polytope

def pre_set(A, Cx):
    """
    Computes the pre-set of polytope Cx under transformation A.
    """
    return Polytope(Cx.A @ A, Cx.b)

def one_step_controllable_set(A, B, Cx, Cu):
    """
    Computes the one-step controllable set given system matrices A, B,
    state constraints Cx, and control constraints Cu.
    """
    # Ensure correct shapes of b vectors (flatten them if needed)
    Cxb = Cx.b.reshape(-1)  # Make sure Cx.b is a 1D array
    Cub = Cu.b.ravel()  # Flatten Cu.b (if it's a column vector)

    # Block matrix combining state and control constraints
    C = np.block([
        [Cx.A @ A, Cx.A @ B], 
        [np.zeros((len(Cub), A.shape[1])), Cu.A]
    ])

    # Concatenate b vectors
    c = np.concatenate((Cxb, Cub))

    # Project onto the state variables (first dimensions of A)
    x_dims = list(range(A.shape[1]))
    c_poly = Polytope(C, c)
    P = c_poly.projection(x_dims)
    return P

def invariant_set(A, Cx):
    """
    Computes the invariant set under transformation A with constraints Cx.
    """
    converged = False
    S = Cx
    d_idx = 0
    while not converged:
        S_new = S.intersect(pre_set(A, S))
        if S_new == S:
            converged = True
        else:
            d_idx += 1
        S = S_new
    print(f'Converged with determinedness index {d_idx}.\n')
    return S

def control_invariant_set(A, B, Cx, Cu):
    """
    Computes the control-invariant set under dynamics A, B with state constraints Cx and control constraints Cu.
    """
    converged = False
    S = Cx
    d_idx = 0
    while not converged:
        S_new = S.intersect(one_step_controllable_set(A, B, S, Cu))
        if S_new == S:
            converged = True
        else:
            d_idx += 1
        S = S_new
    print(f'Converged with determinedness index {d_idx}.\n')
    return S

def lqr_set(A, B, Q, R, Cx, Cu):
    """
    Computes the LQR-invariant set for system matrices A, B, and weights Q, R.
    """
    # Solve the discrete-time algebraic Riccati equation
    (P, E, L) = ctrl.dare(A, B, Q, R)
    L = np.array(L)

    # Closed-loop system
    C = np.block([[Cx.A], [Cu.A @ (-L)]])
    c = np.concatenate((Cx.b, Cu.b)).reshape(-1, 1)
    
    x_cstr_set_closed_loop = Polytope(C, c)
    A_closed_loop = A - B @ L
    return invariant_set(A_closed_loop, x_cstr_set_closed_loop)

def zero_set(n):
    """
    Creates a zero set of dimension n (near origin).
    """
    eye_mat = np.eye(n)
    Cb = np.ones((n * 2, 1)) * 1e-6
    CA = np.concatenate((eye_mat, -eye_mat), axis=0)
    return Polytope(CA, Cb)

def get_n_step_controllable_set(A, B, Cx, Cu, N, XT):
    """
    Computes the N-step controllable set for system matrices A, B,
    state constraints Cx, control constraints Cu, and terminal set XT.
    """
    S = XT
    r_sets = [S]
    for i in range(N):
        S = one_step_controllable_set(A, B, S, Cu).intersect(Cx)
        r_sets.append(S)
    return r_sets

def is_invariant(C, A):
    """
    Checks if polytope C is invariant under the dynamics x_{t+1}=Ax_t
    """
    vertices = C.get_V_representation()
    for v in vertices:
        if not C.contains(A @ v):
            return False  
    return True
