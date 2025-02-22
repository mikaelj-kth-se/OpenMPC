import casadi as ca
import numpy as np

from  openmpc.support.integrators import RK
from  openmpc.mpc.parameters import MPCProblem
from openmpc.invariant_sets import Polytope
from openmpc.support import Constraint
from openmpc.models import NonlinearSystem

class NMPC:
    def __init__(self, mpc_params  : MPCProblem):
        """
        Initialize the MPC controller with the given problem data.

        Parameters:
        mpcProblemData (MPCParameters): Contains the prediction horizon (N), sampling time (dt),
                                        cost matrices Q, R, QT, prediction model, control limits,
                                        state limits, output function, output limits, 
                                        baseController (feedback gain L), and slack penalty weight.

        """

        # Extract MPC parameters
        self.params               = mpc_params
        self.system               : NonlinearSystem = self.params.system

        self.n                    : int = self.system.size_state
        self.m                    : int = self.system.size_input
        self.N                    : int = self.params.horizon
        self.Q                    : np.ndarray = self.params.Q
        self.R                    : np.ndarray = self.params.R
        self.QT                   : np.ndarray = self.params.QT
        
        self.u_constraints         : list[Constraint] = self.params.u_constraints
        self.x_constraints         : list[Constraint] = self.params.x_constraints
        self.y_constraints         : list[Constraint] = self.params.y_constraints
        self.terminal_constraints  : Constraint       = self.params.terminal_constraints
        self.global_penalty_weight : float            = self.params.global_penalty_weight
        
        self.solver               : str        = self.params.solver
        self.slack_penalty        : float      = self.params.slack_penalty
        self.terminal_set         : Polytope   = self.params.terminal_set
        self.dual_mode_controller : np.ndarray = self.params.dual_mode_controller
        self.dual_mode_horizon    : int        = self.params.dual_mode_horizon

        self.L              : np.ndarray = self.params.reference_controller
        self.Ldm               : np.ndarray = self.params.dual_mode_controller
        self.Ndm               : int        = self.params.dual_mode_horizon
        
        self.opti = ca.Opti()

        # Define optimization variables
        self.x  = self.opti.variable(self.n, self.N + self.Ndm + 1)
        self.v  = self.opti.variable(self.m, self.N)
        self.x0 = self.opti.parameter(self.n)  # Initial state parameter


        self.previous_X     : np.ndarray | None = None
        self.previous_V     : np.ndarray | None = None
        self.mpc_controller : ca.Function = None

        self._setup_problem()

    
    def _setup_problem(self):        
        """
        Formulate the optimization problem and create the function for optimal control.
        """

        f = self.system.updfcn
        g = self.system.outfcn

        # remove distrurbances from the dynamics
        x_var = ca.MX.sym('x', self.n)
        u_var = ca.MX.sym('u', self.m)
        d_zero = np.zeros((self.system.size_disturbance,))

        self.f = ca.Function('f', [x_var, u_var], [f(x_var, u_var, d_zero)])
        self.g = ca.Function('g', [x_var, u_var], [g(x_var, u_var, d_zero)])
        
        self.cost        = 0
        slack_variables  = []  # To collect slack variables for soft constraints

        self.constraints = [self.x[:, 0] == self.x0] # initial state constraints


        # Main MPC horizon loop
        for t in range(self.N):
            
            x_k = self.x[:, t]
            v_k = self.v[:, t]
            u_k = -self.L @ x_k + v_k
            self.cost += ca.mtimes([x_k.T, self.Q, x_k]) + ca.mtimes([u_k.T, self.R, u_k])

            # System dynamics
            self.constraints += [self.x[:, t + 1] == self.f(x_k, u_k)]
            
            # Add input constraints
            for constraint in self.u_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @u_k <= b]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = self.opti.variable()
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @u_k <= b + np.ones(H.shape[0]) * slack]

            # Add state constraints 
            for constraint in self.x_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @x_k <= b]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = self.opti.variable()
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @x_k <= b + np.ones(H.shape[0]) * slack]

            # Add output constraints
            for constraint in self.y_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.g(x_k,u_k)  <= b]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = self.opti.variable()
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @self.g(x_k,u_k) <= b + np.ones(H.shape[0]) * slack]


        # Dual mode implementation
        if self.dual_mode_horizon != 0 :
            # Predict states using the dual mode controller beyond the main horizon
            x_dual = self.x[:, self.N]  # Initial state for the dual mode phase
            
            for t in range(self.dual_mode_horizon):
                # Compute control using the dual mode controller
                u_dual = -self.dual_mode_controller @ x_dual

                # Add state update for the dual mode
                x_next = self.x[:, self.N + t + 1]
                self.constraints += [x_next == self.f(x_dual, u_dual)]
                x_dual = x_next

                # Add state and input constraints during dual mode
                for constraint in self.x_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [H @x_dual <= b]

                for constraint in self.u_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [H @u_dual <= b]
                
                for constraint in self.y_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [H @self.g(x_dual, u_dual) <= b]

            # Apply terminal cost and constraints at the end of the dual mode horizon
            self.cost += ca.mtimes([x_dual.T, self.QT, x_dual])
            
            if self.terminal_set:
                A_terminal, b_terminal = self.terminal_set.A, self.terminal_set.b
                self.constraints += [A_terminal @ x_dual <= b_terminal]
        else:
            # Apply terminal cost and constraints at the end of the main horizon (if no dual mode is specified)
            
            
            self.cost += ca.mtimes([self.x[:, self.N].T, self.QT, self.x[:, self.N]])
            if self.terminal_set:
                A_terminal, b_terminal = self.terminal_set.A, self.terminal_set.b
                self.constraints += [A_terminal @ self.x[:, self.N] <= b_terminal]

        # Add slack penalties to the cost function
        for slack, penalty_weight in slack_variables:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.global_penalty_weight * penalty_weight * slack  # Linear penalty
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.global_penalty_weight * penalty_weight * slack**2  # Quadratic penalty
            else :
                raise ValueError("Invalid slack penalty type. Must be 'LINEAR' or 'SQUARE'.")

        

        for constraint in self.constraints:
            self.opti.subject_to(constraint)
        
        self.opti.minimize(self.cost)

        # set solver options
        opts = {'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.tol': 1e-3,
                'expand': 1,
                'error_on_fail':1,                              # to guarantee transparency if solver fails
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.warm_start_bound_push': 1e-8,
                'ipopt.warm_start_mult_bound_push': 1e-8,
                'ipopt.mu_init': 1e-5,
                'ipopt.bound_relax_factor': 1e-9}

        self.opti.solver('ipopt', opts)
        
        # Now, convert the Opti object to a function
        self.mpc_controller = self.opti.to_function('MPCPlanner', [self.x0, self.x, self.v], [self.x, self.v], ['initial_state', 'x_guess', 'v_guess'], ['x_opt', 'u_opt'])


        
    def compute_predicted_optimal_controls(self, x0):
        """
        Compute the optimal control and state trajectories for the given initial state.

        Parameters:
        x0 (np.array): Initial state.

        Returns:
        (np.array, np.array): Optimal control trajectory and optimal state trajectory.
        """
        
        
        if self.previous_X is not None and self.previous_V is not None:

            # Prepare the initial guess :
            # 1) Take all the elements of the previus solution except the first one
            X_initial  = self.previous_X[:,1:]

            # 2) Add a new last state guess computed from the nomial controller as x_N_guess = f(x_N-1, u_N-1) with u_N-1 = -L @ x_N-1 + v_N-1
            xx          = self.previous_X[:, -1]                                    
            uu          = - self.Ldm @ xx if self.Ndm>0 else -self.L @ xx
            x_last      = self.f(xx, uu).full()

            X_initial = np.hstack((X_initial,x_last))     
            
            # 3) set initial state of the guess as the one you measure
            X_initial[:,0] = x0

            # 4) for the control input guess, take all the elements of the previous solution except the last one which is set to zero
            V_initial        = np.hstack((self.previous_V[:, 1:], np.zeros((self.m, 1))))


        else:
            # Initial guess for self.v: zero matrix
            V_initial = np.zeros((self.m, self.N))

            # Initial guess for self.x: linear interpolation from x0 to the origin
            X_initial = np.zeros((self.n, self.N + self.Ndm+ 1))
            for i in range(self.n):
                X_initial[i, :] = np.linspace(x0[i], 0, self.N + self.Ndm + 1)

        try:
            (x_opt, v_opt) = self.mpc_controller(x0, X_initial, V_initial)
        except Exception as e:
            print(f"Solver failed: {e}")
            

        # Extract the control and state trajectories
        u_opt = -self.L @ x_opt[:, :self.N] + v_opt
                
        # Store the solution for the next time step
        self.previous_X = x_opt
        self.previous_V = v_opt

        return u_opt.full(), x_opt.full()

    def get_control_action(self, x):
        """
        MPC controller that plans optimal controls and states over the prediction horizon
        and returns the first control action in the predicted optimal sequence.

        Parameters:
        x (np.array): Current state.

        Returns:
        np.array: The first control action in the predicted optimal sequence.
        """
        # Plan optimal controls and states over the next N samples
        uPred, xPred = self.compute_predicted_optimal_controls(x)

        # Apply the first control action in the predicted optimal sequence
        return uPred[:, 0]



class SetPointTrackingNMPC:
    """

    Class for set-point output tracking Nonlinear MPC (NMPC)

    Problem formulation :

    min_{x,u}  sum_{k=0}^{N-1} (x_k - x_ref)^T Q (x_k - x_ref) + (u_k - u_ref)^T R (u_k - u_ref) 

    s.t.       
               x_{k+1} = f(x_k, u_k, d_k)
               y_k     = g(x_k, u_k, d_k)
               u_k     = -L @ (x_k - x_ref) + u_ref + v_k
                
               u_k \in U, x_k \in X, y_k \in Y
               x_0 = x
                
    
    where : 
    x is the state, 
    u is the control input, 
    d is the disturbance, 
    y is the output
    f, g are the system dynamics and output functions, respectively.


    The control input is assumed to be of the form u_k = -L(x_k - x_ref) + u_ref + v_k where L is a reference the feedback gain (for example an LQR computer over the linearised dynamics), 
    x_ref is the reference state, u_ref is the reference input, and v_k is the control deviation term (which is optimised by the NMPC). By default L = 0 byt it but it could be set to an 
    LQR controller for example.
     
    The system takes a reference output point y_ref from which a state and input u_ref,x_ref are computed such that f(x_ref, u_ref, d) = 0 and g(x_ref, u_ref, d) = y_ref. 
    The solution with minimum norm control input is selected among all the possible solutions.

    The disturbance acting on the system is a parameter of the optimization.
    """

    def __init__(self, mpc_params : MPCProblem):


        """
        SetPointTrackingNMPC constructor

        :param mpc_params: MPCParameters object containing the MPC problem data.
        :type mpc_params: MPCParameters
        
        """
        
      

        # Extract MPC parameters
        self.params               : MPCProblem   = mpc_params
        self.system               : NonlinearSystem = self.params.system

        self.n                    : int        = self.system.size_state
        self.m                    : int        = self.system.size_input
        self.N                    : int        = self.params.horizon
        self.Q                    : np.ndarray = self.params.Q
        self.R                    : np.ndarray = self.params.R
        self.QT                   : np.ndarray = self.params.QT
        
        self.u_constraints         : list[Constraint] = self.params.u_constraints
        self.x_constraints         : list[Constraint] = self.params.x_constraints
        self.y_constraints         : list[Constraint] = self.params.y_constraints
        self.terminal_constraints  : Constraint       = self.params.terminal_constraints
        self.global_penalty_weight : float            = self.params.global_penalty_weight
        
        self.solver               : str        = self.params.solver
        self.slack_penalty        : float      = self.params.slack_penalty
        self.terminal_set         : Polytope   = self.params.terminal_set

        self.L                 : np.ndarray = self.params.reference_controller  # reference linear feedback gain (default to zero)
        self.Ldm               : np.ndarray = self.params.dual_mode_controller  # dual mode controller (default to zero)
        self.Ndm               : int        = self.params.dual_mode_horizon     # dual mode horizon (default to zero)   
        
        
        self.opti = ca.Opti()

        # Tracking reference and siturbance parameters
        self.r       = self.opti.parameter(self.system.size_output)       # reference output
        self.d       = self.opti.parameter(self.system.size_disturbance)  # Disturbance parameter
        self.x_ref   = self.opti.parameter(self.n)                        # reference state s.t.  f(x_ref, u_ref, d) = 0 and g(x_ref, u_ref, d) = y_ref
        self.u_ref   = self.opti.parameter(self.m)                        # reference input s.t.  f(x_ref, u_ref, d) = 0 and g(x_ref, u_ref, d) = y_ref
        self.x0      = self.opti.parameter(self.n)                        # Initial state parameter
        
        # state and input variables
        self.x       = self.opti.variable(self.n, self.N + self.Ndm+ 1)
        self.v       = self.opti.variable(self.m, self.N)

        self.reference_value   : np.ndarray = np.zeros(self.system.size_output)       # Reference output set-point
        self.disturbance_value : np.ndarray = np.zeros(self.system.size_disturbance)  # Disturbance value
        
        # Option for soft tracking constraint from mpc_params
        self.soft_tracking           : bool  = self.params.soft_tracking
        self.tracking_penalty_weight : float = self.params.tracking_penalty_weight
        
        # Define the slack variable if soft tracking is enabled
        if self.soft_tracking:
            self.slack_tracking = self.opti.variable(self.system.size_output)  # Slack variable
        else:
            self.slack_tracking = None
        

        self.previous_X : np.ndarray | None = None
        self.previous_V : np.ndarray | None = None

        self._setup_problem()

    
    def set_reference(self, reference : np.ndarray):
        """
        Set the reference for tracking.

        :param reference: The reference trajectory.
        :type reference: np.ndarray
        """
        self.reference_value = np.reshape(reference, (self.system.size_output,))

    def set_disturbance(self, disturbance : np.ndarray):
        """
        Set the disturbance parameter if disturbances are defined (known disturbance entering the model).

        :param disturbance: The disturbance.
        :type disturbance: np.ndarray
        """
        self.disturbance_value = np.reshape(disturbance, (self.system.size_disturbance,))


    def _setup_problem(self):        
        """
        Formulate the optimization problem and create the function for optimal control.
        """

    
        self.f = self.system.updfcn  # system dynamics x_next = f(x, u, d)
        self.g = self.system.outfcn  # output function y = g(x, u, d)

        self.cost        = 0   # mpc cost
        slack_variables  = []  # To collect slack variables for soft constraints

        self.constraints = [(self.x[:, 0] == self.x0,"initial_constraint")] # initial state constraints


        # Main MPC horizon loop
        for t in range(self.N):
            
            x_k = self.x[:, t]
            v_k = self.v[:, t]
            u_k = - self.L @ (x_k - self.x_ref) + self.u_ref + v_k # reference feedback input + deviation 
            self.cost += ca.mtimes([(x_k - self.x_ref).T, self.Q, (x_k - self.x_ref)]) + ca.mtimes([(u_k - self.u_ref).T, self.R, (u_k - self.u_ref)])

            # System dynamics
            self.constraints += [(self.x[:, t + 1] == self.f(x_k, u_k, self.d),"dynamics_constraint")]
            
            # Add input constraints
            for constraint in self.u_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [(H @u_k <= b,"input_constraint")]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = self.opti.variable()
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [(H @u_k <= b + np.ones(H.shape[0]) * slack,"input_constraint")]

            # Add state constraints 
            for constraint in self.x_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [(H @x_k <= b,"state_constraint")]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = self.opti.variable()
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [(H @x_k <= b + np.ones(H.shape[0]) * slack,"state_constraint")]

            # Add output constraints
            for constraint in self.y_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [(H @self.g(x_k,u_k,self.d)  <= b,"output_constraint")]
                else:
                    # Single slack variable for all inequalities in this constraint
                    slack = self.opti.variable()
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [(H @self.g(x_k,u_k,self.d) <= b + np.ones(H.shape[0]) * slack,"output_constraint")]


        # Dual mode implementation
        if self.Ndm != 0 :
            # Predict states using the dual mode controller beyond the main horizon
            x_dual = self.x[:, self.N]  # Initial state for the dual mode phase
            
            for t in range(self.Ndm):
                # Compute control using the dual mode controller
                u_dual = - self.Ldm @ (x_dual- self.x_ref) + self.u_ref
                
                # Add state update for the dual mode
                x_next           = self.x[:, self.N + t + 1]
                self.constraints += [(x_next == self.f(x_dual, u_dual,self.d),"dual_mode_dynamics_constraint")]
                x_dual           = x_next

                # Add state and input constraints during dual mode
                for constraint in self.x_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [(H @x_dual <= b,"dual_mode_state_constraint")]

                for constraint in self.u_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [(H @u_dual <= b,"dual_mode_input_constraint")]
                
                for constraint in self.y_constraints:
                    H,b = constraint.to_polytope()
                    self.constraints += [(H @self.g(x_dual, u_dual,self.d) <= b,"dual_mode_output_constraint")]

            # Apply terminal cost and constraints at the end of the dual mode horizon
            self.cost += ca.mtimes([(x_dual - self.x_ref).T, self.QT, (x_dual - self.x_ref)])
            
            if self.terminal_set:
                A_terminal, b_terminal = self.terminal_set.A, self.terminal_set.b
                self.constraints += [A_terminal @ x_dual <= b_terminal]
        else:
            # Apply terminal cost and constraints at the end of the main horizon (if no dual mode is specified)
            self.cost += ca.mtimes([(self.x[:, self.N]- self.x_ref).T, self.QT, (self.x[:, self.N]- self.x_ref)])

            if self.terminal_set:
                A_terminal, b_terminal = self.terminal_set.A, self.terminal_set.b
                self.constraints += [A_terminal @ self.x[:, self.N] <= b_terminal]

        # Add slack penalties to the cost function
        for slack, penalty_weight in slack_variables:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.global_penalty_weight * penalty_weight * slack  # Linear penalty
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.global_penalty_weight * penalty_weight * slack**2  # Quadratic penalty
            else :
                raise ValueError("Invalid slack penalty type. Must be 'LINEAR' or 'SQUARE'.")

        
        for constraint,type in self.constraints:
            # print(f"Adding constraint : {type}") # debugging
            self.opti.subject_to(constraint)
        
        self.opti.minimize(self.cost)

        # set solver options
        opts = {'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.tol': 1e-3,
                'expand': 1,
                'error_on_fail':1,                              # to guarantee transparency if solver fails
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.warm_start_bound_push': 1e-8,
                'ipopt.warm_start_mult_bound_push': 1e-8,
                'ipopt.mu_init': 1e-5,
                'ipopt.bound_relax_factor': 1e-9}

        self.opti.solver('ipopt', opts)
        
        # Now, convert the Opti object to a function
        self.mpc_controller = self.opti.to_function('MPCPlanner', [self.x0, self.x_ref, self.u_ref, self.d, self.x, self.v], [self.x, self.v], ['initial_state', 'x_ref', 'v_ref', 'd', 'x_guess', 'v_guess'], ['x_opt', 'v_opt'])
        
        
    def compute_predicted_optimal_controls(self, x0 : np.ndarray, yref : np.ndarray, d : np.ndarray | None =None):
        """
        Compute the optimal control and state trajectories for the given initial state.

        :param x0: Initial state.
        :type x0: np.array
        :param yref: Reference output.
        :type yref: np.array
        :param d: Disturbance (optional).
        :type d: np.array

        :return: Optimal control trajectory and optimal state trajectory.
        :rtype: (np.array, np.array)
        """

        
        # Determine reference point
        self.disturbance_value = np.reshape(d, (self.system.size_disturbance,)) if d is not None else np.zeros(self.system.size_disturbance)
        self.reference_value   = np.reshape(yref, (self.system.size_output,))
        
        (xref, uref) = self.system.get_target_point(self.reference_value , self.disturbance_value) 



        if self.previous_X is not None and self.previous_V is not None:

            # Prepare the initial guess :
            # 1) Take all the elements of the previus solution except the first one
            X_initial  = self.previous_X[:,1:]

            # 2) Add a new last state guess computed from the nomial controller as x_next = f(x, u) with u = -L @ (x - x_ref) + uref (L is reference or dual mode controller depending on the horizon)
            xx        = self.previous_X[:, -1]                                    
            uu        = - self.Ldm @ (xx-xref) + uref if self.Ndm>0 else -self.L @ (xx-xref) + uref
            x_last    = self.f(xx, uu, self.disturbance_value).full()
            X_initial = np.hstack((X_initial,x_last))     
            
            # 3) set initial state of the guess as the one you measure
            X_initial[:,0] = x0

            # 4) for the control input guess, take all the elements of the previous solution except the last one which is set to zero
            V_initial  = np.hstack((self.previous_V[:, 1:], np.zeros((self.m, 1))))


        else:
            # In this case just propagate your trajectory using the constant uref at the set point
            
            X_initial       = np.zeros((self.n, self.N + self.Ndm + 1))
            X_initial[:, 0] = x0
            for k in range(self.N + self.Ndm):
                u = - self.L @ (X_initial[:, k] - xref) + uref
                X_initial[:, k + 1] = self.f(X_initial[:, k], u, self.disturbance_value).full().flatten()

            # Initialize V_initial to make u = uref with the proposed state trajectory
            V_initial = np.zeros((self.m, self.N))

        try:
            (x_opt, v_opt) = self.mpc_controller(x0, xref, uref, self.disturbance_value , X_initial, V_initial)
        except Exception as e:
            print(f"Solver failed: {e}")

        # Extract the control and state trajectories
        #v_opt = sol.value(self.v).reshape(m, N)
        #x_opt = sol.value(self.x)
        u_opt = - self.L @ (x_opt[:, :self.N] - ca.repmat(xref, 1, self.N)) + uref + v_opt


        # Store the solution for the next time step
        self.previous_X = x_opt
        self.previous_V = v_opt

        return u_opt.full(), x_opt.full()


    def get_control_action(self, x : np.ndarray , yref : np.ndarray , d : np.ndarray | None = None):
        """
        MPC controller that plans optimal controls and states over the prediction horizon
        and returns the first control action in the predicted optimal sequence.

        :param x: Current state.
        :type x: np.array
        :param yref: Reference output.
        :type yref: np.array
        :param d: Disturbance (optional).
        :type d: np.array, optional
        :return: The first control action in the predicted optimal sequence.
        :rtype: np.array
        """
        # Plan optimal controls and states over the next N samples
        uPred, xPred = self.compute_predicted_optimal_controls(x, yref, d)

        # Apply the first control action in the predicted optimal sequence
        return uPred[:, 0]

