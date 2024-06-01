import casadi as ca
import numpy as np
from configuration import compute_centers_of_approximation_circles, compute_approximating_circle_radius, update_state, find_closest_distance_with_road_boundary
import sys
import time

class CasadiOptimizer(object):
    def __init__(self, configuration, init_values, predict_horizon, goal_state):
        self.configuration = configuration
        # get some initial values
        self.init_position_x, self.init_position_y, self.init_orientation = init_values[0], init_values[1], init_values[2]
        self.init_values = init_values
        self.predict_horizon = predict_horizon
        self.radius_ego, _ = compute_approximating_circle_radius(configuration.l, configuration.w)
        self.goal_state = goal_state

    def equal_constraints(self, states, controls, f):
        """
        Define constraints
        """
        g = []
        g.append(states[:, 0] - self.init_values)
        # x_k+1 = x_k + f(x,u)*T
        for i in range(self.predict_horizon):
            x_next_ = f(states[:, i], controls[:, i])
            g.append(states[:, i + 1] - x_next_)
        # Define the expressions for the squared distance between ego and obstacle vehicle, in all 9 expressions
        for i in range(self.predict_horizon + 1):
            ego_circles_centers_tuple = compute_centers_of_approximation_circles(states[0, i], states[1, i],
                                                                                 self.configuration.l,
                                                                                 self.configuration.w, states[2, i])
            # for road boundary constraints
            # ego vehicle is approximated to three circles, find the closest point on boundaries of both sides corresponding to three centers
            # 2 boundaries * 3 circle points = 6 constraints
            closest_distance_left_boundary = [find_closest_distance_with_road_boundary(self.configuration.left_road_boundary, ego_circles_centers_tuple[0]),
                                               find_closest_distance_with_road_boundary(self.configuration.left_road_boundary, ego_circles_centers_tuple[1])]
            closest_distance_right_boundary = [find_closest_distance_with_road_boundary(self.configuration.right_road_boundary, ego_circles_centers_tuple[0]),
                                                find_closest_distance_with_road_boundary(self.configuration.right_road_boundary, ego_circles_centers_tuple[1])]

            # 6 constraints for road bound
            g.append(closest_distance_left_boundary[0])
            g.append(closest_distance_left_boundary[1])
            #g.append(closest_distance_left_boundary[2])
            g.append(closest_distance_right_boundary[0])
            g.append(closest_distance_right_boundary[1])
            #g.append(closest_distance_right_boundary[2])


        return g
    def inequal_constraints(self):
        """
        Define lower bound and upper bound for each constraint
        lbg: lower bound of constraints g
        ubg: upper bound of constraints g
        lbx: lower bound of constraints x
        ubx: upper bound of constraints x
        """

        lbg = []
        ubg = []

        lbg.extend([0.0, 0.0, 0.0])
        ubg.extend([0.0, 0.0, 0.0])
        # 5 states constraints for x_k+1 = x_k + f(x,u)*T
        for _ in range(self.predict_horizon):
            lbg.extend([0.0, 0.0, 0.0])  #  x, y, Psi for each time step
            ubg.extend([0.0, 0.0, 0.0])


        # 9 comparisons between centers
        for _ in range(self.predict_horizon + 1):

            lbg.append(0.7)
            lbg.append(0.7)
            lbg.append(0.7)
            lbg.append(0.7)

            ubg.append(np.inf)
            ubg.append(np.inf)
            #ubg.append(np.inf)
            ubg.append(np.inf)
            ubg.append(np.inf)
            #ubg.append(np.inf)


        lbx = []
        ubx = []
        # control inputs constraint
        for _ in range(self.predict_horizon):
            lbx.append(0)
            ubx.append(1)
            lbx.append(-0.25)
            ubx.append(0.25)
        # states constraint
        for _ in range(self.predict_horizon + 1):  # note that this is different with the method using structure
            lbx.append(-np.inf)
            lbx.append(-np.inf)
            lbx.append(-np.inf)

            ubx.append(np.inf)
            ubx.append(np.inf)
            ubx.append(np.inf)
        return lbg, ubg, lbx, ubx


    def cost_function(self, states):
        obj = 0

        # define penalty matrices



        P = np.diag([10, 10, 10])
        #goal_state = np.array([8., 16., 0])
        goal_state = self.goal_state
        # cost
        obj = obj + (states[:, -1] - goal_state.T).T @ P @ (states[:, -1] - goal_state.T)

        return obj
        """
        Define cost function with states cost, input cost and end term cost
        :return: Least Square Lost
        """
        # goal_state = np.array([8., 16., 0])
        # goal_error = states[:, self.predict_horizon] - goal_state.T
        # obj = ca.mtimes([goal_error.T, goal_error])
        #
        # return obj

    def solver(self):
        """
        Use casadi symblolic framework define NLP problem and create IPOPT solver
        :return: solver: casadi solver with ipopt plugin
                      f: dynamic equations
        """
        # define prediction horizon
        horizon = self.predict_horizon
        # set states variables
        sx = ca.SX.sym('sx')
        sy = ca.SX.sym('sy')
        #delta = ca.SX.sym('delta')
        #vel = ca.SX.sym('vel')
        Psi = ca.SX.sym('Psi')
        #states = ca.vertcat(*[sx, sy, delta, vel, Psi])
        states = ca.vertcat(*[sx, sy, Psi])
        num_states = states.size()[0]
        # set control variables
        u0 = ca.SX.sym('u0')
        u1 = ca.SX.sym('u1')
        controls = ca.vertcat(*[u0, u1])
        num_controls = controls.size()[0]
        # get dynamic euqations
        #d = VehicleDynamics()
        rhs = update_state(states, controls)
        f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

        # # build states & control inputs & reference state & reference input for MPC
        U = ca.SX.sym('U', num_controls, horizon)
        X = ca.SX.sym('X', num_states, horizon + 1)
        #U_ref = ca.SX.sym('U_ref', num_controls, horizon)
        #X_ref = ca.SX.sym('X_ref', num_states, horizon + 1)

        # define cost function
        #cost_function = self.cost_function(X, U, X_ref)
        cost_function = self.cost_function(X)


        # get constrains
        #g = self.equal_constraints(X, X_ref, U, f)
        g = self.equal_constraints(X, U, f)
        # define optimize variables for NLP problem/ multi-shooting method
        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))


        # define optimize parameters for NLP problem
        #opt_params = ca.vertcat(ca.reshape(U_ref, -1, 1), ca.reshape(X_ref, -1, 1))
        # build NLP problem
        #nlp_prob = {'f': cost_function, 'x': opt_variables, 'p': opt_params, 'g': ca.vcat(g)}  # here also can use ca.vcat(g) or ca.vertcat(*g)
        nlp_prob = {'f': cost_function, 'x': opt_variables, 'g': ca.vcat(g)}
        # parameters for ipopt solver
        # opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
        #                 'ipopt.acceptable_obj_change_tol': 1e-6, }
        opts_setting = {'ipopt.max_iter': 10000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-9,
                        'ipopt.acceptable_obj_change_tol': 1e-9, }

        # NLP solver created with ipopt plugin
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        return solver, f

    def optimize(self):
        """
        run simulation for MPC problrm
        :return: traj_s: planned optimal states -> ndarray(iter_length, num_states)
                 u: planned optimal control inputs -> ndarray(iter_length, num_inputs)
                 t_v: solve time for each time step -> ndarray(iter_length,)
        """
        # model parameters
        num_states = 3
        num_controls = 2
        # constraints
        lbg, ubg, lbx, ubx = self.inequal_constraints()

        # initial state
        # init_state = np.array(
        #     [self.init_position_x, self.init_position_y, self.init_orientation]).reshape(-1,1)
        init_state = np.array(
            [0, 0, 0]).reshape(-1, 1)
        # set current state
        current_state = init_state.copy()
        u0 = np.array([0.0, 0.0] * self.predict_horizon).reshape(-1, 2).T
        next_trajectories = np.tile(current_state.reshape(1, -1), self.predict_horizon + 1).reshape(
            self.predict_horizon + 1, -1)
        next_states = next_trajectories.copy()
        next_controls = np.zeros((self.predict_horizon, 2))

        # initial variables
        init_control = np.concatenate((u0.T.reshape(-1, 1), next_states.T.reshape(-1, 1)))

        # get solver
        sol, f = self.solver()
        # solving NLP problem
        res = sol(x0=init_control, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)

        estimated_opt = res['x'].full()  # the feedback is in the series [u0, x0, u1, x1, ...]

        # get optimal control input
        u0 = estimated_opt[:int(num_controls * self.predict_horizon)].reshape(self.predict_horizon,
                                                                                  num_controls).T
        x_m = estimated_opt[int(num_controls * self.predict_horizon):].reshape(self.predict_horizon + 1, num_states).T
        last_state = x_m[:, -1]

        return u0, last_state, x_m

