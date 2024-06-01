import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy
from mpc_rl_collision_avoidance.mpc import FORCESNLPsolver_py

class MPCPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="MPCPolicy")
        self.is_still_learning = True
        self.ppo_or_learning_policy = True
        self.debug = False
        # Goal Parameterization
        self.goal_ = np.zeros([2])
        self.current_state_ = np.zeros([5])
        self.next_robot_state = np.zeros([5])
        self.control_cmd_linear = 0.0
        self.control_cmd_angular = 0.0
        self.dt = 0.1

        # FORCES parameters
        self.FORCES_x0 = np.zeros(160, dtype="double")
        self.FORCES_xinit = np.zeros(5, dtype="double")
        self.FORCES_all_parameters = np.zeros(1320, dtype="double")
        self.FORCES_N = 20
        self.FORCES_NU = 3
        self.FORCES_NX = 5
        self.FORCES_TOTAL_V = 8
        self.FORCES_NPAR = 66

        # Dynamic weights
        self.x_error_weight_ = 1.0
        self.y_error_weight_ = 1.0
        self.velocity_weight_ = 0.005
        self.angular_velocity_weight_ = 0.005
        self.theta_error_weight_ = 0.0
        self.reference_velocity_ = 1.0
        self.reference_ang_velocity_ = 0.0
        self.slack_weight_ = 10000
        self.repulsive_weight_ = 0.0
        self.n_obstacles_ = 6

        # Cost function approximation coefficients
        self.coefs = np.zeros([6])
        self.d = 0
        self.cost_function_weight = 0

        self.enable_collision_avoidance = True
        self.only_mpc = False

        self.policy_name = "MPC"

    def network_output_to_action(self, id, agents, network_output):
        agent = agents[id]

        self.goal_[0] = agent.goal_global_frame[0]
        self.goal_[1] = agent.goal_global_frame[1]

        self.current_state_[0] = agent.pos_global_frame[0]
        self.current_state_[1] = agent.pos_global_frame[1]
        self.current_state_[2] = agent.heading_global_frame
        self.current_state_[3] = self.control_cmd_linear
        self.current_state_[4] = self.control_cmd_angular

        exit_flag, _ = self.run_solver(agents,update=True)

        if exit_flag == 1:
            agent.set_state(px=self.next_robot_state[0],py=self.next_robot_state[1],heading =self.next_robot_state[2],
                            vx=self.control_cmd_linear*np.cos(self.next_robot_state[2]), vy=self.control_cmd_linear*np.sin(self.next_robot_state[2]))
            agent.next_state = self.next_robot_state
            agent.is_infeasible = False
        else:
            agent.is_infeasible = True

        # if we are within the planning horizon then use directly mpc
        return np.array([self.FORCES_x0[19*self.FORCES_TOTAL_V + self.FORCES_NU]-agent.pos_global_frame[0],
                            self.FORCES_x0[19 * self.FORCES_TOTAL_V + self.FORCES_NU+1]-agent.pos_global_frame[1]])

    def mpc_output(self, id, agents):
        agent = agents[id]
        self.policy_name = "MPC"

        self.goal_[0] = agent.goal_global_frame[0]
        self.goal_[1] = agent.goal_global_frame[1]

        self.x_error_weight_ = 1.0
        self.y_error_weight_ = 1.0
        self.cost_function_weight = 0.0

        self.current_state_[0] = agent.pos_global_frame[0]
        self.current_state_[1] = agent.pos_global_frame[1]
        self.current_state_[2] = agent.heading_global_frame
        self.current_state_[3] = self.control_cmd_linear
        self.current_state_[4] = self.control_cmd_angular

        exit_flag, output = self.run_solver(agents,update=False)

        if exit_flag == 1:
            return (np.array([output["x20"][self.FORCES_NU],output["x20"][self.FORCES_NU+1]])-self.current_state_[:2]), exit_flag
        else:
            return (np.array([self.goal_[0],self.goal_[1]]) - self.current_state_[:2]), exit_flag

    def reset_solver(self):
        self.FORCES_x0[:] = 0.0
        self.FORCES_xinit[:] = 0.0
        self.FORCES_all_parameters[:] = 0.0

    def update_current_state(self, new_state_):
        self.current_state_[0] = new_state_[0]
        self.current_state_[1] = new_state_[1]
        self.current_state_[2] = new_state_[2]
        self.current_state_[3] = new_state_[3]

    def run_solver(self,agents,update):
        # Initial conditions
        self.FORCES_xinit[0] = self.current_state_[0] # x position
        self.FORCES_xinit[1] = self.current_state_[1] # y position
        self.FORCES_xinit[2] = self.current_state_[2] # theta position
        self.FORCES_xinit[3] = self.current_state_[3]  # velocity
        self.FORCES_xinit[4] = self.current_state_[4]  # angular velocity
        self.FORCES_x0[self.FORCES_NU] = self.current_state_[0] # x position
        self.FORCES_x0[self.FORCES_NU + 1] = self.current_state_[1] # y position
        self.FORCES_x0[self.FORCES_NU + 2] = self.current_state_[2] # theta position
        self.FORCES_x0[self.FORCES_NU + 3] = self.current_state_[3]  # velocity
        self.FORCES_x0[self.FORCES_NU + 4] = self.current_state_[4]  # velocity

        other_agents_ordered = np.zeros(((len(agents)-1), 6))

        for ag_id in range(1,len(agents)):
            other_agents_ordered[ag_id-1, 0] = agents[ag_id].pos_global_frame[0]
            other_agents_ordered[ag_id-1, 1] = agents[ag_id].pos_global_frame[1]
            other_agents_ordered[ag_id-1, 2] = agents[ag_id].vel_global_frame[0]
            other_agents_ordered[ag_id-1, 3] = agents[ag_id].vel_global_frame[1]
            other_agents_ordered[ag_id-1, 4] = np.linalg.norm(agents[ag_id].pos_global_frame - agents[0].pos_global_frame)
            other_agents_ordered[ag_id-1, 5] = agents[ag_id].radius +0.1

        other_agents_ordered = other_agents_ordered[other_agents_ordered[:, 4].argsort()]

        for N_iter in range(0, self.FORCES_N):
            k = N_iter * self.FORCES_NPAR

            self.FORCES_all_parameters[k + 0] = self.goal_[0]
            self.FORCES_all_parameters[k + 1] = self.goal_[1]
            self.FORCES_all_parameters[k + 2] = self.repulsive_weight_
            self.FORCES_all_parameters[k + 3] = self.x_error_weight_
            self.FORCES_all_parameters[k + 4] = self.y_error_weight_
            self.FORCES_all_parameters[k + 5] = self.angular_velocity_weight_
            self.FORCES_all_parameters[k + 6] = self.theta_error_weight_
            self.FORCES_all_parameters[k + 7] = self.velocity_weight_
            self.FORCES_all_parameters[k + 8] = self.slack_weight_
            self.FORCES_all_parameters[k + 9] = self.reference_velocity_
            self.FORCES_all_parameters[k + 10] = self.reference_ang_velocity_
            self.FORCES_all_parameters[k + 26] = agents[0].radius  # disc radius +0.05
            self.FORCES_all_parameters[k + 27] = 0.0 # disc position
            self.FORCES_all_parameters[k + 58] = self.coefs[0]
            self.FORCES_all_parameters[k + 59] = self.coefs[1]
            self.FORCES_all_parameters[k + 60] = self.coefs[2]
            self.FORCES_all_parameters[k + 61] = self.coefs[3]
            self.FORCES_all_parameters[k + 62] = self.coefs[4]
            self.FORCES_all_parameters[k + 63] = self.coefs[5]
            self.FORCES_all_parameters[k + 64] = self.d
            self.FORCES_all_parameters[k + 65] = self.cost_function_weight

            # todo: order agents by distance , other agent is hard coded
            if self.enable_collision_avoidance:
                for obs_id in range(np.minimum(len(agents)-1,self.n_obstacles_)):
                    # Constant Velocity Assumption
                    self.FORCES_all_parameters[k + 28 + obs_id * 5] = other_agents_ordered[obs_id, 0] + N_iter*self.dt*other_agents_ordered[obs_id, 2]
                    self.FORCES_all_parameters[k + 29 + obs_id * 5] = other_agents_ordered[obs_id, 1] + N_iter*self.dt*other_agents_ordered[obs_id, 3]
                    self.FORCES_all_parameters[k + 30 + obs_id * 5] = 0.0  # orientation
                    self.FORCES_all_parameters[k + 31 + obs_id * 5] = other_agents_ordered[obs_id, 5]  # major axis
                    self.FORCES_all_parameters[k + 32 + obs_id * 5] = other_agents_ordered[obs_id, 5]  # minor axis

                for j in range(np.minimum(len(agents)-1,self.n_obstacles_),self.n_obstacles_):
                    self.FORCES_all_parameters[k + 28 + j * 5] = other_agents_ordered[obs_id, 0] + N_iter*self.dt*other_agents_ordered[obs_id, 2]
                    self.FORCES_all_parameters[k + 29 + j * 5] = other_agents_ordered[obs_id, 1] + N_iter*self.dt*other_agents_ordered[obs_id, 3]
                    self.FORCES_all_parameters[k + 30 + j * 5] = 0.0 # orientation
                    self.FORCES_all_parameters[k + 31 + j * 5] = other_agents_ordered[obs_id, 5]  # major axis
                    self.FORCES_all_parameters[k + 32 + j * 5] = other_agents_ordered[obs_id, 5]  # major axis
            else:
                print("Collision avoidance OFF")
                self.slack_weight_ = 0
                for obs_id in range(0,self.n_obstacles_):
                    self.FORCES_all_parameters[k + 28 + obs_id * 5] = 100.0
                    self.FORCES_all_parameters[k + 29 + obs_id * 5] = 100.0
                    self.FORCES_all_parameters[k + 30 + obs_id * 5] = 0.0 # orientation
                    self.FORCES_all_parameters[k + 31 + obs_id * 5] = 0.4 # major axis
                    self.FORCES_all_parameters[k + 32 + obs_id * 5] = 0.4 # minor axis

        PARAMS = {"x0": self.FORCES_x0, "xinit": self.FORCES_xinit, "all_parameters": self.FORCES_all_parameters}
        OUTPUT, EXITFLAG, INFO = FORCESNLPsolver_py.FORCESNLPsolver_solve(PARAMS)

        if self.debug:
            print(INFO.pobj)

        if EXITFLAG == 1 and update:
            for i in range(0, self.FORCES_TOTAL_V):
                self.FORCES_x0[i] = OUTPUT["x01"][i]
                self.FORCES_x0[i + self.FORCES_TOTAL_V] = OUTPUT["x02"][i]
                self.FORCES_x0[i + 2*self.FORCES_TOTAL_V] = OUTPUT["x03"][i]
                self.FORCES_x0[i + 3*self.FORCES_TOTAL_V] = OUTPUT["x04"][i]
                self.FORCES_x0[i + 4*self.FORCES_TOTAL_V] = OUTPUT["x05"][i]
                self.FORCES_x0[i + 5*self.FORCES_TOTAL_V] = OUTPUT["x06"][i]
                self.FORCES_x0[i + 6*self.FORCES_TOTAL_V] = OUTPUT["x07"][i]
                self.FORCES_x0[i + 7*self.FORCES_TOTAL_V] = OUTPUT["x08"][i]
                self.FORCES_x0[i + 8*self.FORCES_TOTAL_V] = OUTPUT["x09"][i]
                self.FORCES_x0[i + 9*self.FORCES_TOTAL_V] = OUTPUT["x10"][i]
                self.FORCES_x0[i + 10*self.FORCES_TOTAL_V] = OUTPUT["x11"][i]
                self.FORCES_x0[i + 11*self.FORCES_TOTAL_V] = OUTPUT["x12"][i]
                self.FORCES_x0[i + 12*self.FORCES_TOTAL_V] = OUTPUT["x13"][i]
                self.FORCES_x0[i + 13*self.FORCES_TOTAL_V] = OUTPUT["x14"][i]
                self.FORCES_x0[i + 14*self.FORCES_TOTAL_V] = OUTPUT["x15"][i]
                self.FORCES_x0[i + 15*self.FORCES_TOTAL_V] = OUTPUT["x16"][i]
                self.FORCES_x0[i + 16*self.FORCES_TOTAL_V] = OUTPUT["x17"][i]
                self.FORCES_x0[i + 17*self.FORCES_TOTAL_V] = OUTPUT["x18"][i]
                self.FORCES_x0[i + 18*self.FORCES_TOTAL_V] = OUTPUT["x19"][i]
                self.FORCES_x0[i + 19*self.FORCES_TOTAL_V] = OUTPUT["x20"][i]


            self.next_robot_state[0] = self.FORCES_x0[self.FORCES_TOTAL_V + self.FORCES_NU]
            self.next_robot_state[1] = self.FORCES_x0[self.FORCES_TOTAL_V + self.FORCES_NU + 1]
            self.next_robot_state[2] = self.FORCES_x0[self.FORCES_TOTAL_V + self.FORCES_NU + 2]
            self.next_robot_state[3] = self.FORCES_x0[self.FORCES_TOTAL_V + self.FORCES_NU + 3]
            self.next_robot_state[4] = self.FORCES_x0[self.FORCES_TOTAL_V + self.FORCES_NU + 4]

            self.control_cmd_linear = self.FORCES_x0[self.FORCES_TOTAL_V+ self.FORCES_NU + 3]
            self.control_cmd_angular = self.FORCES_x0[self.FORCES_TOTAL_V+ self.FORCES_NU + 4]
        elif EXITFLAG == 0:
            self.FORCES_x0[self.FORCES_TOTAL_V:] *=0

        return EXITFLAG, OUTPUT

    def mpc_output1(self, x0, goal_state, goal_states):
        self.goal_state = goal_state
        self.goal_states = goal_states

        if np.all(self.goal_state == self.goal_states[2]) and np.linalg.norm(
                x0[:2] - self.goal_states[0][:2]) < 1.5:
            self.goal_state = self.goal_states[1]
        if np.all(self.goal_state == self.goal_states[1]) and np.linalg.norm(
                x0[:2] - self.goal_states[1][:2]) < 1.5:
            self.goal_state = self.goal_states[2]

        # Formulate MPC problem
        ca_optimizer = CasadiOptimizer(configuration=self.config, init_values=x0, predict_horizon=3,
                                       goal_state=self.goal_state)

        # Solve MPC problem
        optimal_U_opti, x17, xm = ca_optimizer.optimize()

        return optimal_U_opti, xm[:, -1] - xm[:, 0]

    def find_next_action(self, obs, agents, i):
        self.network_output_to_action(i,agents,obs)