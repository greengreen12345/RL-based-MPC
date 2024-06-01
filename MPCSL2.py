from configuration import Configuration
from optimizer1 import CasadiOptimizer
#from optimizer import CasadiOptimizer
import casadi as ca
import numpy as np
import gym
import mujoco_maze

# env = gym.make('PointNMaze-v0')
# goal_states = [
#             np.array([8., 8., 0.]),
#             np.array([0., 16., np.pi / 2]),
#             np.array([8., 16., 0])
#         ]
# goal_state = goal_states[2]
# observation = env.reset()

class MPCSL2:
    def __init__(self):
        self.config = Configuration()
        # self.goal_states = goal_states
        # self.goal_state = goal_states[2]  # Initialize the goal state
        #self.ca_optimizer = None

    def mpc_output(self, x0, goal_state):
        # Update goal based on the distance between the current position and the goal
        # if np.all(self.goal_state == self.goal_states[2]) and np.linalg.norm(x0[:2] - self.goal_states[0][:2]) < 1.5:
        #     self.goal_state = self.goal_states[1]
        # if np.all(self.goal_state == self.goal_states[1]) and np.linalg.norm(x0[:2] - self.goal_states[1][:2]) < 1.5:
        #     self.goal_state = self.goal_states[2]

        # # Initialize CasadiOptimizer
        ca_optimizer = CasadiOptimizer(configuration=self.config, init_values=x0, predict_horizon=3,
                                       goal_state=goal_state)

        # Solve MPC
        optimal_U_opti, x17, xm = ca_optimizer.optimize()

        return optimal_U_opti

# controller = MPCSL(goal_states)
# x0 = observation['observation'][:3].T
# for i in range(1000):
#     optimal_U_opti, actions = controller.mpc_output(x0)
#     next_state, reward, done, info = env.step(optimal_U_opti[:, 0])
#     x0 = next_state['observation'][:3].T
#     env.render()


