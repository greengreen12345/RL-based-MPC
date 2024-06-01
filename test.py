import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import argparse
import pkg_resources
import importlib
import warnings
import scipy.io as sio
import yaml
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
#from tqdm import tqdm
import gym
import numpy as np
import time
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, SAC, TD3
try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None
if mpi4py is None:
    DDPG, TRPO = None, None
else:
    from stable_baselines import DDPG, TRPO
# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer
#from gym_collision_avoidance.scripts.utils import get_latest_run_id, get_saved_hyperparams, find_saved_model
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env
from gym_collision_avoidance.envs.config import Config
from mpc_rl_collision_avoidance.algorithms.ppo2.ppo2mpc import PPO2MPC
from mpc_rl_collision_avoidance.utils.compute_performance_results import *

from MPCSL2 import MPCSL2

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3,
    'ppo2-mpc': PPO2MPC
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='PointNMaze-v0')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2-mpc',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--scenario', help='Testing scenario', default='',
                        type=str, required=False)
    parser.add_argument('-n', '--n-episodes', help='number of episodes', default=100,
                        type=int)
    parser.add_argument('--n-envs', help='number of environments', default=1,
                        type=int)
    parser.add_argument('--n-agents', help='number of agents', default=5,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=144,
                        type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--no-render', action='store_true', default=True,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--coll_avd', action='store_true', default=True,
                        help='Enable collision avoidance')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--load-best', action='store_true', default=False,
                        help='Load best model instead of last model if available')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--record', action='store_true', default=False,
                        help='Save episode images and gifs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--policy', help='Ego agent policy', default='MPCPolicy', type=str)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    args = parser.parse_args()



    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = dir_path + '/' + folder
    # if args.exp_id == 0:
    #     args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
    #     print('Loading latest experiment, id={}'.format(args.exp_id))

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
    else:
        log_path = os.path.join(folder, algo)



    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    #model_path = find_saved_model(algo, log_path, env_id, load_best=args.load_best)

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        args.n_envs = 1

    set_global_seeds(args.seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    #hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    log_dir = args.reward_log if args.reward_log != '' else None

    ####### Gym-collision-avodiance Environment - Swap Scenario
    Config.TRAIN_SINGLE_AGENT = True
    Config.ANIMATE_EPISODES = args.record
    Config.SHOW_EPISODE_PLOTS = False
    Config.SAVE_EPISODE_PLOTS = args.record
    Config.EVALUATE_MODE = True

    #env, one_env = create_env()
    # if args.scenario != "":
    #     one_env.scenario = [args.scenario]
    #
    # one_env.number_of_agents = args.n_agents
    # env.unwrapped.envs[0].env.ego_policy = args.policy
    from gym.wrappers.time_limit import TimeLimit
    import mujoco_maze

    env = TimeLimit(gym.make('PointNMaze-v0'), max_episode_steps=5000)
    from stable_baselines.common.vec_env import DummyVecEnv


    env = DummyVecEnv([lambda: env])

    #model_path = "/home/my/go-mpc/logs/ppo2-mpc/PointNMaze-v0_241/PointNMaze-v0.zip"
    #model_path = "/home/my/go-mpc/logs/ppo2-mpc/PointNMaze-v0_241/rl_model_600000_steps.zip"
    #model_path = "/home/my/go-mpc/logs/ppo2-mpc/PointNMaze-v0_243/rl_model_851056_steps.zip"
    # model_path = "/home/my/go-mpc/logs/ppo2-mpc/PointNMaze-v0_244/rl_model_751056_steps.zip"
    #model_path = "/home/my/go-mpc/logs/ppo2-mpc/PointNMaze-v0_244/rl_model_600000_steps.zip"
    #model_path = "/home/my/go-mpc/logs/ppo2-mpc/PointNMaze-v0_246/rl_model_600000_steps.zip"
    model_path = "/home/my/RL-based-MPC/logs/ppo2-mpc/PointNMaze-v0_248/rl_model_2183616_steps.zip"

    model = ALGOS[algo].load(model_path)
    #model = ALGOS[algo].load(model_path)
    #print(env.observation_space)
    #print(model.observation_space)
    obs = env.reset()


    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not args.stochastic

    # Save plot trajectories
    # plot_save_dir = log_path + '/figs/'
    # os.makedirs(plot_save_dir, exist_ok=True)
    # one_env.plot_save_dir = plot_save_dir

    total_reward = 0
    step = 0
    done = False
    num_test_cases = 1
    trajs = [[] for _ in range(num_test_cases)]
    episode_stats = []
    total_n_infeasible = 0
    #for ep_id in tqdm(range(args.n_episodes)):


    while True:
        actions = []
        # agents = env.unwrapped.envs[0].env.agents
        # ego_agent = agents[0]
        # number_of_agents = len(one_env.agents)-1
        # agents[0].policy.x_error_weight_ = 1.0
        # agents[0].policy.y_error_weight_ = 1.0
        # agents[0].policy.cost_function_weight = 0.0
        # agents[0].policy.policy_network = model
        # agents[0].policy.reset_network()
        # agents[0].policy.enable_collision_avoidance = args.coll_avd
        episode_step = 0
        state = None
        n_infeasible= 0
        episode_nn_processing_times = []
        episode_mpc_processing_times = []

        x0 = obs['observation'][:3].T
        x0 = x0[:3, 0]
        obs = obs['observation'][:3].T


        # obs[:3, 0] =  [  3.871812, 12.060381,  0]
        # action, state = model.predict(obs[:3, 0], state=state, deterministic=deterministic,
        #                               seq_length=np.ones(1) * (1 * 9))
        # print("actionn", action)
        j = 0
        while True:

            #mpc = MPCController()
            mpc = MPCSL2()
            action, state = model.predict(obs[:3, 0], state=state, deterministic=deterministic,
                                          seq_length=np.ones(1) * (1 * 9))
            #print("observation", obs[:3, 0])
            #print("action", action)

            # if j == 0:
            #     action = np.array([[6, 0]])
                #action = np.array([[2,1]])
                #j += 1
            # #action = np.array([[8, 0]])
            print("action", action)
            # #print("observation", obs[:3, 0])
            goal_state1 = np.append(action, [[0]], axis=1)
            goal_state1 = goal_state1[0]
            env.unwrapped.envs[0].env.set_goal(goal_state1)
            #env.unwrapped.envs[0].env.set_goal(action)

            for i in range(60):
            #while True:
                start = time.time()

                #print("obs.shape", obs.shape)
                #print(obs[:3, 0].shape)

                #action, state = model.predict(obs[:3, 0], state=state, deterministic=deterministic,seq_length = np.ones(1) * (1 * 9))

                #print("obs['observation']", obs['observation'])
                #print("obs", obs)
                #print("observation", obs[:3, 0])
                #print("action", action)
                end = time.time()
                episode_nn_processing_times.append(end - start)
                actions.append(action)
                #print("observation", obs[:3, 0])
                #print(action)
                # Send some info for collision avoidance env visualization (need a better way to do this)
                # one_env.set_perturbed_info({'perturbed_obs': perturbed_obs[0], 'perturber': perturber})

                # Update the rendering of the environment (optional)

                env.render()

                if x0[1]>6 and x0[1]<16:
                    goal_state = np.append(action, [[np.pi]], axis=1)
                else:
                    goal_state = np.append(action, [[0]], axis=1)

                # goal_state = np.array([0., 0., 0.])

                # goal_state = np.append(action, [[0]], axis=1)
                # goal_state = goal_state[0]
                #print("goal_state", goal_state)
                # print("x0", x0)

                #env.unwrapped.envs[0].env.set_goal(goal_state)



                #optimal_U_opti = mpc.get_optimal_control(x0, goal_state)
                optimal_U_opti = mpc.mpc_output(x0, goal_state)


                # Take a step in the environment, record reward/steps for logging
               # ego_agent.policy.network_output_to_action(0, agents, action[0])
                next_state, rewards, done, which_agents_done = env.unwrapped.envs[0].env.step(optimal_U_opti[:, 0])

                x0 = next_state['observation'][:3].T
                # print("next_state['observation']", next_state['observation'])
                obs = next_state['observation'].reshape(-1, 1)


                #episode_mpc_processing_times.append(agents[0].policy.solve_time)

                #n_infeasible += agents[0].is_infeasible
                total_reward += rewards
                step += 1
                episode_step += 1

                # if np.linalg.norm(x0[:2] - action) < 2.5:
                #     break

        # After end of episode, store some statistics about the environment
        # Some stats apply to every gym env...

        generic_episode_stats = {
            'total_reward': total_reward,
            'steps': step,
            'actions': actions
        }

        #agents = one_env.prev_episode_agents
        time_to_goal = np.array([a.t for a in agents])
        extra_time_to_goal = np.array([a.t - a.straight_line_time_to_reach_goal for a in agents])

        #print("N infeasible solutions: " + str(n_infeasible))
        total_n_infeasible += n_infeasible

        collision = agents[0].in_collision
        timeout = agents[0].ran_out_of_time
        all_at_goal = np.array(
                np.all([a.is_at_goal for a in agents])).tolist()
        any_stuck = np.array(
                np.any([not a.in_collision and not a.is_at_goal for a in agents])).tolist()
        outcome = "collision" if collision else "all_at_goal" if all_at_goal else "stuck"
        if len(agents) > 1:
            specific_episode_stats = {
                    'num_agents': len(agents),
                    'time_to_goal': time_to_goal,
                    'total_time_to_goal': np.sum(time_to_goal),
                    'extra_time_to_goal': extra_time_to_goal,
                    'collision': collision,
                    'stuck': timeout,
                    'succeeded': agents[0].is_at_goal,
                    'all_at_goal': all_at_goal,
                    'any_stuck': any_stuck,
                    'outcome': outcome,
                    'ego_agent_traj': agents[0].global_state_history[:episode_step],
                    'other_agents_traj': agents[1].global_state_history[:episode_step],
                    'episode_nn_processing_times': np.asarray(episode_nn_processing_times),
                    'episode_mpc_processing_times': np.asarray(episode_mpc_processing_times)
            }
        else:
            specific_episode_stats = {
                    'num_agents': len(agents),
                    'time_to_goal': time_to_goal,
                    'total_time_to_goal': np.sum(time_to_goal),
                    'extra_time_to_goal': extra_time_to_goal,
                    'collision': collision,
                    'stuck': timeout,
                    'ego_agent_traj': agents[0].global_state_history[:episode_step],
                    'all_at_goal': all_at_goal,
                    'any_stuck': any_stuck,
                    'outcome': outcome,
                    'episode_nn_processing_times': np.asarray(episode_nn_processing_times),
                    'episode_mpc_processing_times': np.asarray(episode_mpc_processing_times)
            }

        # Merge all stats into a single dict
        episode_stats.append({**generic_episode_stats, **specific_episode_stats})
        done = False
        #one_env.test_case_index = ep_id
        print("N infeasible solutions: " + str(n_infeasible))
        total_n_infeasible += n_infeasible

    episode_stats_dict = {
        "all_episodes_stats": episode_stats
    }
    results_file = stats_path + '_model_'+str(args.exp_id)+'_'+str(args.n_agents)+'_agents_perf_results.mat'
    sio.savemat(results_file, episode_stats_dict)

    perf_results = process_statistics(episode_stats)
    print("***********Number of Infeasibilities**********************")
    print("***********       "+str(total_n_infeasible)+"           **********************")

    with open(os.path.join(stats_path, 'model_'+str(args.exp_id)+'_'+str(args.n_agents)+'_agents_perf_results.yml'), 'w') as f:
        yaml.dump(perf_results, f)

if __name__ == '__main__':
    main()
