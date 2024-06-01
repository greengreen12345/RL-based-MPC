import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import time

import gym
import numpy as np
import tensorflow as tf
import gc
# from copy import deepcopy
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from timeit import default_timer as timer
#from pympler.tracker import SummaryTracker
from mpc_rl_collision_avoidance.external.stable_baselines import logger
from mpc_rl_collision_avoidance.external.stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from mpc_rl_collision_avoidance.external.stable_baselines.common.runners import AbstractEnvRunner
from mpc_rl_collision_avoidance.external.stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from mpc_rl_collision_avoidance.external.stable_baselines.common.schedules import get_schedule_fn
from mpc_rl_collision_avoidance.external.stable_baselines.common.tf_util import total_episode_reward_logger
from mpc_rl_collision_avoidance.external.stable_baselines.common.math_util import safe_mean
from gym_collision_avoidance.envs.config import Config

#from MPCSL import MPCSL
from MPCSL1 import MPCSL
from MPCSL2 import MPCSL2

class PPO2MPC(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.n_steps = n_steps
        #self.n_steps = 1
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.value = None
        #self.n_batch = None

        self.n_batch = 512
        self.summary = None

        #self.pre_training_steps = 100000
        self.pre_training_steps = 300000
        #self.pre_training_steps = 0
        #self.curriculum_learning = Config.CURRICULUM_LEARNING

        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        # session = tf.Session()
        # #self.model = self.policy(session)
        # self.model = self.policy(session, ob_space=env.observation_space, ac_space=env.action_space,
        #                          n_env=1, n_steps=n_steps, n_batch=self.n_batch)

        if _init_setup_model:
            self.setup_model()

    def _make_runner(self):
        return Runner(env=self.env, model=self, n_steps=self.n_steps,
                      gamma=self.gamma, lam=self.lam)

    def _make_mpc_runner(self):
        return MPCRunner(env=self.env, model=self, n_steps=self.n_steps,
                      gamma=self.gamma, lam=self.lam)

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches


                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.norm_advs_ph = tf.placeholder(tf.float32, [None], name="norm_advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")
                    self.collisions_ph = tf.placeholder(tf.float32, [], name="collisions_ph")
                    self.timeouts_ph = tf.placeholder(tf.float32, [], name="timeouts_ph")
                    self.infesiable_ph = tf.placeholder(tf.float32, [], name="infesiable_ph")
                    self.n_reach_goal_ph = tf.placeholder(tf.float32, [], name="n_reach_goal_ph")
                    self.n_mpc_actions_ph = tf.placeholder(tf.float32, [], name="n_mpc_actions_ph")
                    self.sequence_legth_ph = train_model.seq_length_ph

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model.value_flat
                    self.values = train_model.value_flat

                    # Value function clipping: not present in the original PPO
                    if self.cliprange_vf is None:
                        # Default behavior (legacy from OpenAI baselines):
                        # use the same clipping as for the policy
                        self.clip_range_vf_ph = self.clip_range_ph
                        self.cliprange_vf = self.cliprange
                    elif isinstance(self.cliprange_vf, (float, int)) and self.cliprange_vf < 0:
                        # Original PPO implementation: no value function clipping
                        self.clip_range_vf_ph = None
                    else:
                        # Last possible behavior: clipping range
                        # specific to the value function
                        self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                    if self.clip_range_vf_ph is None:
                        # No clipping
                        vpred_clipped = train_model.value_flat
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        vpred_clipped = self.old_vpred_ph + \
                            tf.clip_by_value(train_model.value_flat - self.old_vpred_ph,
                                             - self.clip_range_vf_ph, self.clip_range_vf_ph)


                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.norm_advs_ph * ratio
                    pg_losses2 = -self.norm_advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    supervised_loss = tf.reduce_mean(
                        tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.action_ph,train_model.deterministic_action)), axis=1))) # before: train_model.policy

                    self.supervised_policy_loss = tf.reduce_mean(supervised_loss) + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('supervised_policy_loss', self.supervised_policy_loss)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads = tf.gradients(loss, self.params)
                    supervised_grads = tf.gradients(supervised_loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                        supervised_grads, _grad_norm = tf.clip_by_global_norm(supervised_grads, 10.0*self.max_grad_norm)
                    grads = list(zip(grads, self.params))
                    supervised_grads = list(zip(supervised_grads, self.params))
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                supervised_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self.supervised_train = supervised_trainer.apply_gradients(supervised_grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('number_of_mpc_actions', self.n_mpc_actions_ph)
                    tf.summary.scalar('number_of_collisions', self.collisions_ph)
                    tf.summary.scalar('number_of_timeouts', self.timeouts_ph)
                    tf.summary.scalar('number_of_infeasible_solutions', self.infesiable_ph)
                    tf.summary.scalar('number_of_successful_events', self.n_reach_goal_ph)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('normalized advantage', tf.reduce_mean(self.norm_advs_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('values', tf.reduce_mean(self.values))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    if self.clip_range_vf_ph is not None:
                        tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))

                    tf.summary.scalar('old_neglog_action_probability', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.norm_advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probability', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, states, update,
                    writer,n_collisions,n_mpc_actions,n_timeouts,n_other_agents,n_infeasible_sols,n_reach_goal, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        norm_advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        seq_length = np.full((self.n_batch,), 9)
        td_map = {self.train_model.obs_ph: obs,
                  self.action_ph: actions,
                  self.advs_ph: advs,
                  self.norm_advs_ph: norm_advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values,
                  self.collisions_ph: n_collisions,
                  self.timeouts_ph: n_timeouts,
                  self.infesiable_ph: n_infeasible_sols,
                  self.n_reach_goal_ph: n_reach_goal,
                  self.train_model.seq_length_ph: seq_length,
                  self.n_mpc_actions_ph: n_mpc_actions}
        if states is not None:
            td_map[self.train_model.states_ph] = states*1.0
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def _mpc_train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, states, update,
                    writer,n_collisions,n_mpc_actions,n_timeouts,n_other_agents,n_infeasible_sols,n_reach_goal, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        norm_advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        seq_length = np.full((self.n_batch,), 9)
        td_map = {self.train_model.obs_ph: obs[:self.n_batch],
                  self.action_ph: actions[:self.n_batch],
                  self.advs_ph: advs[:self.n_batch],
                  self.norm_advs_ph: norm_advs[:self.n_batch], self.rewards_ph: returns[:self.n_batch],
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs[:self.n_batch], self.old_vpred_ph: values[:self.n_batch],
                  self.collisions_ph: n_collisions,
                  self.timeouts_ph: n_timeouts,
                  self.infesiable_ph: n_infeasible_sols,
                  self.n_reach_goal_ph: n_reach_goal,
                  self.train_model.seq_length_ph: seq_length,
                  self.n_mpc_actions_ph: n_mpc_actions}

        if states is not None:
            td_map[self.train_model.states_ph] = states*1.0
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, supervised_policy_loss,  clipfrac, _ = self.sess.run(
                    [self.summary, self.supervised_policy_loss, self.clipfrac, self.supervised_train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, supervised_policy_loss, clipfrac, _ = self.sess.run(
                    [self.summary, self.supervised_policy_loss, self.clipfrac, self.supervised_train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            supervised_policy_loss, clipfrac, _ = self.sess.run(
                [self.supervised_policy_loss, self.clipfrac, self.supervised_train], td_map)

        return supervised_policy_loss, clipfrac

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        #total_timesteps = 365000
        #total_timesteps = 120000
        # print("total_timesteps", total_timesteps)

        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)
        self.mpc_runner = self._make_mpc_runner()
        #self.runner = self._make_runner()

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        n_collisions = 0

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            t_first_start = time.time()
            n_updates = total_timesteps // self.n_batch

            callback.on_training_start(locals(), globals())

            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0, ("The number of minibatches (`nminibatches`) "
                                                               "is not a factor of the total number of samples "
                                                               "collected per rollout (`n_batch`), "
                                                               "some samples won't be used."
                                                               )
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)

                callback.on_rollout_start()

                # Curriculum learning
                #self.episode_number = self.env.unwrapped.envs[0].env.episode_number
                self.total_number_of_steps = self.env.unwrapped.envs[0].env.total_number_of_steps
                #self.total_number_of_steps = self.env.unwrapped.envs[0].env.t
                self.curriculum_learning = 1
                if self.curriculum_learning:
                    if self.total_number_of_steps < self.pre_training_steps:
                        mpc_train = True
                        print("mpc", self.total_number_of_steps)
                    elif self.total_number_of_steps < 4e5:
                        mpc_train = False
                        print("rl", self.total_number_of_steps)
                    elif self.total_number_of_steps < 4e5 + self.pre_training_steps:
                        mpc_train = True
                        print("mpc", self.total_number_of_steps)
                    elif self.total_number_of_steps < 8e5:
                        mpc_train = False
                        print("rl", self.total_number_of_steps)
                    elif self.total_number_of_steps < 8e5 + self.pre_training_steps:
                        mpc_train = True
                        print("mpc", self.total_number_of_steps)
                    elif self.total_number_of_steps < 12e5:
                        mpc_train = False
                        print("rl", self.total_number_of_steps)
                    elif self.total_number_of_steps < 12e5 + self.pre_training_steps:
                        mpc_train = True
                        print("mpc", self.total_number_of_steps)
                    elif self.total_number_of_steps < 16e5:
                        mpc_train = False
                        print("rl", self.total_number_of_steps)
                    elif self.total_number_of_steps < 16e5 + self.pre_training_steps:
                        mpc_train = True
                        print("mpc", self.total_number_of_steps)
                    elif self.total_number_of_steps >= 16e5 + self.pre_training_steps:
                        mpc_train = False
                        print("rl", self.total_number_of_steps)
                else:
                    if self.total_number_of_steps < self.pre_training_steps:
                        mpc_train = True
                        print("mpc", self.total_number_of_steps)
                    else:
                        mpc_train = False
                        print("rl", self.total_number_of_steps)

                # mpc_train = False
                # print("rl", self.total_number_of_steps)

                # true_reward is the reward without discount
                if mpc_train:
                    # this flag is not used anymore
                    #self.env.unwrapped.envs[0].env.agents[0].warm_start = True
                    rollout = self.mpc_runner.run(callback)
                else:
                    # this flag is not used anymore
                    #self.env.unwrapped.envs[0].env.agents[0].warm_start = False
                    rollout = self.runner.run(callback)
                # Unpack
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, n_collisions_rollout, n_infeasible_sols,n_timeouts,n_reach_goal, n_mpc_actions, mb_n_other_agents = rollout

                callback.on_rollout_end()

                #np.roll(n_collisions,1)
                n_collisions = n_collisions_rollout

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version

                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,n_collisions=n_collisions,
                                                n_mpc_actions = np.mean(n_mpc_actions),update=timestep, cliprange_vf=cliprange_vf_now),
                                                n_infeasible_sols=n_infeasible_sols,n_timeouts=n_timeouts,n_other_agents=mb_n_other_agents)
                else:  # recurrent version

                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()

                            # print("mb_flat_inds 最大值:", mb_flat_inds.max())
                            # print("obs 数组大小:", len(obs))

                            if mpc_train:

                                slices = (arr[mb_flat_inds] for arr in
                                          (obs, returns, masks, actions, values, neglogpacs,states))
                                #mb_states = states[mb_env_inds]
                                mb_loss_vals.append(self._mpc_train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                     writer=writer,n_collisions=n_collisions,
                                                                     n_mpc_actions = np.mean(n_mpc_actions),n_infeasible_sols=n_infeasible_sols,
                                                                     n_timeouts=n_timeouts,cliprange_vf=cliprange_vf_now
                                                                     ,n_other_agents=mb_n_other_agents,n_reach_goal=n_reach_goal))
                            else:

                                slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs,states))
                                #mb_states = states[mb_env_inds]
                                mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                     writer=writer,n_collisions=n_collisions,
                                                                     n_mpc_actions = np.mean(n_mpc_actions),n_infeasible_sols=n_infeasible_sols,
                                                                     n_timeouts=n_timeouts, cliprange_vf=cliprange_vf_now
                                                                     ,n_other_agents=mb_n_other_agents,n_reach_goal=n_reach_goal))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward[:self.n_steps].reshape((self.n_envs, self.n_steps)),
                                                masks[:self.n_steps].reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    #explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    #logger.logkv("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                #tracker = SummaryTracker()
                gc.collect()
                # ... some code you want to investigate ...

                #tracker.print_diff()

            callback.on_training_end()

            return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma

    def _run(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """

        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_mpc_actions, mb_n_other_agents = [], [], [], [], [], [], [], []
        mb_states = []
        ep_infos = []

        n_collisions = 0
        n_timeouts = 0
        n_infeasible_sols = 0
        n_reach_goal = 0

        observation = self.env.reset()
        x0 = observation['observation'][0][:3].T

        #self.t = 0


        #mpc = MPCController()
        mpc = MPCSL2()
        dones = 0

        for i in range(self.n_steps):

            # Disable coolision avoidance
            # agents = self.env.unwrapped.envs[0].env.agents
            # ego_agent = agents[0]
            # ego_agent.policy.enable_collision_avoidance = Config.ENABLE_COLLISION_AVOIDANCE

            # Used in case we want to consider a variable number of agents
            # n_other_agents = len(self.env.unwrapped.envs[0].env.agents)-1
            # mb_n_other_agents.append(n_other_agents)
            #self.t += 1
            #reward = 0
            mb_states.append(self.states)

            #actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones,deterministic=False,seq_length=np.ones([self.obs.shape[0]])*(n_other_agents*9))
            actions, values, self.states, neglogpacs = self.model.step(self.obs[:, :3], self.states, self.dones,
                                                                 deterministic=False,
                                                                 seq_length=np.ones(1) * (1 * 9))
            #print("actions", actions)
            mb_mpc_actions.append(0.0)
            # collect data

            #mb_obs.append(self.obs.copy())
            mb_obs.append(self.obs[:, :3].copy())
            mb_actions.append(actions)

            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            # clipped_actions = actions
            # # Clip the actions to avoid out of bound error
            # if isinstance(self.env.action_space, gym.spaces.Box):
            #     clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            # Repeat MPC Step for stability
            #ego_agent.policy.network_output_to_action(0, agents, clipped_actions[0])
            if x0[1] > 6 and x0[1] < 16:
                goal_state = np.append(actions, [[np.pi]], axis=1)
            else:
                goal_state = np.append(actions, [[0]], axis=1)
            goal_state = goal_state[0]

            # action = np.append(actions, [[0]], axis=1)
            # goal_state = action[0]

            # print("observation", self.obs[:, :3])
            # print("goal_state", goal_state)
            # print("x0", x0)

            for j in range(60):


                #optimal_U_opti = mpc.get_optimal_control(x0, goal_state)
                optimal_U_opti = mpc.mpc_output(x0, goal_state)

                self.obs, rewards, self.dones, infos = self.env.unwrapped.envs[0].env.step(optimal_U_opti[:, 0])

                x0 = self.obs['observation'][:3].T

                #reward += rewards

                if np.linalg.norm(x0[:2] - goal_state[:2]) < 0.8:
                    break


            #mb_rewards.append(reward)
            mb_rewards.append(rewards)

            self.obs = self.obs['observation'].reshape(1, -1)
            self.dones = np.array([self.dones])


            self.model.num_timesteps += self.n_envs

            # if ego_agent.in_collision:
            #     n_collisions +=1
            # n_infeasible_sols += ego_agent.is_infeasible
            # n_reach_goal += ego_agent.is_at_goal
            # n_timeouts += ego_agent.ran_out_of_time

            if dones:
                next_obs = self.env.unwrapped.envs[0].env.reset()
                x0 = next_obs['observation'][:3].T
                self.states *= 0.0
                #break



            #if np.linalg.norm(x0[:2] - np.array([8.0, 16.0])) < 0.8 or i >= 2000:
            if np.linalg.norm(x0[:2] - np.array([8.0, 16.0])) < 0.8 or i == self.n_steps-2:

                self.dones = np.array([1])
                #self.dones = True
                dones = np.array([1])


            # if self.dones[0]:
            #     self.states *= 0.0

            if self.callback is not None:
                # Abort training early
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            # for info in infos:
            #     maybe_ep_info = info.get('episode')
            #     if maybe_ep_info is not None:
            #         ep_infos.append(maybe_ep_info)
            #mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_mpc_actions  = np.asarray(mb_mpc_actions, dtype=np.float32)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_n_other_agents = np.asarray(mb_n_other_agents)
        mb_states = np.asarray(mb_states)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        #last_values = self.model.value(self.obs, self.states, self.dones,np.ones([self.obs.shape[0]])*self.obs.shape[1])
        last_values = self.model.value(self.obs[:, :3], self.states, self.dones,
                                       seq_length=np.ones([self.obs.shape[0]]) * (1 * 9))
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values


        mb_obs = np.atleast_2d(mb_obs)
        mb_returns = np.atleast_2d(mb_returns)
        mb_dones = np.atleast_2d(mb_dones[:self.n_steps])
        mb_actions = np.atleast_2d(mb_actions)
        mb_values = np.atleast_2d(mb_values)
        mb_neglogpacs = np.atleast_2d(mb_neglogpacs)
        true_reward = np.atleast_2d(true_reward)

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, np.squeeze(mb_states), ep_infos, true_reward, n_collisions,n_infeasible_sols,n_timeouts,n_reach_goal, mb_mpc_actions, mb_n_other_agents

class MPCRunner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma

    def _run(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_mpc_actions, mb_n_other_agents = [], [], [], [], [], [], [], []
        mb_states = []
        ep_infos = []

        n_collisions = 0
        n_timeouts = 0
        n_infeasible_sols = 0
        n_reach_goal = 0
        step_it = 0

        goal_states = [
            np.array([8., 8., 0.]),
            np.array([0., 16., np.pi / 2]),
            np.array([8., 16., 0])
        ]
        #goal_state = self.goal_states[2]

        controller = MPCSL(goal_states)
        observation = self.env.reset()
        x0 = observation['observation'][0][:3].T


        while step_it < self.n_steps:

            # MPC Action
            # actions sampled expert motion planner
            # agents = self.env.unwrapped.envs[0].env.agents
            # ego_agent = agents[0]
            # agents[0].policy.enable_collision_avoidance = True
            #
            # # Used in case we want to consider a variable number of agents
            # n_other_agents = len(self.env.unwrapped.envs[0].env.agents)-1
            # mb_n_other_agents.append(n_other_agents)
            mb_states.append(self.states)

            #controller = MPCSL(self.goal_states)
            # observation = self.env.reset()
            # x0 = observation['observation'][0][:3].T

            #print(observation['observation'])
            #print("x0", x0)


            optimal_U_opti, actions = controller.mpc_output(x0)
            actions = np.expand_dims(actions, axis=0)
            actions = actions[:, :2]

            #print("actions", actions)
            # TODO: Fix Assumes that agent 0 is the one learning
            #actions, exit_flag = self.env.unwrapped.envs[0].env.agents[0].policy.mpc_output(0, agents)
            #actions = np.expand_dims(actions,axis=0)
            #_, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)

            # print(type(self.obs))
            #print(self.obs)
            #print(self.dones)
            #print(self.obs.shape[0])

            #_, values, self.states, neglogpacs = self.model.step(self.obs[:, :3], self.states, self.dones, deterministic=False, seq_length=np.ones([self.obs.shape[0]])*(1*8))
            _, values, self.states, neglogpacs = self.model.step(self.obs[:, :3], self.states, self.dones,
                                                                 deterministic=False,
                                                                 seq_length=np.ones(1) * (1 * 9))
            #seq_length_array = np.ones([self.obs.shape[0]]) * (1 * 9)
            #print("Shape of seq_length_array:", seq_length_array.shape)
            #seq_length = np.full((2048,), 9)
            # seq_length = np.full((2048,), 9)
            #
            # #seq_length = np.array([9])
            # _, values, self.states, neglogpacs = self.model.step(self.obs[:, :3], self.states, self.dones,
            #                                                      deterministic=False,
            #                                                      seq_length=None)
            #print("Shape of seq_length_array:", seq_length.shape)
            #self.env.render()


            mb_mpc_actions.append(1)
            #mb_obs.append(self.obs.copy())
            mb_obs.append(self.obs[:, :3].copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            # clipped_actions = actions
            # # Clip the actions to avoid out of bound error
            # if isinstance(self.env.action_space, gym.spaces.Box):
            #     clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            # Repeat Step for stability
            #ego_agent.policy.network_output_to_action(0, agents, clipped_actions[0])
            #self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            # TODO: To be fixed 2
            # Apply control input to the environment
            #self.obs, rewards, self.dones, infos = self.env.step(optimal_U_opti[:, 0])
            self.obs, rewards, self.dones, infos = self.env.unwrapped.envs[0].env.step(optimal_U_opti[:, 0])

            x0 = self.obs['observation'][:3].T
            self.obs = self.obs['observation'].reshape(1, -1)
            self.dones = np.array([self.dones])


            if x0[1]>13 and np.linalg.norm(x0[:2] - goal_states[2][:2]) < 0.7:

                next_obs  = self.env.unwrapped.envs[0].env.reset()
                x0 = next_obs['observation'][:3].T
                self.states *= 0.0

                #print("Next obs", next_obs)
                #break
            # if agents[0].in_collision:
            #     n_collisions += 1
            #     step_it -= agents[0].step_num
            #     mb_rewards = mb_rewards[:step_it]
            #     mb_obs = mb_obs[:step_it]
            #     mb_actions = mb_actions[:step_it]
            #     mb_dones = mb_dones[:step_it]
            #     self.model.num_timesteps -= agents[0].step_num
            #     mb_states = mb_states[:step_it]
            #     mb_n_other_agents = mb_n_other_agents[:step_it]
            # else:
            mb_rewards.append(rewards)
            step_it += 1
            # n_infeasible_sols += agents[0].is_infeasible
            # n_reach_goal += agents[0].is_at_goal
            # n_timeouts += agents[0].ran_out_of_time
            self.model.num_timesteps += self.n_envs

            # if self.dones:
            #     self.states *= 0.0

            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            # for info in infos:
            #     maybe_ep_info = info.get('episode')
            #     if maybe_ep_info is not None:
            #         ep_infos.append(maybe_ep_info)

        # batch of steps to batch of rollouts
        mb_mpc_actions = np.asarray(mb_mpc_actions, dtype=np.float32)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_n_other_agents = np.asarray(mb_n_other_agents)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        #last_values = self.model.value(self.obs, self.states, self.dones, np.ones([self.obs.shape[0]])*self.obs.shape[1])
        last_values = self.model.value(self.obs[:, :3], self.states, self.dones,
                                                             seq_length=np.ones([self.obs.shape[0]]) * (1 * 9))


        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        #mb_values = np.zeros([self.n_steps,1])
        for step in reversed(range(self.n_steps)):

            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_neglogpacs = np.zeros([self.n_steps,1])

        mb_returns = np.zeros([self.n_steps,1])


        mb_obs = np.atleast_2d(mb_obs)
        mb_returns = np.atleast_2d(mb_returns)
        mb_dones = np.atleast_2d(mb_dones[:self.n_steps])
        mb_actions = np.atleast_2d(mb_actions)
        mb_values = np.atleast_2d(mb_values)
        mb_neglogpacs = np.atleast_2d(mb_neglogpacs)
        true_reward = np.atleast_2d(true_reward)

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones[:self.n_steps], mb_actions, mb_values, mb_neglogpacs, true_reward))



        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, np.squeeze(mb_states), ep_infos, true_reward, n_collisions,n_infeasible_sols,n_timeouts,n_reach_goal, mb_mpc_actions, mb_n_other_agents

# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
