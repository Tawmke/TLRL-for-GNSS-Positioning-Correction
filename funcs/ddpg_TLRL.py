from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import torch as th
from torch.nn import functional as F
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps

DDPGSelf = TypeVar("DDPGSelf", bound="DDPG")


class DDPG(TD3):
    """
    Deep Deterministic Policy Gradient (DDPG).

    Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    Note: we treat DDPG as a special case of its successor TD3.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically (Only available when passing string for the environment).
        Caution, this parameter is deprecated and will be removed in the future.
        Please use `EvalCallback` or a custom Callback instead.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        value_w,  # 价值权重
        value_w_target,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        B: np.ndarray = None,
        value_step_size: float = 1e-3,  # 价值权重步长
        betaFi_control: float = None,  # 目标价值权重
        reg_method: str = None,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )

        # set parameters for TLRL
        self.book = {}
        self.B = B # TLRL字典
        self.value_w = value_w
        self.value_w_target = value_w_target
        self.value_step_size = value_step_size
        self.betaFi_control = betaFi_control
        self.reg_method = reg_method
        self.value_optimizer = th.optim.SGD(self.value_w.parameters(), lr=self.value_step_size)
        self.episode_rewards = [] # record rewards for sp
        self.culReward = 0

        # Use only one critic
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(buffer_actions)
            # reward episode rewards
            self.culReward = self.culReward + rewards[-1]
            if dones[-1]:
                self.episode_rewards.append(self.culReward)
                print(f'ep:{len(self.episode_rewards)}, culReward={self.culReward}')
                self.culReward = 0

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _encoder(self, observation):
        # observation_ = self.basic.net(torch.from_numpy(observation.copy()).to(torch.float64)).detach().numpy()
        observation_  = observation.flatten() # flatten
        observation_encode = self.getSparseForm(observation_)
        return np.squeeze(observation_encode)

    def getSparseForm(self, observation):
        tempObservation = observation.copy()
        for index, element in enumerate(tempObservation):
            tempObservation[index] = round(element, 7) # 将每个元素四舍五入到小数点后7位
        temp = "*".join(tempObservation.astype('str').tolist()) # 将数值数组转换为字符串数组,用星号连接所有字符串
        if temp in self.book.keys():
            return self.book[temp] # 如果状态已经存在，则不需要重新进行稀疏表示
        # result = self.quickGetSparseForm(observation)
        # result = self.quickGetSparseForm_L1(observation)
        if self.reg_method == 'L0':
            SparseForm = np.dot(observation, self.B).T
            zeros = np.zeros_like(SparseForm)
            result = np.where(np.abs(SparseForm) > self.betaFi_control, SparseForm, zeros)

        self.book[temp] = result
        return self.book[temp]

    def predict_value(self,obs: np.ndarray):
        z = self._encoder(obs)
        z = th.unsqueeze(th.FloatTensor(z), 0).cuda()
        prd_value = self.value_w.forward(z)
        values = prd_value.detach().cpu()
        return values

    def predict_batch_value(self,obs: th.tensor, act:th.tensor, target: bool):
        obs = obs.detach().cpu().numpy()
        batch_list = []
        for i in range(obs.shape[0]):
            _obs = obs[i,:,:]
            z = self._encoder(_obs)
            z = th.unsqueeze(th.FloatTensor(z), 0).cuda()
            batch_list.append(z)
        z = th.stack(batch_list, dim=0)
        value_input = th.cat([z, act.unsqueeze(1)], dim=2)
        if target:
            prd_value = self.value_w_target.forward(value_input)
        else:
            prd_value = self.value_w.forward(value_input)
        return prd_value.squeeze(1)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_obs_32 = replay_data.next_observations.float()
                next_actions_32  = next_actions.float()
                next_q_values = self.predict_batch_value(next_obs_32, next_actions_32, True)
                # next_q_values = th.cat(self.critic_target(next_obs_32, next_actions_32), dim=1)
                #next_q_values, _ = th.min(next_q_values, dim=2, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            next_obs_32 = replay_data.observations.float()
            next_actions_32 = replay_data.actions.float()
            current_q_values = self.predict_batch_value(next_obs_32, next_actions_32, False)

            # Compute critic loss
            critic_loss = F.mse_loss(current_q_values, target_q_values) #sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.value_optimizer.zero_grad()
            critic_loss.backward()
            self.value_optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.predict_batch_value(replay_data.observations,self.actor(replay_data.observations),False).mean()
                # actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                self.value_w_target.load_state_dict(self.value_w.state_dict())
                # polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

                # Copy running stats, see GH issue #996
                # polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


    def learn(
        self: DDPGSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DDPG",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> DDPGSelf:

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self.episode_rewards
