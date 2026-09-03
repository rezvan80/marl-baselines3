import warnings
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: gym.Env | VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    reward_threshold: float | None = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> tuple[float, float] | tuple[list[float], list[int]]:
    """
    Runs the policy for ``n_eval_episodes`` episodes and outputs the average return
    per episode (sum of undiscounted rewards).
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to perform additional checks,
        called ``n_envs`` times after each step.
        Gets locals() and globals() passed as parameters.
        See https://github.com/DLR-RM/stable-baselines3/issues/1912 for more details.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean return per episode (sum of rewards), std of reward per episode.
        Returns (list[float], list[int]) when ``return_episode_rewards`` is True, first
        list containing per-episode return and second containing per-episode lengths
        (in number of steps).
    """


    n_envs = env.num_envs
    episode_rewards = []
    episode_queue_lengths = []
    episode_queue_nums = []
    episode_waiting_times = []
    episode_travel_times = []
    episode_lengths = []
    episode_counts = 0
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = 10

    current_rewards = 0
    current_queue_lengths = []
    current_waiting_times = []
    current_travel_times = 0
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets):
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones , queue_lengths , waiting_times , travel_times, infos = env.step(actions)
        current_rewards += rewards.sum()
        current_queue_lengths.append(queue_lengths)
        current_waiting_times.append(waiting_times)
        current_travel_times += travel_times
        current_lengths += 1
        
        if episode_counts < episode_count_targets:
                # unpack values so that the callback can access the local variables
                reward = rewards
                done = dones
                info = infos
                episode_starts = done

                if callback is not None:
                    callback(locals(), globals())

                if dones.all():
                    

                    
                    episode_rewards.append(current_rewards)
                    episode_queue_lengths.append(current_queue_lengths)
                    episode_waiting_times.append(current_waiting_times)
                    episode_travel_times.append(current_travel_times)
                    episode_lengths.append(current_lengths)

                    episode_counts += 1
                    current_rewards = 0
                    current_queue_lengths = []
                    current_waiting_timess = []
                    current_travel_times = 0
                    current_lengths = 0

        observations = new_observations



    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_queue_length = np.mean(episode_queue_lengths)
    std_queue_length = np.std(episode_queue_lengths)
    mean_queue_num = np.mean(episode_queue_nums)
    std_queue_num = np.std(episode_queue_nums)
    mean_waiting_time = np.mean(episode_waiting_times)
    std_waiting_time = np.std(episode_waiting_times)
    mean_travel_time = np.mean(episode_travel_times)
    std_travel_time = np.std(episode_travel_times)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward, mean_queue_length, std_queue_length, mean_queue_num, std_queue_num, mean_waiting_time, std_waiting_time, mean_travel_time, std_travel_time 
