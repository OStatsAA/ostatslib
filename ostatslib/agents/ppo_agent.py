"""
PPO Agent module
"""

from numpy import ndarray
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ostatslib.agents.agent import Agent
from ostatslib.environments import GymEnvironment


class PPOAgent(Agent):
    """
    Agent built on PPO algorithm model
    """

    def __init__(self, path: str | None = None, training_envs_count: int = 10) -> None:
        self.__model = self.__init__model(path, training_envs_count)

    def train(self, steps: int = 100000) -> None:
        self.__model.learn(total_timesteps=steps)

    def save(self, path: str) -> None:
        self.__model.save(path)

    def _predict(self, observation: dict) -> ndarray:
        action, _ = self.__model.predict(observation, deterministic=True)
        return action[0]

    def __init__model(self, path, training_envs_count) -> PPO:
        if path is None:
            environments = make_vec_env(GymEnvironment,
                                        n_envs=training_envs_count)
            return PPO("MultiInputPolicy", environments, verbose=1)

        return PPO.load(path)
