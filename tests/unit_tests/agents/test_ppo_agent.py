from ostatslib.agents.ppo_agent import PPOAgent
from ostatslib.environments.data_generators import datacooker_generator, sklearn_generator


def test_training() -> None:
    agent = PPOAgent(training_envs_count=2,
                     environment_kwargs={'data_generators': [datacooker_generator, sklearn_generator]})
    agent.train(steps=1024, save_freq=512)
    agent.analyze(*datacooker_generator())
