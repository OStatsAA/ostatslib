from ostatslib.agents import PPOAgent
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


agent = PPOAgent(training_envs_count=4)
agent.train(int(250e3))
agent.save(f'ppo_agent_{datetime.now()}')