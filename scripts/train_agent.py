from ostatslib.agents import PPOAgent
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


agent = PPOAgent(training_envs_count=4)
agent.train(200000)
agent.save(f'ppo_agent_{datetime.now()}')