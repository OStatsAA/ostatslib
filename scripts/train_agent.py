from ostatslib.agents import PPOAgent
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


agent = PPOAgent(training_envs_count=2)
agent.train()
agent.save(f'ppo_agent_{datetime.now()}')