import argparse
from ostatslib.agents import PPOAgent
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Agent training script")
parser.add_argument("--steps", nargs='?', const=1, type=int, default=int(250e3))
parser.add_argument("--env-count", nargs='?', const=1, type=int, default=4)
parser.add_argument("--name", nargs='?', const=1, type=str, default=f'ppo_agent{datetime.now()}')

args = parser.parse_args()

agent = PPOAgent(training_envs_count=int(args.env_count))
agent.train(int(args.steps))
agent.save(args.name)