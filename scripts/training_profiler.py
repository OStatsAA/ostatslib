import cProfile
from ostatslib.agents import PPOAgent


agent = PPOAgent(training_envs_count=4)
cProfile.run('agent.train(10000, 5000)', 'cProfiler2')
agent.save('cProfiler_trained_agent')
