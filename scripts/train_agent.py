from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    import argparse
    from ostatslib.agents import PPOAgent
    from datetime import datetime

    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Agent training script")
    parser.add_argument("--steps", nargs='?', const=1,
                        type=int, default=int(500e3))
    parser.add_argument("--save-freq", nargs='?', const=1,
                        type=int, default=int(25e3))
    parser.add_argument("--env-count", nargs='?',
                        const=1, type=int, default=10)
    parser.add_argument("--name", nargs='?', const=1,
                        type=str, default=f'ppo_agent{datetime.now()}')
    parser.add_argument("--model-path", nargs='?', const=1,
                        type=str, default=None)

    args = parser.parse_args()

    if args.model_path is None:
        agent = PPOAgent(training_envs_count=int(args.env_count))
    else:
        agent = PPOAgent(args.model_path)

    agent.train(int(args.steps), int(args.save_freq))
    agent.save(args.name)
