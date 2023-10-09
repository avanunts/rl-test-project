
def print_last_episode_logs(logs):
    print(
"""
Episode reward:                       {:.1f}
Episode num steps before termination: {}
Episode lr:                           {:.10f}
""".format(logs['reward'][-1], logs['step_count'][-1], logs['lr'][-1])
          )
