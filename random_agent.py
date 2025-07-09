import os
import sys
import numpy as np
from sumo_rl import SumoEnvironment

# Ensure SUMO_HOME is set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

env = SumoEnvironment(
    net_file="sumo_rl/nets/single-intersection/single-intersection.net.xml",
    route_file="sumo_rl/nets/single-intersection/single-intersection.rou.xml",
    out_csv_name="outputs/single-intersection/random",
    use_gui=True,
    num_seconds=5400,
    yellow_time=4,
    min_green=5,
    max_green=60,
    single_agent=True
)

obs = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"✅ Random Agent Simulation Finished. Total Reward: {total_reward}")
