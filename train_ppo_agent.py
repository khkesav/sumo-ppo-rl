import os
import sys
from stable_baselines3 import PPO
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
    out_csv_name="outputs/single-intersection/ppo",
    use_gui=True,
    num_seconds=5400,
    yellow_time=4,
    min_green=5,
    max_green=60,
    single_agent=True
)

model = PPO(
    env=env,
    policy="MlpPolicy",
    learning_rate=2.5e-4,
    verbose=1,
)

model.learn(total_timesteps=100_000)
model.save("ppo_big_intersection")

print("✅ PPO Training Complete. Model saved as 'ppo_big_intersection'.")
