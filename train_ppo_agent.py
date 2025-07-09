import os
import sys
import pandas as pd
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

# Ensure SUMO_HOME is set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Make sure output directory exists
out_csv_dir = "outputs/single-intersection"
os.makedirs(out_csv_dir, exist_ok=True)

os.makedirs("outputs/single-intersection/train-ppo", exist_ok=True)
out_csv_name = "outputs/single-intersection/train-ppo/ppo"

# Create SUMO environment
env = SumoEnvironment(
    net_file="sumo_rl/nets/single-intersection/single-intersection.net.xml",
    route_file="sumo_rl/nets/single-intersection/single-intersection.rou.xml",
    out_csv_name=out_csv_name,
    use_gui=False,
    num_seconds=400,
    yellow_time=4,
    min_green=5,
    max_green=60,
    single_agent=True
)

# Create PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=2.5e-4,
    verbose=1,
)

# Train PPO agent
model.learn(total_timesteps=50_000)
model.save("ppo_single_intersection")

# Save final CSV output
env.save_csv(out_csv_name, 0)
env.close()

print("PPO Training Complete. Model saved as 'ppo_single_intersection'.")
print(f"Output CSV saved to: {out_csv_name}_ep_0.csv")