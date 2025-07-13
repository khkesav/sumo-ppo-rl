import os
import sys
import traci
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

os.makedirs("outputs/single-intersection", exist_ok=True)
out_csv_name = "outputs/single-intersection/ppo"

# Create SUMO environment
env = SumoEnvironment(
    net_file="sumo_rl/nets/single-intersection/single-intersection.net.xml",
    route_file="sumo_rl/nets/single-intersection/single-intersection.rou.xml",
    out_csv_name=out_csv_name,
    use_gui=True,
    num_seconds=800,
    yellow_time=4,
    min_green=5,
    max_green=60,
    single_agent=True
)

# Load the trained model
model = PPO.load("ppo_single_intersection.zip")

# Run simulation using the trained policy
obs, _ = env.reset()

# Adjust GUI settings
if env.sumo is not None:
    try:
        view_id = traci.gui.getIDList()[0]
        traci.gui.setZoom(view_id, 300)      
        traci.gui.setDelay(view_id, 50)      
        traci.gui.setOffset(view_id, 0, 0)   
    except Exception as e:
        print(f"Error adjusting GUI view: {e}")

done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Save final CSV output
env.save_csv(out_csv_name, 0)
env.close()

print("Evaluation complete using trained PPO model.")