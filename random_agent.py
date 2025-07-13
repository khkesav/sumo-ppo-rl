import os
import sys
import traci
import numpy as np
from sumo_rl import SumoEnvironment

# Make sure output directory exists
os.makedirs("outputs/single-intersection", exist_ok=True)

# Ensure SUMO_HOME is set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Correct out_csv_name (remove trailing comma to avoid tuple)
out_csv_name = "outputs/single-intersection/random"

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

# Run simulation
obs = env.reset()

# Access traci via env.sumo
if env.sumo is not None:
    try:
        view_id = traci.gui.getIDList()[0]
        traci.gui.setZoom(view_id, 300)
        traci.gui.setDelay(view_id, 50)      
        traci.gui.setOffset(view_id, 0, 0)
    except Exception as e:
        print(f"Error adjusting GUI view: {e}")


print(f"Expected CSV path: {env.out_csv_name}_ep_0.csv")
print(f"Vehicles in simulation: {env.vehicles}")

done = False
total_reward = 0
step = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step += 1
    if step % 10 == 0:
        print(f"Step {step}: Action={action}, Reward={reward}")

# Save results manually (as in Q-Learning experiments)
env.save_csv(out_csv_name, 0)
env.close()

print(f"Random Agent Simulation Finished in {step} steps. Total Reward: {total_reward}")