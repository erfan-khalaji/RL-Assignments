# main.py

from light_switch_env import LightSwitchEnv
import matplotlib.pyplot as plt

# Create the environment
env = LightSwitchEnv()

# Initialize variables
state = env.reset()  # Reset the environment to the initial state
done = False  # Track if the episode is done
total_reward = 0  # Keep track of total rewards

# Run through the episode (one full day of 144 time steps)
while not done:
    action = env.action_space.sample()  # Random action for now (replace with policy later)

    # Take a step in the environment with the chosen action
    next_state, reward, done, _ = env.step(action)

    # Render the environment state (prints to console)
    env.render()

    # Accumulate total reward for tracking
    total_reward += reward

# After the episode ends, we can visualize the light status over time using matplotlib
# Extract the time steps and light statuses from the environment history
time_steps, light_statuses = zip(*env.history)  # Unpack the history to get time steps and statuses

# Plot the result
plt.plot(time_steps, light_statuses, drawstyle='steps-post', marker='o', label='Light Status')
plt.xlabel('Time Steps')
plt.ylabel('Light Status (0 = Off, 1 = On)')
plt.title('Light Switch Status Over Time')
plt.grid(True)
plt.legend()
plt.show()

# Print the total reward accumulated during the episode
print(f"Total reward for the episode: {total_reward}")
