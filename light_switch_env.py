import gym
from gym import spaces
import numpy as np


class LightSwitchEnv(gym.Env):
    """Custom Environment for controlling a light switch in discrete time intervals."""

    def __init__(self):
        super(LightSwitchEnv, self).__init__()

        # Define the action space: 0 = do nothing, 1 = switch light
        self.action_space = spaces.Discrete(2)

        # Define the observation space:
        # 0 or 1 for light state (off/on), and time step (0 to 143)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(2),  # Light status: 0 = off, 1 = on
            spaces.Discrete(144)  # Time step: 0 to 143 (144 time steps for 24 hours in 10-min intervals)
        ))

        # Initialize the environment state and the history attribute
        self.history = []  # To store light status and time step history
        self.reset()

    def reset(self):
        """Reset the environment to its initial state"""
        self.light_status = 0  # Start with the light off
        self.time_step = 0  # Start at the beginning of the day
        self.positive_reward_count = 0  # Count of consecutive positive feedbacks
        self.holding_time = 0  # Time the agent has held a correct action
        self.history = []  # Reset history at the start of each episode
        return (self.light_status, self.time_step)

    def step(self, action):
        """Take an action in the environment"""

        # Handle the action
        if action == 1:
            # If action is 1, switch the light status (on->off or off->on)
            self.light_status = 1 - self.light_status

        # Simulate user feedback
        feedback = self._get_user_feedback()

        # Calculate reward based on feedback
        if feedback == 1:
            reward = 1  # Positive feedback (user approves the action)
            self.positive_reward_count += 1
        elif feedback == -1:
            reward = -1  # Negative feedback (user disapproves)
            self.positive_reward_count = 0  # Reset positive reward count
            self.holding_time = 0  # Reset holding time for negative feedback
        else:
            reward = 0  # Neutral feedback (no response)

        # After getting 10 positive rewards, agent should hold for 10 time steps
        if self.positive_reward_count >= 10:
            self.holding_time += 1
            if self.holding_time >= 10:
                reward += 10  # Bonus reward for holding correct state
                self.holding_time = 0  # Reset holding time
                self.positive_reward_count = 0  # Reset positive reward count

        # Advance time step
        self.time_step += 1
        done = self.time_step >= 144  # Episode ends after 144 time steps (24 hours)

        # Append the time step and light status to the history list
        self.history.append((self.time_step, self.light_status))

        return (self.light_status, self.time_step), reward, done, {}

    def _get_user_feedback(self):
        """Simulate user feedback based on current time and light status"""
        # For simplicity, let's assume the user prefers the light on at night (time steps 96-143)
        # and off during the day (time steps 0-95).
        if 96 <= self.time_step <= 143:
            # Night time: user wants light on
            if self.light_status == 1:
                return 1  # Positive feedback (light is on as expected)
            else:
                return -1  # Negative feedback (light is off at night)
        else:
            # Day time: user wants light off
            if self.light_status == 0:
                return 1  # Positive feedback (light is off as expected)
            else:
                return -1  # Negative feedback (light is on during the day)

    def render(self, mode="human"):
        """Render the current state of the environment"""
        light_state = "On" if self.light_status == 1 else "Off"
        print(f"Time Step: {self.time_step}, Light: {light_state}")
