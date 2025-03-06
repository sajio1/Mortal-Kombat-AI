import retro
import gym
import numpy as np
import cv2

# -------------------- Preprocessing Functions --------------------
def preprocess_frame(frame):
    """
    Convert the game frame to grayscale, resize, normalize, and reshape.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (84, 84))  # Resize to 84x84
    frame = frame / 255.0  # Normalize pixel values
    return np.expand_dims(frame, axis=0).astype(np.float32)  # Reshape for MLP input

# -------------------- Utility Functions --------------------
def get_health(info):
    """
    Extract player health from game info.
    """
    return info.get('player1_health', 176), info.get('player2_health', 176)

def compute_reward(prev_health, curr_health, info, prev_info):
    """
    Compute reward based on available game info.
    Encourages attacking the enemy, avoiding damage, and winning rounds.
    """
    p1_health, p2_health = curr_health
    prev_p1_health, prev_p2_health = prev_health

    # Health difference reward
    damage_reward = 20 * ((prev_p2_health - p2_health) - (prev_p1_health - p1_health))

    # Round win/loss bonus
    round_win_bonus = 200 * (info.get('rounds_won', 0) - prev_info.get('rounds_won', 0))
    round_loss_penalty = -200 * (info.get('enemy_rounds_won', 0) - prev_info.get('enemy_rounds_won', 0))

    # Total reward
    reward = damage_reward + round_win_bonus + round_loss_penalty

    return reward

# -------------------- Custom Mortal Kombat Gym Environment --------------------
class MortalKombatEnv(gym.Env):
    """
    Custom Gym environment for Mortal Kombat II (Genesis).
    Uses OpenAI Gym's Retro framework.
    """
    def __init__(self):
        super(MortalKombatEnv, self).__init__()
        self.env = retro.make(game='MortalKombatII-Genesis')

        # Define action space (all buttons as discrete actions)
        self.buttons = self.env.unwrapped.buttons
        
        # https://gamefaqs.gamespot.com/genesis/563224-mortal-kombat-ii/faqs/60303

        # https://github.com/vladimirjankov/Mortal-kombat-2-AI-bot/blob/master/scripts/mortalkombat_env.py
        self.valid_actions = [
            # Basic Actions (Individual button presses)
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Press only 'B' - 'Block'
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Press only 'A' - 'Low Punch'
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Press only 'UP'
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Press only 'DOWN'
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Press only 'LEFT'
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Press only 'RIGHT'
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Press only 'C' - 'Low Kick'
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Press only 'Y' - 'Block'
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Press only 'X' - 'High Punch'
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Press only 'Z' - 'High Kick'

            # Combo Moves (Multiple button presses)

            # Crouch + Punch (DOWN + B)
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # Crouch + Punch (DOWN + B)
            
            # Jump + Kick (UP + C)
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  #
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # Jump + Kick (UP + C)

            # Low Fireball: Forward, Forward, Low Punch
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Forwards + Low Punch (Assume RIGHT is forward)
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Forwards + Low Punch (Assume LEFT is forward)

            # High Fireball: Forward, Forward, High Punch
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # Forwards + High Punch (Assume RIGHT is forward)
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # Forwards + High Punch (Assume LEFT is forward)

            # Super Kick: Forward, Forward, High Kick
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Forwards + High Kick (Assume RIGHT is forward)
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # Forwards + High Kick (Assume LEFT is forward)
        ]

        self.action_space = gym.spaces.Discrete(len(self.valid_actions))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 84, 84), dtype=np.float32)  # Grayscale (1, H, W)

        self.prev_health = (176, 176)
        self.prev_info = {'matches_won': 0, 'enemy_matches_won': 0, 'blocked_attacks': 0, 'combo_hits': 0}

    def step(self, action):
        """
        Execute an action in the game and return new state, reward, and done flag.
        """
        mapped_action = self.valid_actions[action]  # Convert Discrete action to MultiBinary
        obs, _, done, info = self.env.step(mapped_action)
        obs = preprocess_frame(obs)

        curr_health = get_health(info)
        reward = compute_reward(self.prev_health, curr_health, info, self.prev_info)

        self.prev_health = curr_health
        self.prev_info = info.copy()  # Copy to prevent reference issues

        return obs, reward, done, info

    def reset(self):
        """
        Reset the environment to start a new episode.
        """
        obs = self.env.reset()
        obs = preprocess_frame(obs)
        self.prev_health = (176, 176)
        return obs

    def render(self, mode='human'):
        """
        Render the game screen.
        """
        self.env.render()

    def close(self):
        """
        Close the environment.
        """
        self.env.close()
