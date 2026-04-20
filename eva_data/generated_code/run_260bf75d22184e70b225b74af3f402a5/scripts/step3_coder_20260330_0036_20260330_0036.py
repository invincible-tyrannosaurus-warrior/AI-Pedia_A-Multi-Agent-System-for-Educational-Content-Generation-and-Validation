import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Create the directory for saving assets
assets_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_260bf75d22184e70b225b74af3f402a5/assets'
os.makedirs(assets_dir, exist_ok=True)

class Actor(nn.Module):
    """
    Actor network that outputs action probabilities for each action
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        # Define the neural network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        # Forward pass through the network
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Use softmax to get probability distribution over actions
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    """
    Critic network that evaluates the value of states
    """
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        # Define the neural network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output single value (state value)
        
    def forward(self, state):
        # Forward pass through the network
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ActorCriticAgent:
    """
    Actor-Critic agent that combines both networks
    """
    def __init__(self, state_dim, action_dim, lr_actor=0.001, lr_critic=0.001):
        # Initialize networks
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Store rewards and values for training
        self.rewards = []
        self.values = []
        self.actions = []
        self.log_probs = []
        
    def select_action(self, state):
        """
        Select an action based on the current policy (actor)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        # Sample action from the probability distribution
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[0, action])
        
        # Store for training
        self.actions.append(action)
        self.log_probs.append(log_prob)
        
        return action
    
    def store_transition(self, reward, value):
        """
        Store reward and value for later training
        """
        self.rewards.append(reward)
        self.values.append(value)
    
    def compute_returns(self, gamma=0.99):
        """
        Compute discounted returns for each time step
        """
        returns = []
        R = 0
        # Calculate returns backwards in time
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.FloatTensor(returns)
    
    def train(self, gamma=0.99, entropy_weight=0.01):
        """
        Train the actor and critic networks
        """
        # Compute returns
        returns = self.compute_returns(gamma)
        
        # Convert stored values to tensors
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        
        # Compute advantages
        advantages = returns - values.squeeze()
        
        # Actor loss (policy gradient with advantage)
        actor_loss = (-log_probs * advantages).mean()
        
        # Critic loss (mean squared error)
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy bonus to encourage exploration
        action_probs = self.actor(torch.FloatTensor(np.array(self.states)))
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(-1).mean()
        actor_loss -= entropy_weight * entropy
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Clear stored transitions
        self.rewards = []
        self.values = []
        self.actions = []
        self.log_probs = []
        self.states = []

# Simple environment for demonstration (CartPole-like)
class SimpleEnv:
    """
    A simple environment for demonstration purposes
    """
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2
        self.state = None
        self.reset()
        
    def reset(self):
        # Initialize state randomly
        self.state = np.random.randn(self.state_dim)
        return self.state
    
    def step(self, action):
        # Simple reward function
        reward = 1.0 if action == 1 else 0.0  # Reward for taking action 1
        # Simple transition dynamics
        self.state += np.random.randn(self.state_dim) * 0.1
        done = False
        return self.state, reward, done

def train_agent(episodes=1000):
    """
    Main training loop for the Actor-Critic agent
    """
    env = SimpleEnv()
    agent = ActorCriticAgent(env.state_dim, env.action_dim)
    
    # Track performance
    episode_rewards = []
    moving_avg_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # Collect trajectory
        while not done:
            # Select action using actor network
            action = agent.select_action(state)
            
            # Execute action in environment
            next_state, reward, done = env.step(action)
            
            # Get value estimate from critic
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = agent.critic(state_tensor).item()
            
            # Store transition
            agent.store_transition(reward, value)
            
            state = next_state
            total_reward += reward
            
        # Train agent at the end of episode
        agent.train()
        
        # Store rewards for analysis
        episode_rewards.append(total_reward)
        
        # Calculate moving average
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            moving_avg_rewards.append(avg_reward)
        else:
            moving_avg_rewards.append(np.mean(episode_rewards))
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Avg Reward (100): {moving_avg_rewards[-1]:.2f}")
    
    return episode_rewards, moving_avg_rewards

def plot_results(episode_rewards, moving_avg_rewards):
    """
    Plot training results
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.5, label='Episode Rewards')
    plt.plot(moving_avg_rewards, label='Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Moving average only
    plt.subplot(1, 2, 2)
    plt.plot(moving_avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (100 episodes)')
    plt.title('Moving Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'actor_critic_training_results.png'))
    plt.close()

# Run the training
if __name__ == "__main__":
    print("Starting Actor-Critic training...")
    episode_rewards, moving_avg_rewards = train_agent(episodes=1000)
    plot_results(episode_rewards, moving_avg_rewards)
    print("Training completed. Results saved to assets directory.")