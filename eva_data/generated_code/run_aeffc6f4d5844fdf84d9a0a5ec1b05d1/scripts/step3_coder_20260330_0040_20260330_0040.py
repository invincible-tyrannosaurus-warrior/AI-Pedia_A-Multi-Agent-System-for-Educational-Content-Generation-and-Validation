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

# Create the assets directory if it doesn't exist
assets_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_aeffc6f4d5844fdf84d9a0a5ec1b05d1/assets'
os.makedirs(assets_dir, exist_ok=True)

class Actor(nn.Module):
    """Actor network that outputs action probabilities"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Use softmax to get probability distribution over actions
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    """Critic network that estimates state value"""
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ActorCriticAgent:
    """Actor-Critic agent combining both networks"""
    def __init__(self, state_dim, action_dim, lr_actor=0.001, lr_critic=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Store rewards for plotting
        self.rewards_history = []
        
    def select_action(self, state):
        """Select action based on actor policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.actor(state_tensor)
        # Sample action from probability distribution
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs
    
    def update(self, state, action, reward, next_state, done):
        """Update actor and critic networks"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        # Get current state value and next state value
        current_value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor) if not done else torch.FloatTensor([0]).to(self.device)
        
        # Compute advantage
        advantage = reward_tensor + 0.99 * next_value - current_value
        
        # Update critic
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action_probs = self.actor(state_tensor)
        # Get log probability of selected action
        log_prob = torch.log(action_probs[0][action])
        actor_loss = -log_prob * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss.item(), actor_loss.item()

# Simple environment for demonstration (CartPole-like)
class SimpleEnv:
    """Simple environment with discrete actions"""
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2
        self.reset()
        
    def reset(self):
        # Initialize state with random values
        self.state = np.random.randn(self.state_dim)
        return self.state.copy()
    
    def step(self, action):
        # Simple transition dynamics
        # In a real environment, this would be more complex
        self.state += np.random.randn(self.state_dim) * 0.1
        # Clip state values to reasonable range
        self.state = np.clip(self.state, -5, 5)
        
        # Reward is based on how close we are to zero
        reward = -np.sum(np.abs(self.state))
        
        # Episode ends after 100 steps
        done = False
        self.steps += 1
        if self.steps >= 100:
            done = True
            
        return self.state.copy(), reward, done
    
    def render(self):
        print(f"State: {self.state}")

# Training function
def train_agent(episodes=500):
    env = SimpleEnv()
    agent = ActorCriticAgent(env.state_dim, env.action_dim)
    
    # Lists to store metrics
    episode_rewards = []
    critic_losses = []
    actor_losses = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        num_steps = 0
        
        # Run one episode
        while True:
            # Select action
            action, action_probs = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Store rewards for plotting
            total_reward += reward
            
            # Update agent
            crit_loss, act_loss = agent.update(state, action, reward, next_state, done)
            episode_critic_loss += crit_loss
            episode_actor_loss += act_loss
            
            # Move to next state
            state = next_state
            num_steps += 1
            
            if done:
                break
                
        # Store episode metrics
        episode_rewards.append(total_reward)
        critic_losses.append(episode_critic_loss / num_steps)
        actor_losses.append(episode_actor_loss / num_steps)
        
        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return agent, episode_rewards, critic_losses, actor_losses

# Plotting function
def plot_results(episode_rewards, critic_losses, actor_losses):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Plot critic losses
    ax2.plot(critic_losses)
    ax2.set_title('Critic Losses')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    
    # Plot actor losses
    ax3.plot(actor_losses)
    ax3.set_title('Actor Losses')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'actor_critic_training.png'))
    plt.close()

# Main execution
if __name__ == "__main__":
    print("Training Actor-Critic Agent...")
    
    # Train the agent
    agent, rewards, critic_losses, actor_losses = train_agent(episodes=500)
    
    # Plot results
    plot_results(rewards, critic_losses, actor_losses)
    
    # Print final statistics
    print(f"Final average reward over last 50 episodes: {np.mean(rewards[-50:]):.2f}")
    print(f"Best episode reward: {max(rewards):.2f}")
    
    # Save model weights
    torch.save(agent.actor.state_dict(), os.path.join(assets_dir, 'actor_weights.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(assets_dir, 'critic_weights.pth'))
    
    print("Training completed! Results saved to assets directory.")