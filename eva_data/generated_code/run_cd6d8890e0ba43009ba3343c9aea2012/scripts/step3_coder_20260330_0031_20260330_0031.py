import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class QLearningAgent:
    """
    Q-Learning Agent implementation
    """
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize the Q-Learning agent
        
        Args:
            state_size: Number of possible states
            action_size: Number of possible actions
            learning_rate: How quickly the agent learns (alpha)
            discount_factor: Importance of future rewards (gamma)
            epsilon: Exploration rate (probability of taking random action)
            epsilon_decay: Rate at which exploration decreases
            epsilon_min: Minimum exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
        
    def get_action(self, state):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            action: Chosen action
        """
        # With probability epsilon, choose random action (exploration)
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            # Otherwise, choose best action according to Q-table (exploitation)
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning update rule
        
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Get current Q-value
        current_q = self.q_table[state, action]
        
        # Get maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_state])
        
        # Calculate target Q-value
        target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value using learning rate
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state, action] = new_q
    
    def decay_epsilon(self):
        """
        Decay exploration rate over time
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SimpleEnvironment:
    """
    Simple environment for demonstration (Grid world)
    """
    def __init__(self):
        # Define grid world dimensions
        self.rows = 4
        self.cols = 4
        self.state_size = self.rows * self.cols
        self.action_size = 4  # Up, Down, Left, Right
        
        # Define actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        # Define rewards and terminal states
        self.rewards = np.full((self.rows, self.cols), -0.1)  # Default reward
        self.rewards[0, 3] = 1.0  # Terminal state with positive reward
        self.rewards[1, 1] = -1.0  # Obstacle with negative reward
        self.rewards[2, 1] = -1.0  # Obstacle with negative reward
        
        # Define terminal states
        self.terminal_states = [(0, 3)]
        self.obstacles = [(1, 1), (2, 1)]
    
    def get_state_index(self, row, col):
        """Convert 2D coordinates to 1D state index"""
        return row * self.cols + col
    
    def get_position(self, state_index):
        """Convert 1D state index to 2D coordinates"""
        row = state_index // self.cols
        col = state_index % self.cols
        return row, col
    
    def step(self, state_index, action):
        """
        Take an action in the environment
        
        Args:
            state_index: Current state
            action: Action to take
            
        Returns:
            next_state: Next state index
            reward: Reward received
            done: Whether episode is finished
        """
        row, col = self.get_position(state_index)
        
        # Apply action
        delta_row, delta_col = self.actions[action]
        new_row = row + delta_row
        new_col = col + delta_col
        
        # Check if move is valid (within bounds)
        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            # Check if new position is obstacle
            if (new_row, new_col) not in self.obstacles:
                row, col = new_row, new_col
        
        # Convert back to state index
        next_state = self.get_state_index(row, col)
        
        # Get reward for the new state
        reward = self.rewards[row, col]
        
        # Check if we reached terminal state
        done = (row, col) in self.terminal_states
        
        return next_state, reward, done

def train_agent(episodes=1000):
    """
    Train the Q-Learning agent
    
    Args:
        episodes: Number of training episodes
        
    Returns:
        agent: Trained agent
        rewards_history: List of total rewards per episode
    """
    # Create environment
    env = SimpleEnvironment()
    
    # Create agent
    agent = QLearningAgent(env.state_size, env.action_size)
    
    # Track rewards over episodes
    rewards_history = []
    
    # Training loop
    for episode in range(episodes):
        # Reset environment to starting state
        state = env.get_state_index(3, 0)  # Start at bottom-left corner
        total_reward = 0
        done = False
        
        # Run one episode until termination
        while not done:
            # Choose action
            action = agent.get_action(state)
            
            # Take action and observe result
            next_state, reward, done = env.step(state, action)
            
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state)
            
            # Move to next state
            state = next_state
            total_reward += reward
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Store total reward for this episode
        rewards_history.append(total_reward)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode}, Average Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, rewards_history

def visualize_results(agent, rewards_history):
    """
    Visualize training results and Q-table
    
    Args:
        agent: Trained agent
        rewards_history: List of rewards per episode
    """
    # Plot rewards over time
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Rewards over episodes
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot 2: Moving average of rewards
    plt.subplot(1, 2, 2)
    window_size = 50
    moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg)
    plt.title(f'Moving Average Reward (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_cd6d8890e0ba43009ba3343c9aea2012/assets/q_learning_training.png')
    plt.close()
    
    # Visualize Q-table
    plt.figure(figsize=(10, 8))
    
    # Create a heatmap of Q-values for each state-action pair
    q_values = agent.q_table.reshape(4, 4, 4)  # Reshape to 4x4x4
    
    # Plot Q-values for each action
    actions = ['Up', 'Down', 'Left', 'Right']
    for i, action in enumerate(actions):
        plt.subplot(2, 2, i+1)
        plt.imshow(q_values[:, :, i], cmap='coolwarm', interpolation='nearest')
        plt.title(f'Q-Values for Action: {action}')
        plt.colorbar()
        
        # Add text annotations for better visualization
        for r in range(4):
            for c in range(4):
                plt.text(c, r, f'{q_values[r,c,i]:.2f}', 
                        ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_cd6d8890e0ba43009ba3343c9aea2012/assets/q_table_visualization.png')
    plt.close()

def demonstrate_policy(agent, env):
    """
    Demonstrate the learned policy by following optimal path
    
    Args:
        agent: Trained agent
        env: Environment
    """
    print("\nDemonstrating learned policy:")
    print("Starting from bottom-left corner (3,0)")
    
    state = env.get_state_index(3, 0)  # Start position
    path = [state]
    steps = 0
    max_steps = 20
    
    while steps < max_steps:
        # Get optimal action for current state
        action = np.argmax(agent.q_table[state])
        
        # Take action
        next_state, _, done = env.step(state, action)
        
        # Add to path
        path.append(next_state)
        
        # Check if terminal state
        if done:
            print(f"Reached terminal state after {steps+1} steps")
            break
            
        state = next_state
        steps += 1
    
    # Convert path to readable format
    positions = [env.get_position(s) for s in path]
    print(f"Optimal path: {positions}")

# Main execution
if __name__ == "__main__":
    print("Starting Q-Learning Training...")
    
    # Train the agent
    trained_agent, rewards = train_agent(episodes=1000)
    
    # Create environment for demonstrations
    environment = SimpleEnvironment()
    
    # Visualize results
    visualize_results(trained_agent, rewards)
    
    # Demonstrate learned policy
    demonstrate_policy(trained_agent, environment)
    
    # Print final Q-table statistics
    print("\nFinal Q-Table Statistics:")
    print(f"Max Q-value: {np.max(trained_agent.q_table):.3f}")
    print(f"Min Q-value: {np.min(trained_agent.q_table):.3f}")
    print(f"Mean Q-value: {np.mean(trained_agent.q_table):.3f}")
    
    # Save final Q-table to CSV
    np.savetxt('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_cd6d8890e0ba43009ba3343c9aea2012/assets/final_q_table.csv', 
               trained_agent.q_table, delimiter=',')
    
    print("\nTraining completed! Results saved to assets folder.")