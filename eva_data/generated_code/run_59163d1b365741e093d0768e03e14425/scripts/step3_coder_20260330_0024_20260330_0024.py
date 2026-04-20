import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class GridWorldMDP:
    """
    A simple 4x4 grid world MDP for demonstration purposes.
    States are represented as tuples (row, col).
    Actions: 0=up, 1=right, 2=down, 3=left
    """
    
    def __init__(self):
        # Define grid dimensions
        self.rows = 4
        self.cols = 4
        
        # Define states (all grid positions)
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        
        # Define actions (up, right, down, left)
        self.actions = [0, 1, 2, 3]
        
        # Define terminal states
        self.terminal_states = [(0, 3), (3, 0)]
        
        # Define reward for each state
        self.rewards = {}
        for state in self.states:
            if state == (0, 3):  # Terminal state with reward +10
                self.rewards[state] = 10
            elif state == (3, 0):  # Terminal state with reward -10
                self.rewards[state] = -10
            else:  # All other states have reward -1
                self.rewards[state] = -1
                
        # Define transition probabilities (deterministic for simplicity)
        # Each action leads to the adjacent cell in that direction
        self.transition_probabilities = {}
        for state in self.states:
            self.transition_probabilities[state] = {}
            for action in self.actions:
                self.transition_probabilities[state][action] = self._get_next_state(state, action)
                
    def _get_next_state(self, state, action):
        """Get next state given current state and action"""
        row, col = state
        
        # Apply action
        if action == 0:  # Up
            new_row = max(0, row - 1)
            new_col = col
        elif action == 1:  # Right
            new_row = row
            new_col = min(self.cols - 1, col + 1)
        elif action == 2:  # Down
            new_row = min(self.rows - 1, row + 1)
            new_col = col
        elif action == 3:  # Left
            new_row = row
            new_col = max(0, col - 1)
            
        # If we hit a wall, stay in place
        return (new_row, new_col)
    
    def get_reward(self, state):
        """Return reward for being in a particular state"""
        return self.rewards[state]
    
    def is_terminal(self, state):
        """Check if state is terminal"""
        return state in self.terminal_states
    
    def get_transition_prob(self, state, action, next_state):
        """Get probability of transitioning from state to next_state via action"""
        # In this deterministic environment, we return 1 if it's the expected next state
        expected_next = self.transition_probabilities[state][action]
        return 1.0 if expected_next == next_state else 0.0

def value_iteration(mdp, gamma=0.9, theta=1e-6):
    """
    Perform Value Iteration to find optimal policy
    
    Args:
        mdp: The MDP object
        gamma: Discount factor
        theta: Convergence threshold
    
    Returns:
        values: Dictionary mapping states to their value
        policy: Dictionary mapping states to best action
    """
    
    # Initialize value function
    values = {state: 0.0 for state in mdp.states}
    
    # Keep iterating until convergence
    while True:
        delta = 0
        for state in mdp.states:
            if mdp.is_terminal(state):
                continue
                
            # Get all possible actions from this state
            action_values = []
            for action in mdp.actions:
                # Calculate expected value of taking this action
                expected_value = 0
                for next_state in mdp.states:
                    prob = mdp.get_transition_prob(state, action, next_state)
                    reward = mdp.get_reward(next_state)
                    expected_value += prob * (reward + gamma * values[next_state])
                action_values.append(expected_value)
            
            # Update value to maximum action value
            old_value = values[state]
            values[state] = max(action_values)
            delta = max(delta, abs(old_value - values[state]))
        
        # Check for convergence
        if delta < theta:
            break
    
    # Extract optimal policy
    policy = {}
    for state in mdp.states:
        if mdp.is_terminal(state):
            policy[state] = None
            continue
            
        # Find action that maximizes value
        action_values = []
        for action in mdp.actions:
            expected_value = 0
            for next_state in mdp.states:
                prob = mdp.get_transition_prob(state, action, next_state)
                reward = mdp.get_reward(next_state)
                expected_value += prob * (reward + gamma * values[next_state])
            action_values.append(expected_value)
        
        # Choose action with highest value
        best_action = np.argmax(action_values)
        policy[state] = best_action
    
    return values, policy

def plot_policy_and_values(mdp, values, policy, filename):
    """
    Plot the policy and value function on the grid
    
    Args:
        mdp: The MDP object
        values: Dictionary of state values
        policy: Dictionary of optimal actions
        filename: Path to save the plot
    """
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot values
    values_grid = np.zeros((mdp.rows, mdp.cols))
    for state, value in values.items():
        values_grid[state[0], state[1]] = value
    
    im1 = ax[0].imshow(values_grid, cmap='coolwarm', interpolation='nearest')
    ax[0].set_title('State Values')
    ax[0].set_xlabel('Column')
    ax[0].set_ylabel('Row')
    
    # Add text annotations for values
    for i in range(mdp.rows):
        for j in range(mdp.cols):
            ax[0].text(j, i, f'{values[(i,j)]:.1f}', 
                      ha='center', va='center', fontsize=8)
    
    # Add colorbar
    plt.colorbar(im1, ax=ax[0])
    
    # Plot policy
    policy_grid = np.zeros((mdp.rows, mdp.cols))
    for state, action in policy.items():
        if action is not None:
            policy_grid[state[0], state[1]] = action
    
    im2 = ax[1].imshow(policy_grid, cmap='tab10', interpolation='nearest')
    ax[1].set_title('Optimal Policy')
    ax[1].set_xlabel('Column')
    ax[1].set_ylabel('Row')
    
    # Add arrows for policy directions
    for i in range(mdp.rows):
        for j in range(mdp.cols):
            if not mdp.is_terminal((i, j)):
                action = policy[(i, j)]
                if action == 0:  # Up
                    ax[1].arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='black')
                elif action == 1:  # Right
                    ax[1].arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='black')
                elif action == 2:  # Down
                    ax[1].arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='black')
                elif action == 3:  # Left
                    ax[1].arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='black')
    
    # Add colorbar
    plt.colorbar(im2, ax=ax[1])
    
    # Save plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def simulate_episode(mdp, policy, start_state=None):
    """
    Simulate one episode following the policy
    
    Args:
        mdp: The MDP object
        policy: Policy dictionary
        start_state: Starting state (default: random)
    
    Returns:
        states_visited: List of states visited during episode
        total_reward: Total reward collected
    """
    if start_state is None:
        # Start at a random non-terminal state
        start_state = random.choice([s for s in mdp.states if not mdp.is_terminal(s)])
    
    states_visited = [start_state]
    total_reward = 0
    current_state = start_state
    
    # Run up to 20 steps to avoid infinite loops
    for _ in range(20):
        if mdp.is_terminal(current_state):
            break
            
        # Take action according to policy
        action = policy[current_state]
        next_state = mdp.transition_probabilities[current_state][action]
        
        # Collect reward
        reward = mdp.get_reward(next_state)
        total_reward += reward
        
        # Move to next state
        current_state = next_state
        states_visited.append(current_state)
    
    return states_visited, total_reward

def main():
    """Main function to run the MDP example"""
    
    # Create MDP instance
    mdp = GridWorldMDP()
    
    print("Grid World MDP Setup:")
    print(f"States: {len(mdp.states)}")
    print(f"Actions: {len(mdp.actions)}")
    print(f"Terminal states: {mdp.terminal_states}")
    print()
    
    # Perform Value Iteration
    print("Performing Value Iteration...")
    values, policy = value_iteration(mdp, gamma=0.9)
    
    # Print some results
    print("Optimal Values for Key States:")
    key_states = [(0,0), (0,1), (1,1), (2,2), (3,3)]
    for state in key_states:
        print(f"State {state}: Value = {values[state]:.2f}")
    
    print("\nOptimal Policy for Key States:")
    for state in key_states:
        if policy[state] is not None:
            action_names = ['Up', 'Right', 'Down', 'Left']
            print(f"State {state}: Action = {action_names[policy[state]]}")
    
    # Visualize results
    plot_policy_and_values(
        mdp, 
        values, 
        policy,
        '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_59163d1b365741e093d0768e03e14425/assets/mdp_results.png'
    )
    
    # Simulate some episodes
    print("\nSimulating Episodes:")
    for i in range(3):
        states, reward = simulate_episode(mdp, policy)
        print(f"Episode {i+1}: Visited {len(states)} states, Total reward: {reward}")
        print(f"Path: {' -> '.join(str(s) for s in states)}")
    
    # Save values to CSV
    import csv
    with open('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_59163d1b365741e093d0768e03e14425/assets/mdp_values.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Row', 'Col', 'Value'])
        for state in sorted(mdp.states):
            writer.writerow([state[0], state[1], values[state]])
    
    print("\nResults saved to:")
    print("- /root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_59163d1b365741e093d0768e03e14425/assets/mdp_results.png")
    print("- /root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_59163d1b365741e093d0768e03e14425/assets/mdp_values.csv")

if __name__ == "__main__":
    main()