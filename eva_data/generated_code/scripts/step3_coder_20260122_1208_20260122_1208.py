import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os

# Ensure the assets directory exists
assets_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'
os.makedirs(assets_dir, exist_ok=True)

def generate_valo_data():
    """Generate synthetic Valorant player performance data"""
    # Define agents and maps
    agents = ['Jett', 'Reyna', 'Phoenix', 'Breach', 'Omen', 'Sage', 'Cypher', 'Killjoy']
    maps = ['Haven', 'Split', 'Ascent', 'Bind', 'Icebox', 'Fracture', 'Breeze']
    
    # Generate random data for 100 players
    np.random.seed(42)
    data = []
    
    for i in range(100):
        player_data = {
            'player_id': i + 1,
            'agent': np.random.choice(agents),
            'map': np.random.choice(maps),
            'kills': np.random.randint(0, 30),
            'deaths': np.random.randint(0, 30),
            'assists': np.random.randint(0, 20),
            'rounds_played': np.random.randint(5, 50),
            'win_rate': np.random.uniform(0.2, 0.9),
            'accuracy': np.random.uniform(0.3, 0.95),
            'damage_per_round': np.random.randint(100, 500),
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        data.append(player_data)
    
    return pd.DataFrame(data)

def analyze_player_performance(df):
    """Analyze and visualize player performance metrics"""
    
    # Calculate K/D ratio
    df['kd_ratio'] = df['kills'] / (df['deaths'] + 1)  # Add 1 to avoid division by zero
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Valorant Player Performance Analysis', fontsize=16)
    
    # 1. Kills vs Deaths scatter plot
    scatter = axes[0, 0].scatter(df['kills'], df['deaths'], c=df['kd_ratio'], cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel('Kills')
    axes[0, 0].set_ylabel('Deaths')
    axes[0, 0].set_title('Kills vs Deaths')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # 2. Agent popularity bar chart
    agent_counts = df['agent'].value_counts()
    bars = axes[0, 1].bar(range(len(agent_counts)), agent_counts.values, color='skyblue')
    axes[0, 1].set_xlabel('Agents')
    axes[0, 1].set_ylabel('Number of Players')
    axes[0, 1].set_title('Agent Popularity')
    axes[0, 1].set_xticks(range(len(agent_counts)))
    axes[0, 1].set_xticklabels(agent_counts.index, rotation=45)
    
    # 3. Win rate distribution histogram
    axes[1, 0].hist(df['win_rate'], bins=15, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Win Rate')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Win Rates')
    
    # 4. Damage per round vs Accuracy scatter plot
    scatter2 = axes[1, 1].scatter(df['damage_per_round'], df['accuracy'], 
                                 c=df['kills'], cmap='plasma', alpha=0.7)
    axes[1, 1].set_xlabel('Damage per Round')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Damage vs Accuracy (Colored by Kills)')
    plt.colorbar(scatter2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f'{assets_dir}/valo_performance_analysis.png')
    plt.close()

def analyze_map_performance(df):
    """Analyze performance on different maps"""
    
    # Group by map and calculate average stats
    map_stats = df.groupby('map').agg({
        'win_rate': 'mean',
        'kills': 'mean',
        'damage_per_round': 'mean'
    }).reset_index()
    
    # Create map performance visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Map Performance Analysis', fontsize=16)
    
    # Win rate by map
    bars1 = axes[0].bar(range(len(map_stats)), map_stats['win_rate'], color='orange')
    axes[0].set_xlabel('Maps')
    axes[0].set_ylabel('Average Win Rate')
    axes[0].set_title('Win Rate by Map')
    axes[0].set_xticks(range(len(map_stats)))
    axes[0].set_xticklabels(map_stats['map'], rotation=45)
    
    # Kills by map
    bars2 = axes[1].bar(range(len(map_stats)), map_stats['kills'], color='red')
    axes[1].set_xlabel('Maps')
    axes[1].set_ylabel('Average Kills')
    axes[1].set_title('Average Kills by Map')
    axes[1].set_xticks(range(len(map_stats)))
    axes[1].set_xticklabels(map_stats['map'], rotation=45)
    
    # Damage per round by map
    bars3 = axes[2].bar(range(len(map_stats)), map_stats['damage_per_round'], color='purple')
    axes[2].set_xlabel('Maps')
    axes[2].set_ylabel('Average Damage per Round')
    axes[2].set_title('Average Damage by Map')
    axes[2].set_xticks(range(len(map_stats)))
    axes[2].set_xticklabels(map_stats['map'], rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{assets_dir}/valo_map_performance.png')
    plt.close()

def create_correlation_heatmap(df):
    """Create correlation heatmap of numerical variables"""
    
    # Select numerical columns for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Valorant Performance Metrics')
    plt.tight_layout()
    plt.savefig(f'{assets_dir}/valo_correlation_heatmap.png')
    plt.close()

def generate_summary_report(df):
    """Generate summary statistics report"""
    
    # Calculate key statistics
    summary_stats = {
        'Total Players': len(df),
        'Average Kills': df['kills'].mean(),
        'Average Deaths': df['deaths'].mean(),
        'Average Win Rate': df['win_rate'].mean(),
        'Average Accuracy': df['accuracy'].mean(),
        'Average Damage per Round': df['damage_per_round'].mean(),
        'Best Agent': df['agent'].value_counts().index[0],
        'Most Played Map': df['map'].value_counts().index[0]
    }
    
    # Save summary to CSV
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(f'{assets_dir}/valo_summary_statistics.csv', index=False)
    
    return summary_stats

def main():
    """Main function to run the Valorant analysis"""
    
    print("Generating Valorant player performance data...")
    # Generate synthetic data
    valo_data = generate_valo_data()
    
    # Save raw data to CSV
    valo_data.to_csv(f'{assets_dir}/valo_raw_data.csv', index=False)
    
    print("Analyzing player performance...")
    # Analyze player performance
    analyze_player_performance(valo_data)
    
    print("Analyzing map performance...")
    # Analyze map performance
    analyze_map_performance(valo_data)
    
    print("Creating correlation heatmap...")
    # Create correlation heatmap
    create_correlation_heatmap(valo_data)
    
    print("Generating summary report...")
    # Generate summary report
    summary = generate_summary_report(valo_data)
    
    # Print summary to console
    print("\n=== Valorant Performance Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nAll visualizations saved to {assets_dir}")

if __name__ == "__main__":
    main()