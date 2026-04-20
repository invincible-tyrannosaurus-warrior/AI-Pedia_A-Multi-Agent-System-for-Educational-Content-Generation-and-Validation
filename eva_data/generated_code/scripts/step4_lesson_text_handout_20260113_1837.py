import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define the output directory
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate the Markdown content with proper escaping of apostrophes
markdown_content = """# Topic 2: Introduction to Machine Learning

## What is Machine Learning?

Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. These models enable computers to learn from data and make decisions or predictions without being explicitly programmed for every task.

Let\'s explore some key concepts:

### Key Concepts

1. **Supervised Learning**
   - Uses labeled training data
   - Examples: Classification, Regression

2. **Unsupervised Learning**
   - Finds patterns in unlabeled data
   - Examples: Clustering, Association

3. **Reinforcement Learning**
   - Learns through interaction with environment
   - Uses rewards and penalties

## Applications

Machine learning is used in:
- Image recognition
- Natural language processing
- Fraud detection
- Recommendation systems

## Getting Started

To begin your machine learning journey, consider learning:
- Python programming
- Linear algebra and statistics
- Libraries like scikit-learn and TensorFlow

---

*This lesson was generated programmatically.*
"""

# Write to file
file_path = os.path.join(output_dir, 'topic2_lesson.md')
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(markdown_content)

# Print excerpt and file path
print("Generated Markdown excerpt:")
print(markdown_content[:100] + "...")
print(f"\nFile saved to: {file_path}")

# Create a simple plot to demonstrate the requirements
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title('Sample Plot')
plt.xlabel('X values')
plt.ylabel('Y values')
plot_path = os.path.join(output_dir, 'sample_plot.png')
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to: {plot_path}")