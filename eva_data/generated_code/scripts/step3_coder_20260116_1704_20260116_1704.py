import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Ensure the assets directory exists
assets_dir = 'D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets'
os.makedirs(assets_dir, exist_ok=True)

def generate_chapter_3_lesson():
    """
    Complete lesson covering Chapter 3 concepts including:
    - Data preprocessing
    - Feature engineering
    - Model training and evaluation
    - Visualization of results
    """
    
    # Section 1: Introduction to Chapter 3
    print("=== CHAPTER 3 LESSON: MODEL EVALUATION AND VALIDATION ===")
    print("This lesson covers key concepts in model evaluation and validation.")
    print()
    
    # Section 2: Data Generation and Preprocessing
    print("Section 2: Data Generation and Preprocessing")
    
    # Generate synthetic dataset for demonstration
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create DataFrame for better visualization
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    print()
    
    # Section 3: Train-Test Split
    print("Section 3: Train-Test Split")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    print()
    
    # Section 4: Model Training
    print("Section 4: Model Training")
    
    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    print("Model trained successfully!")
    print()
    
    # Section 5: Model Evaluation Metrics
    print("Section 5: Model Evaluation Metrics")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print()
    
    # Section 6: Visualization of Results
    print("Section 6: Visualization of Results")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Feature importance plot
    feature_importance = abs(model.coef_[0])
    feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(feature_importance)[::-1]
    plt.bar(range(len(feature_importance)), feature_importance[indices])
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'feature_importance.png'))
    plt.close()
    
    # Distribution of predictions
    plt.figure(figsize=(8, 6))
    plt.hist([y_test, y_pred], bins=20, alpha=0.7, label=['True Labels', 'Predictions'])
    plt.title('Distribution of True vs Predicted Labels')
    plt.xlabel('Label Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'prediction_distribution.png'))
    plt.close()
    
    # Section 7: Cross-validation Example
    print("Section 7: Cross-Validation Example")
    
    from sklearn.model_selection import cross_val_score
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print()
    
    # Section 8: Overfitting and Underfitting Analysis
    print("Section 8: Overfitting and Underfitting Analysis")
    
    # Test different model complexities
    complexities = [1, 10, 100, 1000]
    train_scores = []
    test_scores = []
    
    for c in complexities:
        # Create a more complex model
        model_complex = LogisticRegression(C=c, random_state=42)
        model_complex.fit(X_train, y_train)
        
        train_score = model_complex.score(X_train, y_train)
        test_score = model_complex.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # Plot complexity analysis
    plt.figure(figsize=(10, 6))
    plt.plot(complexities, train_scores, 'o-', label='Training Score')
    plt.plot(complexities, test_scores, 'o-', label='Testing Score')
    plt.xscale('log')
    plt.xlabel('Regularization Strength (C)')
    plt.ylabel('Accuracy')
    plt.title('Model Complexity Analysis')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'complexity_analysis.png'))
    plt.close()
    
    # Section 9: Summary and Key Takeaways
    print("Section 9: Summary and Key Takeaways")
    
    print("Key Concepts Covered:")
    print("1. Data preprocessing and splitting")
    print("2. Model training with scikit-learn")
    print("3. Evaluation metrics (accuracy, precision, recall)")
    print("4. Confusion matrix interpretation")
    print("5. Feature importance analysis")
    print("6. Cross-validation techniques")
    print("7. Overfitting and underfitting detection")
    print()
    
    print("Best Practices:")
    print("- Always split your data before model training")
    print("- Use multiple evaluation metrics, not just accuracy")
    print("- Visualize results to understand model performance")
    print("- Apply cross-validation for robust model assessment")
    print("- Monitor for overfitting and underfitting")
    print()
    
    # Save dataset for future reference
    df.to_csv(os.path.join(assets_dir, 'chapter3_dataset.csv'), index=False)
    
    print("Lesson completed successfully!")
    print(f"All visualizations saved to: {assets_dir}")

# Run the lesson
if __name__ == "__main__":
    generate_chapter_3_lesson()