import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Define the output directory
OUTPUT_DIR = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'

# Create the directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_sample_documents():
    """
    Create sample documents for our RAG system.
    These represent chunks of information that could be retrieved.
    """
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn patterns in data.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Reinforcement learning involves training agents to make decisions through rewards.",
        "Supervised learning requires labeled training data to make predictions.",
        "Unsupervised learning finds hidden patterns in unlabeled data.",
        "Data preprocessing is crucial for preparing datasets before machine learning.",
        "Neural networks consist of interconnected nodes called neurons organized in layers.",
        "TensorFlow and PyTorch are popular frameworks for deep learning development."
    ]
    
    # Save documents to file for demonstration purposes
    with open(os.path.join(OUTPUT_DIR, 'sample_documents.txt'), 'w') as f:
        for i, doc in enumerate(documents):
            f.write(f"Document {i+1}: {doc}\n")
    
    return documents

def create_vectorizer(documents):
    """
    Create TF-IDF vectorizer and fit it on documents.
    This converts text into numerical vectors for similarity calculations.
    """
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Fit and transform documents into vectors
    document_vectors = vectorizer.fit_transform(documents)
    
    return vectorizer, document_vectors

def retrieve_relevant_documents(query, vectorizer, document_vectors, documents, top_k=3):
    """
    Retrieve most relevant documents based on query using cosine similarity.
    This simulates the retrieval phase of a RAG system.
    """
    # Transform query into same vector space as documents
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarities between query and all documents
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    
    # Get indices of top-k most similar documents
    top_indices = similarities.argsort()[::-1][:top_k]
    
    # Return relevant documents and their scores
    relevant_docs = [(documents[i], similarities[i]) for i in top_indices if similarities[i] > 0]
    
    return relevant_docs

def generate_response(query, relevant_docs):
    """
    Generate a response based on retrieved documents.
    This simulates the generation phase of a RAG system.
    """
    # Combine relevant documents into context
    context = " ".join([doc[0] for doc in relevant_docs])
    
    # Simple response generation - in practice, this would use a language model
    response = f"Based on the retrieved information about '{query}':\n"
    response += f"Context: {context}\n"
    response += "This is a simulated response generated from the retrieved documents."
    
    return response

def visualize_results(query, relevant_docs, output_path):
    """
    Create visualization showing document relevance scores.
    """
    # Extract document names and scores
    doc_names = [f"Doc {i+1}" for i in range(len(relevant_docs))]
    scores = [score for _, score in relevant_docs]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(doc_names, scores, color='skyblue')
    plt.xlabel('Documents')
    plt.ylabel('Relevance Score')
    plt.title(f'Relevance Scores for Query: "{query}"')
    plt.ylim(0, max(scores) * 1.1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """
    Main function demonstrating the RAG workflow.
    """
    print("Starting RAG System Demo...")
    
    # Step 1: Create sample documents
    print("Step 1: Creating sample documents...")
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents.")
    
    # Step 2: Create vectorizer and document vectors
    print("Step 2: Creating TF-IDF vectors...")
    vectorizer, document_vectors = create_vectorizer(documents)
    
    # Save vectorizer for potential reuse
    with open(os.path.join(OUTPUT_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Step 3: Demonstrate retrieval with sample queries
    queries = [
        "machine learning algorithms",
        "deep learning neural networks", 
        "natural language processing"
    ]
    
    results_summary = []
    
    for i, query in enumerate(queries):
        print(f"\nStep 3.{i+1}: Processing query: '{query}'")
        
        # Retrieve relevant documents
        relevant_docs = retrieve_relevant_documents(query, vectorizer, document_vectors, documents)
        
        # Generate response
        response = generate_response(query, relevant_docs)
        
        # Store results
        results_summary.append({
            'query': query,
            'relevant_docs': relevant_docs,
            'response': response[:100] + "..." if len(response) > 100 else response
        })
        
        # Visualize results
        output_file = os.path.join(OUTPUT_DIR, f'retrieval_results_{i+1}.png')
        visualize_results(query, relevant_docs, output_file)
        print(f"Visualization saved to {output_file}")
        
        # Print results
        print(f"Retrieved {len(relevant_docs)} relevant documents:")
        for j, (doc, score) in enumerate(relevant_docs):
            print(f"  {j+1}. Score: {score:.4f} - {doc}")
        print(f"Response preview: {response[:100]}...")
    
    # Save summary to CSV
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'rag_results_summary.csv'), index=False)
    print(f"\nResults summary saved to {os.path.join(OUTPUT_DIR, 'rag_results_summary.csv')}")
    
    # Print final summary
    print("\n=== RAG SYSTEM DEMO COMPLETE ===")
    print("The system demonstrated:")
    print("1. Document creation and storage")
    print("2. Text vectorization using TF-IDF")
    print("3. Document retrieval based on query similarity")
    print("4. Response generation from retrieved context")
    print("5. Visualization of retrieval results")
    print("6. Results export to CSV")

if __name__ == "__main__":
    main()