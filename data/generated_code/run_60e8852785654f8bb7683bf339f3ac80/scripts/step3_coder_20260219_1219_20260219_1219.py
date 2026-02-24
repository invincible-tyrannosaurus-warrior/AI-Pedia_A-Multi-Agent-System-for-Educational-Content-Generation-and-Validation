import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import time
import os

# Ensure the assets directory exists
assets_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/run_60e8852785654f8bb7683bf339f3ac80/assets'
os.makedirs(assets_dir, exist_ok=True)

def bubble_sort(arr):
    """
    Bubble Sort Algorithm
    Repeatedly steps through the list, compares adjacent elements and swaps them if they're in the wrong order.
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    n = len(arr)
    # Make a copy to avoid modifying the original array
    arr_copy = arr.copy()
    
    # Traverse through all array elements
    for i in range(n):
        # Flag to optimize - if no swapping occurs, array is sorted
        swapped = False
        
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Swap if the element found is greater than the next element
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                swapped = True
        
        # If no swapping occurred, array is sorted
        if not swapped:
            break
    
    return arr_copy

def selection_sort(arr):
    """
    Selection Sort Algorithm
    Finds the minimum element from the unsorted part and puts it at the beginning.
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    # Traverse through all array elements
    for i in range(n):
        # Find the minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr_copy[min_idx] > arr_copy[j]:
                min_idx = j
        
        # Swap the found minimum element with the first element
        arr_copy[i], arr_copy[min_idx] = arr_copy[min_idx], arr_copy[i]
    
    return arr_copy

def insertion_sort(arr):
    """
    Insertion Sort Algorithm
    Builds the final sorted array one item at a time by repeatedly taking the next element 
    and inserting it into its correct position.
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    arr_copy = arr.copy()
    
    # Traverse from 1 to len(arr)
    for i in range(1, len(arr_copy)):
        key = arr_copy[i]
        j = i - 1
        
        # Move elements greater than key one position ahead
        while j >= 0 and arr_copy[j] > key:
            arr_copy[j + 1] = arr_copy[j]
            j -= 1
        
        # Insert the key at its correct position
        arr_copy[j + 1] = key
    
    return arr_copy

def merge_sort(arr):
    """
    Merge Sort Algorithm
    Divides the array into halves, recursively sorts them, then merges the sorted halves.
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    # Divide the array into two halves
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Merge the sorted halves
    return merge(left, right)

def merge(left, right):
    """Helper function for merge sort to merge two sorted arrays."""
    result = []
    i = j = 0
    
    # Compare elements from both arrays and merge in sorted order
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

def quick_sort(arr):
    """
    Quick Sort Algorithm
    Picks an element as pivot and partitions the array around the pivot.
    Time Complexity: O(n log n) average, O(n^2) worst case
    Space Complexity: O(log n)
    """
    arr_copy = arr.copy()
    
    def quick_sort_helper(low, high):
        if low < high:
            # Partition the array and get pivot index
            pi = partition(low, high)
            
            # Recursively sort elements before and after partition
            quick_sort_helper(low, pi - 1)
            quick_sort_helper(pi + 1, high)
    
    def partition(low, high):
        # Choose the rightmost element as pivot
        pivot = arr_copy[high]
        
        # Index of smaller element (indicates right position of pivot)
        i = low - 1
        
        for j in range(low, high):
            # If current element is smaller than or equal to pivot
            if arr_copy[j] <= pivot:
                i += 1
                arr_copy[i], arr_copy[j] = arr_copy[j], arr_copy[i]
        
        # Place pivot in its correct position
        arr_copy[i + 1], arr_copy[high] = arr_copy[high], arr_copy[i + 1]
        return i + 1
    
    quick_sort_helper(0, len(arr_copy) - 1)
    return arr_copy

def measure_time(sort_func, arr):
    """
    Measure the execution time of a sorting algorithm on given array.
    Returns the sorted array and the time taken.
    """
    start_time = time.time()
    sorted_arr = sort_func(arr)
    end_time = time.time()
    execution_time = end_time - start_time
    return sorted_arr, execution_time

def generate_test_arrays(size):
    """
    Generate test arrays of specified size with different patterns.
    """
    # Random array
    random_array = [random.randint(1, 1000) for _ in range(size)]
    
    # Sorted array
    sorted_array = list(range(1, size + 1))
    
    # Reverse sorted array
    reverse_sorted_array = list(range(size, 0, -1))
    
    # Nearly sorted array (swap some elements)
    nearly_sorted = list(range(1, size + 1))
    # Swap a few elements
    for _ in range(size // 10):
        i, j = random.randint(0, size-1), random.randint(0, size-1)
        nearly_sorted[i], nearly_sorted[j] = nearly_sorted[j], nearly_sorted[i]
    
    return random_array, sorted_array, reverse_sorted_array, nearly_sorted

def plot_performance_comparison():
    """
    Plot performance comparison of different sorting algorithms.
    """
    # Test sizes
    sizes = [100, 500, 1000, 2000]
    
    # Algorithms to test
    algorithms = [
        ("Bubble Sort", bubble_sort),
        ("Selection Sort", selection_sort),
        ("Insertion Sort", insertion_sort),
        ("Merge Sort", merge_sort),
        ("Quick Sort", quick_sort)
    ]
    
    # Store results
    results = {name: [] for name, _ in algorithms}
    
    # Test with random arrays
    print("Testing sorting algorithms with random arrays...")
    for size in sizes:
        print(f"Testing with array size: {size}")
        random_array, _, _, _ = generate_test_arrays(size)
        
        for name, func in algorithms:
            _, exec_time = measure_time(func, random_array)
            results[name].append(exec_time)
            print(f"  {name}: {exec_time:.6f} seconds")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each algorithm's performance
    for name, times in results.items():
        plt.plot(sizes, times, marker='o', label=name, linewidth=2)
    
    # Customize the plot
    plt.xlabel('Array Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison of Sorting Algorithms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(assets_dir, 'sorting_algorithms_performance.png'))
    plt.close()

def demonstrate_sorting_process():
    """
    Demonstrate how sorting algorithms work on a small example.
    """
    # Small example array
    example_array = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", example_array)
    
    # Show each sorting algorithm in action
    algorithms = [
        ("Bubble Sort", bubble_sort),
        ("Selection Sort", selection_sort),
        ("Insertion Sort", insertion_sort),
        ("Merge Sort", merge_sort),
        ("Quick Sort", quick_sort)
    ]
    
    for name, func in algorithms:
        sorted_array = func(example_array)
        print(f"{name} result: {sorted_array}")
    
    # Create a visualization of one algorithm's process
    visualize_bubble_sort_process(example_array)

def visualize_bubble_sort_process(arr):
    """
    Visualize the bubble sort process step by step.
    """
    # Create a simple visualization of bubble sort
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    # We'll create a simple bar chart showing the state at each step
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot initial state
    bars = ax.bar(range(len(arr_copy)), arr_copy, color='blue')
    ax.set_title('Bubble Sort Visualization')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    
    # Save initial state
    plt.savefig(os.path.join(assets_dir, 'bubble_sort_initial.png'))
    plt.close()
    
    # Simulate sorting steps
    steps = []
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                steps.append(arr_copy.copy())
    
    # Create a plot showing the final sorted array
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(arr_copy)), arr_copy, color='green')
    ax.set_title('Bubble Sort Final Result')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    
    # Save final state
    plt.savefig(os.path.join(assets_dir, 'bubble_sort_final.png'))
    plt.close()

def main():
    """
    Main function to run all demonstrations and comparisons.
    """
    print("Sorting Algorithms Demonstration")
    print("=" * 40)
    
    # Demonstrate sorting on small example
    demonstrate_sorting_process()
    
    # Performance comparison
    plot_performance_comparison()
    
    print("\nDemonstrations completed!")
    print("Results saved to:", assets_dir)

if __name__ == "__main__":
    main()