import json
import pandas as pd
import time
from sentiment_generation import create_sentiment_dataset, create_sentiment_dataset_parallel, generate_sentiment_data

def main():
    # Replace with your actual Google API key
    api_key = "AIzaSyBSnZKQWZfY_sdMU1rJ6VMbItnfk5DCpho"
    
    # How many examples you want to generate in total
    total_examples = 20000
    
    # Using batching approach to generate large number of examples
    examples_per_batch = 50
    num_batches = total_examples // examples_per_batch
    if total_examples % examples_per_batch > 0:
        num_batches += 1  # Add one more batch for any remainder
    
    # Number of parallel threads to use (adjust based on your system capability)
    max_workers = 8  # You can adjust this based on CPU cores and memory
    
    print(f"Generating {total_examples} sentiment examples using Google's Gemini model...")
    print(f"Using multithreading with {max_workers} worker threads for faster generation.")
    
    # Record start time to measure performance improvement
    start_time = time.time()
    
    # Use the multithreaded version instead of sequential generation
    examples = create_sentiment_dataset_parallel(api_key, 
                                              examples_per_batch=examples_per_batch,
                                              num_batches=num_batches,
                                              max_workers=max_workers,
                                              output_file="sentiment_examples5.json")
    
    # Calculate time taken
    elapsed_time = time.time() - start_time
    
    if not examples or len(examples) == 0:
        print("Failed to generate examples. Please check your API key and try again.")
        return
    
    # Convert to pandas DataFrame for better visualization
    df = pd.DataFrame([example.model_dump() for example in examples])
    
    print(f"\nGenerated {len(df)} Sentiment Examples in {elapsed_time:.1f} seconds")
    print(f"Average time per batch: {elapsed_time/num_batches:.2f} seconds")
    print(df.head())  # Show just first few examples
    
    # Count examples by polarity
    polarity_counts = df['polarity'].value_counts().to_dict()
    print("\nDistribution by polarity:")
    print(f"Positive (1): {polarity_counts.get(1, 0)}")
    print(f"Neutral (0): {polarity_counts.get(0, 0)}")
    print(f"Negative (-1): {polarity_counts.get(-1, 0)}")
    
    print(f"\nAll {len(df)} examples saved to sentiment_examples5.json")

if __name__ == "__main__":
    main()