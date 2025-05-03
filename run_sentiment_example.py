import json
import pandas as pd
from sentiment_generation import create_sentiment_dataset, generate_sentiment_data

def main():
    # Replace with your actual Google API key
    api_key = "AIzaSyBQuwy8GobSUqQ6pFwtFstbhg_dlgEWN_0"
    
    # How many examples you want to generate in total
    total_examples = 10000
    
    print(f"Generating {total_examples} sentiment examples using Google's Gemini model...")
    
    # Using batching approach to generate large number of examples
    # Each batch requests 50 examples, and we'll make enough batches to reach desired total
    examples_per_batch = 50
    num_batches = total_examples // examples_per_batch
    if total_examples % examples_per_batch > 0:
        num_batches += 1  # Add one more batch for any remainder
        
    examples = create_sentiment_dataset(api_key, 
                                       examples_per_batch=examples_per_batch,
                                       num_batches=num_batches)
    
    if not examples or len(examples) == 0:
        print("Failed to generate examples. Please check your API key and try again.")
        return
    
    # Convert to pandas DataFrame for better visualization
    df = pd.DataFrame([example.model_dump() for example in examples])
    
    print(f"\nGenerated {len(df)} Sentiment Examples:")
    print(df.head())  # Show just first few examples
    
    # Count examples by polarity
    polarity_counts = df['polarity'].value_counts().to_dict()
    print("\nDistribution by polarity:")
    print(f"Positive (1): {polarity_counts.get(1, 0)}")
    print(f"Neutral (0): {polarity_counts.get(0, 0)}")
    print(f"Negative (-1): {polarity_counts.get(-1, 0)}")
    
    # Save to JSON file
    with open("sentiment_examples.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    
    print(f"\n{len(df)} examples saved to sentiment_examples.json")

if __name__ == "__main__":
    main()