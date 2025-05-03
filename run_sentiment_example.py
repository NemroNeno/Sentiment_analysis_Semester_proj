import json
import pandas as pd
from sentiment_generation import generate_sentiment_data

def main():
    # Replace with your actual Google API key
    api_key = "AIzaSyC8kprzhE-o8w5x3TWKq8Kyrdant3-V4TQ"
    
    print("Generating sentiment examples using Google's Gemini model...")
    examples = generate_sentiment_data(api_key, num_examples=100)
    
    if not examples:
        print("Failed to generate examples. Please check your API key and try again.")
        return
    
    # Convert to pandas DataFrame for better visualization
    df = pd.DataFrame([example.model_dump() for example in examples])
    
    print("\nGenerated Sentiment Examples:")
    print(df)
    
    # Count examples by polarity
    polarity_counts = df['polarity'].value_counts().to_dict()
    print("\nDistribution by polarity:")
    print(f"Positive (1): {polarity_counts.get(1, 0)}")
    print(f"Neutral (0): {polarity_counts.get(0, 0)}")
    print(f"Negative (-1): {polarity_counts.get(-1, 0)}")
    
    # Save to JSON file
    with open("sentiment_examples.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    
    print("\nExamples saved to sentiment_examples.json")

if __name__ == "__main__":
    main()