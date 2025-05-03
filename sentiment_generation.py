import json
import google.generativeai as genai
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field

# Define your output schema for sentiment examples
class SentimentExample(BaseModel):
    sentence: str = Field(..., description="A natural language sentence")
    polarity: Literal[-1, 0, 1] = Field(..., description="Sentiment polarity: -1 (negative), 0 (neutral), or 1 (positive)")

# Function to generate sentiment data using Gemini
def generate_sentiment_data(api_key: str, num_examples: int = 10):
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # List available models and select appropriate one
    try:
        # Try to list available models to see what's available
        models = genai.list_models()
        available_models = [model.name for model in models]
        print("Available models:", available_models)
        
        # Prioritized models to try (in order of preference)
        preferred_models = [
            'models/gemini-1.5-pro',
            'models/gemini-1.5-flash',
            'models/gemini-2.0-pro',
            'models/gemini-2.0-flash',
            'models/gemini-2.0-flash-001',
            'models/gemini-1.5-pro-latest',
            'models/gemini-1.5-flash-latest',
            'models/text-bison-001'  # Fallback to text-only model if no Gemini is available
        ]
        
        # Find the highest priority available model
        model_name = None
        for preferred in preferred_models:
            if preferred in available_models:
                model_name = preferred
                break
                
        # If none of our preferred models are available, try to find any text-capable Gemini model
        if not model_name:
            # Filter out vision models and deprecated models
            text_models = [name for name in available_models 
                          if 'vision' not in name.lower() 
                          and 'embedding' not in name.lower()
                          and 'gemini' in name.lower()]
            
            if text_models:
                model_name = text_models[0]
            else:
                # Last resort - try the first available model
                model_name = available_models[0]
                
        print(f"Selected model: {model_name}")
    except Exception as e:
        print(f"Error listing models: {e}")
        # Just try using a default model name as last resort
        model_name = 'models/gemini-1.5-flash'
        print(f"Using default model name: {model_name}")
    
    # Set up the model
    model = genai.GenerativeModel(model_name)
    
    # Create the prompt
    prompt = f"""Generate {num_examples} different highly sarcastics sentences that may contain emoji that causes that sarcasm sentences with their sentiment polarity.
    Include sarcasm either through tone, emoji usage (especially where the emoji contradicts the tone, as commonly used by Gen Z), or context.
    Optionally include emojis that enhance the sarcastic tone (e.g., using "😊" in a clearly unpleasant context).
Each sentence should have a clear sentiment that can be classified as:
- Positive (1): Sentences expressing happiness, satisfaction, or positive opinions
- Neutral (0): Sentences stating facts or with no clear sentiment
- Negative (-1): Sentences expressing dissatisfaction, sadness, or negative opinions

Return your response in JSON format as a list of objects, each with 'sentence' and 'polarity' keys.
Example:
[
  {{"sentence": "I love this new phone! 🤞", "polarity": -1}},
  {{"sentence": "The sky is blue.", "polarity": 0}},
  {{"sentence": "This service was fun.'😒", "polarity": -1}}
]

Make sure the JSON response includes {num_examples} examples with a mix of positive, negative, and neutral sentences.
"""
    
    # Generate the response
    try:
        response = model.generate_content(prompt)
        
        # Extract the JSON response
        try:
            # Try to extract JSON from the response text
            response_text = response.text
            # Find the JSON part (in case there's additional text)
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                sentiment_data = json.loads(json_str)
                
                # Validate the data structure
                examples = []
                for item in sentiment_data:
                    if isinstance(item, dict) and 'sentence' in item and 'polarity' in item:
                        # Ensure polarity is -1, 0, or 1
                        polarity = item['polarity']
                        if polarity in [-1, 0, 1]:
                            examples.append(SentimentExample(
                                sentence=item['sentence'],
                                polarity=polarity
                            ))
                
                return examples
            else:
                print("Could not find valid JSON in response.")
                print(f"Raw response: {response.text}")
                return []
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response.text}")
            return []
    except Exception as e:
        print(f"Error generating content: {e}")
        return []

# Function to generate and save sentiment data in batches
def create_sentiment_dataset(api_key: str, examples_per_batch: int = 10, num_batches: int = 10):
    all_examples = []
    
    print(f"Generating {num_batches} batches with {examples_per_batch} examples each...")
    
    for i in range(num_batches):
        print(f"Generating batch {i+1}/{num_batches}...")
        examples = generate_sentiment_data(api_key, examples_per_batch)
        all_examples.extend(examples)
        print(f"Batch {i+1} complete: {len(examples)} examples generated.")
    
    # Convert to list of dictionaries
    output_data = [example.model_dump() for example in all_examples]
    
    # Save to JSON file
    with open("sentiment_dataset.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Generated a total of {len(output_data)} examples with sentiment polarity.")
    print(f"Dataset saved to sentiment_dataset.json")
    
    return all_examples

if __name__ == "__main__":
    # Replace with your Google API key
    GOOGLE_API_KEY = "your-google-api-key-here"
    
    # Create the dataset
    create_sentiment_dataset(GOOGLE_API_KEY, examples_per_batch=10, num_batches=5)