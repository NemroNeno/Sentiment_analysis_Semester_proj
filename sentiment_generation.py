import json
import google.generativeai as genai
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
import time  # For rate limiting
import concurrent.futures  # For multithreading support
import threading  # For thread-safe operations


# Define your output schema for sentiment examples
class SentimentExample(BaseModel):
    sentence: str = Field(..., description="A natural language sentence")
    polarity: Literal[-1, 0, 1] = Field(..., description="Sentiment polarity: -1 (negative), 0 (neutral), or 1 (positive)")

# Thread-safe counter for progress tracking
class AtomicCounter:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = threading.Lock()
        
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
            
    def value(self):
        with self._lock:
            return self._value

# Function to generate sentiment data using Gemini
def generate_sentiment_data(api_key: str, num_examples: int = 10):
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # List available models and select appropriate one
    try:
        # Try to list available models to see what's available
        models = genai.list_models()
        available_models = [model.name for model in models]
        
        # Prioritized models to try (in order of preference)
        preferred_models = [
            'models/gemini-2.0-flash-lite',
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
    
    # Limit batch size to a reasonable number that the model can handle
    effective_num_examples = min(num_examples, 100)  # Most models struggle with generating >100 structured items
    
    # Create the prompt
    prompt = f"""Generate exactly {effective_num_examples} different sarcastic sentences with their sentiment polarity.
    Include sarcasm either through tone, emoji usage (especially where the emoji contradicts the tone, as commonly used by Gen Z), or context.
    Optionally include emojis that enhance the sarcastic tone (e.g., using "😊" in a clearly unpleasant context).
    The sentences should be in English and should be of high quality so that they will be used as data for training a sentiment analysis model.
    Keep your sentences a bit long.Try to make more sentences that are positive and neutral than negative.
Each sentence should have a clear sentiment that can be classified as:
- Positive (1): Sentences expressing happiness, satisfaction, or positive opinions
- Neutral (0): Sentences stating facts or with no clear sentiment


Return your response in JSON format as a list of objects, each with 'sentence' and 'polarity' keys.
Example:
[
  {{"sentence": "I love this new phone! 🤞", "polarity": -1}},
  {{"sentence": "The sky is blue.", "polarity": 0}},
  {{"sentence": "This service was fun.'😒", "polarity": -1}}
]

Make sure the JSON response includes exactly {effective_num_examples} examples with a mix of positive, negative, and neutral sentences.
Only return the JSON array, no other text.
"""
    
    # Fixed prompt with proper escaping for the JSON examples
    prompt_pos_sentence = """Generate exactly {} different high-quality English sentences with either positive or neutral sentiment.

Ensure that:
- All sentences are genuinely positive or neutral — do not include any sarcastic, ambiguous, or negative sentences.
- Positive (1): Sentences should express happiness, appreciation, enthusiasm, or positive opinions.
- Neutral (0): Sentences should state facts, ask neutral questions, or describe things without emotional tone.
- Avoid any sarcasm, irony, passive-aggressive tone, or misleading use of emojis.
- Keep sentences moderately long to provide meaningful training data.
- Use emojis only if they enhance the clarity of genuine positivity or neutrality (optional).
- Do not include any negative sentiment or sentiment that could be misinterpreted as negative.

Return your response in JSON format as a list of objects, each with 'sentence' and 'polarity' keys.

Example:
[
  {{"sentence": "The package arrived on time and everything was intact. 📦", "polarity": 1}},
  {{"sentence": "Water boils at 100 degrees Celsius.", "polarity": 0}}
]

Make sure the JSON response includes exactly {} examples.
Only return the JSON array, no other text.
""".format(effective_num_examples, effective_num_examples)

    # Generate the response
    try:
        response = model.generate_content(prompt_pos_sentence)
        
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
                
                print(f"Successfully generated {len(examples)} examples out of {effective_num_examples} requested.")
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

# Worker function for thread pool
def batch_worker(args):
    api_key, batch_num, total_batches, examples_per_batch, counter = args
    try:
        print(f"Starting batch {batch_num+1}/{total_batches}...")
        examples = generate_sentiment_data(api_key, examples_per_batch)
        
        if examples:
            count = counter.increment()
            print(f"✓ Batch {batch_num+1}/{total_batches} complete: {len(examples)} examples. Completed batches: {count}/{total_batches}")
            return examples
        else:
            # Retry with smaller batch size if failed
            print(f"Batch {batch_num+1} failed. Retrying with smaller batch...")
            retry_examples = generate_sentiment_data(api_key, max(10, examples_per_batch // 2))
            if retry_examples:
                count = counter.increment()
                print(f"✓ Batch {batch_num+1}/{total_batches} retry successful: {len(retry_examples)} examples. Completed batches: {count}/{total_batches}")
                return retry_examples
            else:
                print(f"× Batch {batch_num+1} retry failed.")
                return []
    except Exception as e:
        print(f"× Error in batch {batch_num+1}: {e}")
        return []

# Function to generate and save sentiment data in batches
def create_sentiment_dataset(api_key: str, examples_per_batch: int = 50, num_batches: int = 20, output_file: str = "sentiment_examples.json"):
    all_examples = []
    
    print(f"Generating {num_batches} batches with {examples_per_batch} examples per batch...")
    print(f"Target total: {examples_per_batch * num_batches} examples")
    
    for i in range(num_batches):
        print(f"Generating batch {i+1}/{num_batches}...")
        examples = generate_sentiment_data(api_key, examples_per_batch)
        
        if examples:
            all_examples.extend(examples)
            print(f"Batch {i+1} complete: {len(examples)} examples generated. Running total: {len(all_examples)}")
            
            # Add a short delay to avoid rate limiting issues
            if i < num_batches - 1:
                time.sleep(1)
        else:
            print(f"Batch {i+1} failed to generate any examples. Retrying...")
            # Retry with smaller batch size if failed
            retry_examples = generate_sentiment_data(api_key, max(10, examples_per_batch // 2))
            if retry_examples:
                all_examples.extend(retry_examples)
                print(f"Retry successful: {len(retry_examples)} examples generated. Running total: {len(all_examples)}")
            else:
                print(f"Retry failed. Continuing to next batch.")
    
    # Convert to list of dictionaries
    output_data = [example.model_dump() for example in all_examples]
    
    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Generated a total of {len(output_data)} examples with sentiment polarity.")
    print(f"Dataset saved to {output_file}")
    
    return all_examples

# Multithreaded version of create_sentiment_dataset
def create_sentiment_dataset_parallel(api_key: str, examples_per_batch: int = 50, num_batches: int = 20, 
                                      max_workers: int = 10, output_file: str = "sentiment_examples.json"):
    """
    Generate sentiment data in parallel using multiple threads.
    
    Args:
        api_key: Google API key
        examples_per_batch: Number of examples to generate in each batch
        num_batches: Total number of batches to generate
        max_workers: Maximum number of parallel threads to use
        output_file: Output file path to save the dataset
    """
    # Use a thread-safe counter to track progress
    counter = AtomicCounter()
    all_examples = []
    
    # Limit max_workers to not exceed num_batches or system capability
    effective_max_workers = min(max_workers, num_batches, 
                              (concurrent.futures.ThreadPoolExecutor()._max_workers or 32))
    
    print(f"Generating {num_batches} batches with {examples_per_batch} examples per batch...")
    print(f"Target total: {examples_per_batch * num_batches} examples")
    print(f"Using {effective_max_workers} parallel threads")
    
    # Create thread arguments
    batch_args = [(api_key, i, num_batches, examples_per_batch, counter) 
                 for i in range(num_batches)]
    
    # Run batches using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
        results = list(executor.map(batch_worker, batch_args))
    
    # Flatten results (list of lists into single list)
    for batch_examples in results:
        if batch_examples:
            all_examples.extend(batch_examples)
    
    # Convert to list of dictionaries
    output_data = [example.model_dump() for example in all_examples]
    
    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nGeneration complete!")
    print(f"Generated a total of {len(output_data)} examples with sentiment polarity.")
    print(f"Dataset saved to {output_file}")
    
    return all_examples

if __name__ == "__main__":
    # Replace with your Google API key
    GOOGLE_API_KEY = "your-google-api-key-here"
    
    # Create the dataset using the multithreaded version by default
    create_sentiment_dataset_parallel(GOOGLE_API_KEY, examples_per_batch=50, num_batches=20)