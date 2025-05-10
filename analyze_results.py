import torch
import os
from models import SentimentModel
from dataset import prepare_tokenizer, load_data
from utils import visualize_results

if __name__ == "__main__":
    print("Loading data to prepare tokenizer...")
    # We need the data just to initialize the tokenizer
    data = load_data("data\\final_dataset.json")
    
    # Initialize tokenizer with emoji support
    print("Preparing tokenizer...")
    bert_tokenizer, num_emoji_tokens = prepare_tokenizer(data)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find most recent run directory
    run_dirs = sorted([os.path.join("runs", d) for d in os.listdir("runs")], key=os.path.getmtime)
    
    if not run_dirs:
        print("No training runs found in the 'runs' directory.")
    else:
        latest_run = run_dirs[-1]
        print(f"Processing results from latest run: {latest_run}")
        
        # Visualize results from the latest training run
        print("Generating visualizations and reports...")
        visualize_results(latest_run, SentimentModel, bert_tokenizer, device)
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved in: {latest_run}")
        print(f"To view TensorBoard logs, run: tensorboard --logdir={latest_run}")
        print("=========================")
        
        # Get path to best model
        model_dir = os.path.join(latest_run, "checkpoints")
        best_models = [f for f in os.listdir(model_dir) if f.startswith('best_model')]
        if best_models:
            best_model = sorted(best_models, key=lambda x: float(x.split('_f1_')[1].split('.pt')[0]), reverse=True)[0]
            print(f"Best model: {best_model} (F1: {float(best_model.split('_f1_')[1].split('.pt')[0]):.4f})")
