import os
import shutil
import datetime

def backup_model_runs():
    """Backup existing model runs before training new models"""
    # Check if runs directory exists
    if not os.path.exists("runs"):
        print("No runs directory found. No backup needed.")
        return
    
    # Create backup directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = os.path.join("model_backups", f"backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Copy all runs to backup directory
    run_dirs = [d for d in os.listdir("runs") if os.path.isdir(os.path.join("runs", d))]
    
    if not run_dirs:
        print("No model runs found. No backup needed.")
        return
    
    print(f"Backing up {len(run_dirs)} model runs to {backup_dir}...")
    
    for run_dir in run_dirs:
        src_path = os.path.join("runs", run_dir)
        dst_path = os.path.join(backup_dir, run_dir)
        shutil.copytree(src_path, dst_path)
    
    print(f"Backup completed. {len(run_dirs)} model runs saved to {backup_dir}")

if __name__ == "__main__":
    backup_model_runs()
