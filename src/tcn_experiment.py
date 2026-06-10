"""Experiment tracking, hyperparameter search, and benchmarking"""

import json
import time
from pathlib import Path
from dataclasses import asdict
import pandas as pd
import torch

from config import TCNConfig, GRUConfig, DEVICE
from train import train_one 
from datetime import datetime


class ExperimentTracker:
    """Track and save experiment results"""
    
    def __init__(self, search_name="parameter_search"):
        self.search_name = search_name
        self.results = []
        self.exp_dir = Path("experiments")
        self.exp_dir.mkdir(exist_ok=True)
        
        # Results file paths
        self.csv_path = self.exp_dir / f"{search_name}.csv"
        self.json_path = self.exp_dir / f"{search_name}.json"
    
    def add_result(self, config, val_loss, inference_time_ms, param_count):
        """Add a completed experiment result"""
        result = {
            "model_type": config.__class__.__name__,
            "slug": config.slug(),
            "val_loss": val_loss,
            "inference_time_ms": inference_time_ms,
            "realtime_ratio": (config.segment_length * 1000) / (44100 * inference_time_ms), 
            "param_count": param_count,
            "epochs_trained": config.epochs,
            **asdict(config)  
        }
        self.results.append(result)
        self._save()
        
        return result
    
    def _save(self):
        """Save results to CSV and JSON"""
        df = pd.DataFrame(self.results)
        df.to_csv(self.csv_path, index=False)
        
        with open(self.json_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def get_best(self, metric="val_loss"):
        """Get best experiment so far"""
        if not self.results:
            return None
        df = pd.DataFrame(self.results)
        best_idx = df[metric].idxmin()
        return df.iloc[best_idx].to_dict()
    
    def print_summary(self):
        """Print current results summary"""
        if not self.results:
            print("No results yet")
            return
        
        df = pd.DataFrame(self.results)
        print("\n" + "="*80)
        print(f"Experiment Summary: {self.search_name}")
        print("="*80)
        print(df[["model_type", "slug", "param_count", "val_loss", "realtime_ratio"]].to_string())
        print(f"\nBest so far: {self.get_best()['slug']} (val_loss={self.get_best()['val_loss']:.6f})")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_speed(model, device, segment_length=16384, num_runs=500):
    """Measure average inference time for single sample"""
    model.eval()
    
    sample = torch.randn(1, 1, segment_length).to(device)
    
    with torch.no_grad():
        for _ in range(50):
            _ = model(sample)
    
    # Measure
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time_ms = (elapsed / num_runs) * 1000
    
    return avg_time_ms


def run_and_benchmark(config, tracker=None):
    """Run training, then benchmark the model"""

    val_loss, model = train_one(config, return_model=True)
    
    param_count = count_parameters(model)
    inference_time_ms = measure_inference_speed(model, DEVICE, config.segment_length)

    if tracker:
        tracker.add_result(config, val_loss, inference_time_ms, param_count)
    
    return {
        "val_loss": val_loss,
        "param_count": param_count,
        "inference_time_ms": inference_time_ms,
        "realtime_ratio": (config.segment_length * 1000) / (44100 * inference_time_ms)
    }


class SequentialSearcher:
    """Auto hyperparameter search with sequential refinement"""
    
    def __init__(self, tracker):
        self.tracker = tracker
    
    def search_channels(self, channels_list, stacks=2, lr=1e-3, epochs=20):
        """Find best channel size"""
        print("\n" + "="*60)
        print("Phase 1: Channel Search")
        print("="*60)
        
        for channels in channels_list:
            config = TCNConfig(
                channels=channels,
                stacks=stacks,
                lr=lr,
                epochs=epochs
            )
            print(f"\nTrying channels={channels}...")
            run_and_benchmark(config, self.tracker)
        
        best = self.tracker.get_best()
        print(f"\n✓ Best channels: {best['channels']} (val_loss={best['val_loss']:.6f})")
        return best['channels']
    
    def search_stacks(self, stacks_list, channels, lr=1e-3, epochs=20):
        """Find best stack depth"""
        print("\n" + "="*60)
        print("Phase 2: Stack Depth Search")
        print("="*60)
        
        for stacks in stacks_list:
            config = TCNConfig(
                channels=channels,
                stacks=stacks,
                lr=lr,
                epochs=epochs
            )
            print(f"\nTrying stacks={stacks}...")
            run_and_benchmark(config, self.tracker)
        
        best = self.tracker.get_best()
        print(f"\n✓ Best stacks: {best['stacks']} (val_loss={best['val_loss']:.6f})")
        return best['stacks']
    
    def search_lr(self, lr_list, channels, stacks, epochs=20):
        """Find best learning rate"""
        print("\n" + "="*60)
        print("Phase 3: Learning Rate Search")
        print("="*60)
        
        for lr in lr_list:
            config = TCNConfig(
                channels=channels,
                stacks=stacks,
                lr=lr,
                epochs=epochs
            )
            print(f"\nTrying lr={lr:.1e}...")
            run_and_benchmark(config, self.tracker)
        
        best = self.tracker.get_best()
        print(f"\n✓ Best lr: {best['lr']:.1e} (val_loss={best['val_loss']:.6f})")
        return best['lr']


def main_search():
    """Run complete parameter search"""

    date = datetime.now().strftime("%Y-%m-%d")
    tracker = ExperimentTracker(f"tcn_search_{date}")
    searcher = SequentialSearcher(tracker)
    
    # Phase 1: Channel sweep
    best_channels = searcher.search_channels(
        channels_list=[8, 16, 32, 64],
        stacks=2,
        lr=1e-3,
        epochs=20
    )
    
    # Phase 2: Stack depth sweep
    best_stacks = searcher.search_stacks(
        stacks_list=[1, 2, 3, 4],
        channels=best_channels,
        lr=1e-3,
        epochs=20
    )
    
    # Phase 3: Learning rate sweep
    best_lr = searcher.search_lr(
        lr_list=[2e-3, 1e-3, 5e-4, 2e-4, 1e-4],
        channels=best_channels,
        stacks=best_stacks,
        epochs=20
    )
    
    # Phase 4: Final training with best params
    print("\n" + "="*60)
    print("Phase 4: Final Training with Best Parameters")
    print("="*60)
    
    final_config = TCNConfig(
        channels=best_channels,
        stacks=best_stacks,
        lr=best_lr,
        epochs=100 
    )
    
    final_val_loss, final_model = train_one(final_config, return_model=True)
    final_params = count_parameters(final_model)
    final_speed = measure_inference_speed(final_model, DEVICE, final_config.segment_length)
    
    tracker.add_result(final_config, final_val_loss, final_speed, final_params)
    
    # Print final summary
    print("\n" + "="*80)
    print("SEARCH COMPLETE - FINAL SUMMARY")
    print("="*80)
    tracker.print_summary()
    
    print(f"\n Best configuration:")
    best = tracker.get_best()
    for key, value in best.items():
        print(f"  {key}: {value}")
    
    return tracker


if __name__ == "__main__":
    tracker = main_search()
    
    