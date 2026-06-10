# gru_experiment.py
"""GRU hyperparameter search"""

from tcn_experiment import ExperimentTracker, count_parameters, measure_inference_speed, run_and_benchmark
from config import GRUConfig, DEVICE
from datetime import datetime


class GRUSequentialSearcher:
    """Auto hyperparameter search for GRU"""
    
    def __init__(self, tracker):
        self.tracker = tracker
    
    def search_hidden_size(self, hidden_list, num_layers=2, lr=1e-3, epochs=20):
        """Find best hidden size"""
        print("\n" + "="*60)
        print("Phase 1: Hidden Size Search")
        print("="*60)
        
        for hidden_size in hidden_list:
            config = GRUConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                lr=lr,
                epochs=epochs
            )
            print(f"\nTrying hidden_size={hidden_size}...")
            run_and_benchmark(config, self.tracker)
        
        best = self.tracker.get_best()
        print(f"\n✓ Best hidden_size: {best['hidden_size']} (val_loss={best['val_loss']:.6f})")
        return best['hidden_size']
    
    def search_num_layers(self, layers_list, hidden_size, lr=1e-3, epochs=20):
        """Find best number of layers"""
        print("\n" + "="*60)
        print("Phase 2: Number of Layers Search")
        print("="*60)
        
        for num_layers in layers_list:
            config = GRUConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                lr=lr,
                epochs=epochs
            )
            print(f"\nTrying num_layers={num_layers}...")
            run_and_benchmark(config, self.tracker)
        
        best = self.tracker.get_best()
        print(f"\n✓ Best num_layers: {best['num_layers']} (val_loss={best['val_loss']:.6f})")
        return best['num_layers']
    
    def search_lr(self, lr_list, hidden_size, num_layers, epochs=20):
        """Find best learning rate"""
        print("\n" + "="*60)
        print("Phase 3: Learning Rate Search")
        print("="*60)
        
        for lr in lr_list:
            config = GRUConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                lr=lr,
                epochs=epochs
            )
            print(f"\nTrying lr={lr:.1e}...")
            run_and_benchmark(config, self.tracker)
        
        best = self.tracker.get_best()
        print(f"\n✓ Best lr: {best['lr']:.1e} (val_loss={best['val_loss']:.6f})")
        return best['lr']


def main_search():
    """Run complete GRU parameter search"""
    
    date = datetime.now().strftime("%Y-%m-%d")
    tracker = ExperimentTracker(f"gru_search_{date}")
    searcher = GRUSequentialSearcher(tracker)
    
    # Phase 1: Hidden size sweep
    best_hidden = searcher.search_hidden_size(
        hidden_list=[32, 64, 128, 256],
        num_layers=2,
        lr=1e-3,
        epochs=20
    )
    
    # Phase 2: Number of layers sweep
    best_layers = searcher.search_num_layers(
        layers_list=[1, 2, 3, 4],
        hidden_size=best_hidden,
        lr=1e-3,
        epochs=20
    )
    
    # Phase 3: Learning rate sweep
    best_lr = searcher.search_lr(
        lr_list=[2e-3, 1e-3, 5e-4, 2e-4, 1e-4],
        hidden_size=best_hidden,
        num_layers=best_layers,
        epochs=20
    )
    
    # Phase 4: Final training
    print("\n" + "="*60)
    print("Phase 4: Final Training with Best Parameters")
    print("="*60)
    
    final_config = GRUConfig(
        hidden_size=best_hidden,
        num_layers=best_layers,
        lr=best_lr,
        dropout=0.2,
        epochs=100
    )
    
    final_val_loss, final_model = train_one(final_config, return_model=True)
    final_params = count_parameters(final_model)
    final_speed = measure_inference_speed(final_model, DEVICE, final_config.segment_length)
    
    tracker.add_result(final_config, final_val_loss, final_speed, final_params)
    
    print("\n" + "="*80)
    print("GRU SEARCH COMPLETE - FINAL SUMMARY")
    print("="*80)
    tracker.print_summary()
    
    return tracker


if __name__ == "__main__":
    tracker = main_search()