import os
import argparse
import sys
import json

from ablation_study import (
    AblationTestbedClassifier,
    AblationGlobalResultsTracker,
    run_cmv_ablation,
    run_mage_ablation,
    load_feature_groups,
    Tee
)
from main_classifier import MAGETestbedConfig


class CumulativeAblationTracker:
    """Track cumulative ablation results."""
    
    def __init__(self, removed_groups_list, output_dir='../results/cumulative_ablation'):
        self.output_dir = output_dir
        self.removed_groups = removed_groups_list
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
    
    def add_result(self, testbed_name, accuracy, auroc, f1_macro):
        """Add a cumulative ablation result."""
        self.results.append({
            'removed_groups': self.removed_groups,
            'num_groups_removed': len(self.removed_groups),
            'testbed': testbed_name,
            'accuracy': float(accuracy),
            'auroc': float(auroc),
            'f1_macro': float(f1_macro)
        })
    
    def save_results(self):
        """Save cumulative ablation results."""
        removed_str = '_'.join(self.removed_groups) if self.removed_groups else 'none'
        filepath = os.path.join(self.output_dir, f'cumulative_{removed_str}.json')
        
        results_with_metadata = {
            'removed_groups': self.removed_groups,
            'num_groups_removed': len(self.removed_groups),
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nCumulative ablation results saved to: {filepath}")


def run_cumulative_experiment(removed_groups_list, all_feature_groups, config_manager,
                              testbed, cmv_path=None, cmv_run=False):
    """Run one cumulative ablation experiment."""
    
    features_to_remove = []
    for group in removed_groups_list:
        features_to_remove.extend(all_feature_groups[group])
    
    removed_str = '_'.join(removed_groups_list)
    print(f"\nRemoving {len(features_to_remove)} features from groups: {removed_groups_list}")
    
    global_tracker = CumulativeAblationTracker(removed_groups_list)
    
    if cmv_run and cmv_path:
        print("\n" + "#"*56)
        print("# RUNNING CMV TESTBEDS (CUMULATIVE ABLATION)")
        print("#"*56)
        
        for ai_model in ['combined', 'gpt', 'llama', 'mistral']:
            try:
                run_cmv_ablation('combined', ai_model, removed_str,
                               features_to_remove, cmv_path, global_tracker)
            except Exception as e:
                print(f"Error in CMV {ai_model}: {e}\n")
    
    print("\n" + "#"*56)
    print("# RUNNING MAGE TESTBEDS (CUMULATIVE ABLATION)")
    print("#"*56)
    
    if testbed == 4:
        print("\n" + "="*56)
        print("TESTBED 4: Arbitrary-domains & Arbitrary-models")
        print("="*56)
        
        config = config_manager.arbitrary_domains_arbitrary_models()
        try:
            run_mage_ablation(config, 'combined', removed_str,
                            features_to_remove, config_manager, global_tracker)
        except Exception as e:
            print(f"Error: {e}\n")
    
    elif testbed == 7:
        print("\n" + "="*56)
        print("TESTBED 7: Unseen-domains & Unseen-model")
        print("="*56)
        
        config = config_manager.unseen_domains_unseen_model()
        try:
            run_mage_ablation(config, 'combined', removed_str,
                            features_to_remove, config_manager, global_tracker)
        except Exception as e:
            print(f"Error: {e}\n")
    else:
        print("Invalid testbed number for cumulative ablation!")
    
    return global_tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cumulative ablation study')
    parser.add_argument('--testbed', type=int, choices=[4, 7], default=7,
                        help='Testbed number (4 or 7)')
    parser.add_argument('--ablation-order', type=str, required=True,
                        help='Comma-separated list of feature groups in ablation order')
    parser.add_argument('--feature-json', type=str, default='../data/feature_consistency_report.json',
                        help='Path to feature groups JSON file')
    parser.add_argument('--data-path', type=str, default='../data/mage',
                        help='Path to MAGE data')
    parser.add_argument('--cmv-path', type=str, default='../data/cmv',
                        help='Path to CMV data')
    parser.add_argument('--output-dir', type=str, default='../results/cumulative_ablation',
                        help='Output directory for results')
    parser.add_argument('--log-dir', type=str, default='../logs/cumulative_ablation',
                        help='Directory for log files')
    parser.add_argument('--run-cmv', action='store_true',
                        help='Also run CMV testbeds')
    
    args = parser.parse_args()
    
    ablation_order = [g.strip() for g in args.ablation_order.split(',')]
    
    print("="*56)
    print("### CUMULATIVE ABLATION STUDY ###")
    print(f">> Ablation order: {' -> '.join(ablation_order)}")
    print(f">> Testbed: {args.testbed}")
    print("="*56)
    
    all_feature_groups = load_feature_groups(args.feature_json)
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    config_manager = MAGETestbedConfig(base_path=args.data_path)
    master_results = []
    
    for i in range(1, len(ablation_order)):
        removed_groups = ablation_order[:i]
        remaining_groups = ablation_order[i:]
        
        print("\n" + "="*70)
        print(f"CUMULATIVE ABLATION STEP {i}/{len(ablation_order)}")
        print(f"Removed groups ({i}): {removed_groups}")
        print(f"Remaining groups ({len(remaining_groups)}): {remaining_groups}")
        print("="*70)
        
        log_file = os.path.join(args.log_dir, f'step_{i:02d}_testbed{args.testbed}.txt')
        sys.stdout = Tee(log_file)
        
        tracker = run_cumulative_experiment(removed_groups, all_feature_groups,
                                           config_manager, args.testbed,
                                           args.cmv_path, args.run_cmv)
        
        tracker.save_results()
        master_results.extend(tracker.results)
        
        print(f"\n{'='*70}")
        print(f"STEP {i} COMPLETE")
        print(f"{'='*70}\n")
        
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal
    
    master_filepath = os.path.join(args.output_dir, 'all_steps_summary.json')
    with open(master_filepath, 'w') as f:
        json.dump({
            'ablation_order': ablation_order,
            'testbed': args.testbed,
            'all_results': master_results
        }, f, indent=2)
    
    print(f"\nMaster summary saved to: {master_filepath}")
    
    print("\n" + "="*70)
    print("CUMULATIVE ABLATION STUDY COMPLETE")
    print("="*70)