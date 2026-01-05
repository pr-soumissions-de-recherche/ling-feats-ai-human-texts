import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            confusion_matrix, roc_auc_score, average_precision_score, f1_score)
import os
import argparse
import sys
import json
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

from main_classifier import TestbedClassifier, GlobalResultsTracker, MAGETestbedConfig, Tee


def load_feature_groups(json_path):
    """Load feature groups from JSON file."""
    with open(json_path, 'r') as f:
        all_groups = json.load(f)
    
    if 'combined' in all_groups:
        del all_groups['combined']
    
    feature_groups = {group: data['features'] for group, data in all_groups.items()}
    return feature_groups


class AblationTestbedClassifier(TestbedClassifier):
    """Extended classifier for ablation studies."""
    
    def __init__(self, testbed_name, feature_group, ablated_group, output_dir='../results/ablation'):
        self.ablated_group = ablated_group
        super().__init__(testbed_name, feature_group, output_dir)
        
        if testbed_name.startswith('1_fixed_domain_model_specific_'):
            domain = testbed_name.split('_')[5]
            self.output_dir = os.path.join(output_dir, f'ablated_{ablated_group}', 
                                          'testbed1_all_models', domain, testbed_name, feature_group)
        elif testbed_name.startswith('11_fixed_domain_model_family_specific_'):
            domain = testbed_name.split('_')[6]
            self.output_dir = os.path.join(output_dir, f'ablated_{ablated_group}',
                                          'testbed_1_model_family', domain, testbed_name, feature_group)
        else:
            self.output_dir = os.path.join(output_dir, f'ablated_{ablated_group}', 
                                          testbed_name, feature_group)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_features_with_ablation(self, df, features_to_remove):
        """Extract features and labels, excluding ablated features."""
        exclude_cols = ['doc_id', 'id', 'comment_id', 'post_id', 'label', 'text']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        available_features_to_remove = [f for f in features_to_remove if f in df.columns]
        feature_cols = [col for col in feature_cols if col not in available_features_to_remove]
        
        X = df[feature_cols].values
        y = df['label'].values
        
        print(f"Features shape after ablation: {X.shape}")
        print(f"Removed {len(available_features_to_remove)} features from '{self.ablated_group}' group")
        print(f"Labels: Human={sum(y==0)}, AI={sum(y==1)}")
        
        return X, y, feature_cols
    
    def save_results_with_ablation(self, metrics):
        """Save results with ablation information."""
        filepath = os.path.join(self.output_dir, 'results.txt')
        
        with open(filepath, 'w') as f:
            f.write(f"ABLATION STUDY\n")
            f.write(f"Ablated Feature Group: {self.ablated_group}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Testbed: {self.testbed_name}\n")
            f.write(f"Feature Group: {self.feature_group}\n")
            f.write("="*60 + "\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*60 + "\n")
            f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
            f.write(f"AUROC:              {metrics['auroc']:.4f}\n")
            f.write(f"Average Precision:  {metrics['avg_precision']:.4f}\n")
            f.write(f"F1 Macro:           {metrics['f1_macro']:.4f}\n")
            f.write(f"F1 Micro:           {metrics['f1_micro']:.4f}\n")
            f.write(f"F1 Weighted:        {metrics['f1_weighted']:.4f}\n")
            f.write(f"Average Recall:     {metrics['avg_recall']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-"*60 + "\n")
            f.write("Human (Class 0):\n")
            f.write(f"  Precision: {metrics['precision_human']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_human']:.4f}\n")
            f.write(f"  F1:        {metrics['f1_human']:.4f}\n\n")
            
            f.write("AI (Class 1):\n")
            f.write(f"  Precision: {metrics['precision_ai']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_ai']:.4f}\n")
            f.write(f"  F1:        {metrics['f1_ai']:.4f}\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-"*60 + "\n")
            cm = metrics['confusion_matrix']
            f.write(f"          Predicted\n")
            f.write(f"          Human  AI\n")
            f.write(f"True Human {cm[0,0]:5d} {cm[0,1]:5d}\n")
            f.write(f"     AI    {cm[1,0]:5d} {cm[1,1]:5d}\n")
        
        print(f"Ablation results saved to: {filepath}")


class AblationGlobalResultsTracker(GlobalResultsTracker):
    """Track ablation results across all testbeds."""
    
    def __init__(self, ablated_group, output_dir='../results/ablation', testbed=None):
        self.ablated_group = ablated_group
        self.output_dir = os.path.join(output_dir, f'ablated_{ablated_group}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        self.testbed = testbed
    
    def save_results(self):
        """Save ablation results to JSON."""
        if self.testbed is None:
            filepath = os.path.join(self.output_dir, 'global_results.json')
        else:
            filepath = os.path.join(self.output_dir, f'global_results_testbed{self.testbed}.json')
        
        results_with_metadata = {
            'ablated_group': self.ablated_group,
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nAblation global results saved to: {filepath}")


def run_cmv_ablation(feature_group, ai_model, ablated_group, features_to_remove, 
                     base_path, global_tracker=None):
    """Run CMV testbed with ablation."""
    print("\n" + "="*56)
    print(f"CMV ABLATION - {feature_group} - {ai_model}")
    print(f"Ablating: {ablated_group}")
    print("="*56)
    
    human_path = f"{base_path}/Human/comments_features/{feature_group}_features.csv"
    human_df = pd.read_csv(human_path)
    
    if ai_model == 'combined':
        ai_paths = [
            f"{base_path}/AI/gpt/gpt_features/{feature_group}_features.csv",
            f"{base_path}/AI/llama/llama_features/{feature_group}_features.csv",
            f"{base_path}/AI/mistral/mistral_features/{feature_group}_features.csv"
        ]
        ai_df = pd.concat([pd.read_csv(p) for p in ai_paths], ignore_index=True)
    else:
        ai_path = f"{base_path}/AI/{ai_model}/{ai_model}_features/{feature_group}_features.csv"
        ai_df = pd.read_csv(ai_path)
    
    if len(human_df) > len(ai_df):
        human_df = human_df.sample(n=len(ai_df), random_state=42).reset_index(drop=True)
    
    human_df['label'] = 0
    ai_df['label'] = 1
    df = pd.concat([human_df, ai_df], ignore_index=True)
    
    print(f"Total samples: {len(df)} ({sum(df['label']==0)} H, {sum(df['label']==1)} AI)")
    
    classifier = AblationTestbedClassifier(f'cmv_{ai_model}', feature_group, ablated_group)
    X, y, feature_names = classifier.prepare_features_with_ablation(df, features_to_remove)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    print(f"\nSplits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val, class_weight='balanced')
    metrics = classifier.evaluate(model, scaler, X_test, y_test)
    
    if global_tracker:
        global_tracker.add_result(f'cmv_{ai_model}', metrics['accuracy'], metrics['auroc'], metrics['f1_macro'])
    
    print(f"\nRESULTS:")
    print(f"Accuracy: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    classifier.save_top_features(model, feature_names, top_k=50)
    classifier.save_results_with_ablation(metrics)


def run_mage_ablation(config, feature_group, ablated_group, features_to_remove, 
                      config_manager, global_tracker=None):
    """Run MAGE testbed with ablation."""
    print("\n" + "="*56)
    print(f"MAGE ABLATION: {config['name']} - {feature_group}")
    print(f"Ablating: {ablated_group}")
    print("="*56)
    
    train_df, val_df, test_df = config_manager.load_data_from_config(config, feature_group)
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Skipping - insufficient data")
        return
    
    classifier = AblationTestbedClassifier(config['name'], feature_group, ablated_group)
    
    X_train, y_train, feature_names = classifier.prepare_features_with_ablation(train_df, features_to_remove)
    X_val, y_val, _ = classifier.prepare_features_with_ablation(val_df, features_to_remove)
    X_test, y_test, _ = classifier.prepare_features_with_ablation(test_df, features_to_remove)
    
    model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val, class_weight='balanced')
    metrics = classifier.evaluate(model, scaler, X_test, y_test)
    
    if global_tracker:
        global_tracker.add_result(config['name'], metrics['accuracy'], metrics['auroc'], metrics['f1_macro'])
    
    print(f"\nRESULTS:")
    print(f"Accuracy: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    classifier.save_top_features(model, feature_names, top_k=50)
    classifier.save_results_with_ablation(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--testbed', type=int, required=True,
                        help='Testbed number (1-8)')
    parser.add_argument('--feature-group', type=str, default='combined',
                        help='Feature group to use')
    parser.add_argument('--feature-json', type=str, default='../data/feature_consistency_report.json',
                        help='Path to feature groups JSON file')
    parser.add_argument('--data-path', type=str, default='../data/mage',
                        help='Path to MAGE data')
    parser.add_argument('--cmv-path', type=str, default='../data/cmv',
                        help='Path to CMV data')
    parser.add_argument('--output-dir', type=str, default='../results/ablation',
                        help='Output directory for results')
    parser.add_argument('--log-dir', type=str, default='../logs/ablation',
                        help='Directory for log files')
    parser.add_argument('--run-cmv', action='store_true',
                        help='Also run CMV testbeds')
    
    args = parser.parse_args()
    
    feature_groups_dict = load_feature_groups(args.feature_json)
    
    print(f"\nFeature groups loaded:")
    for group, features in feature_groups_dict.items():
        print(f"  - {group}: {len(features)} features")
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*56)
    print("### ABLATION STUDY ###")
    print(f">> Testing feature group: {args.feature_group}")
    print(f">> Number of ablation experiments: {len(feature_groups_dict)}")
    print("="*56)
    
    config_manager = MAGETestbedConfig(base_path=args.data_path)
    
    for ablated_group, features_to_remove in feature_groups_dict.items():
        print("\n" + "#"*56)
        print(f"# ABLATING GROUP: {ablated_group.upper()}")
        print(f"# Removing {len(features_to_remove)} features")
        print("#"*56)
        
        log_file = os.path.join(args.log_dir, f'ablation_{ablated_group}.txt')
        sys.stdout = Tee(log_file)
        
        global_tracker = AblationGlobalResultsTracker(ablated_group, 
                                                     output_dir=args.output_dir,
                                                     testbed=args.testbed)
        
        if args.run_cmv:
            print("\n" + "#"*56)
            print("# RUNNING CMV TESTBEDS (ABLATION)")
            print("#"*56)
            
            for ai_model in ['combined', 'gpt', 'llama', 'mistral']:
                try:
                    run_cmv_ablation(args.feature_group, ai_model, ablated_group,
                                   features_to_remove, args.cmv_path, global_tracker)
                except Exception as e:
                    print(f"Error in CMV {ai_model} ablation: {e}\n")
        
        print("\n" + "#"*56)
        print("# RUNNING MAGE TESTBEDS (ABLATION)")
        print("#"*56)
        
        if args.testbed == 1:
            all_models = config_manager.get_all_available_models(config_manager.train_path)
            for model in all_models:
                for domain in config_manager.domains:
                    config = config_manager.fixed_domain_model_specific(domain, model)
                    if config['train']['ai'] is None:
                        continue
                    try:
                        run_mage_ablation(config, args.feature_group, ablated_group,
                                        features_to_remove, config_manager, global_tracker)
                    except Exception as e:
                        print(f"Error: {e}\n")
        
        elif args.testbed == 2:
            for family in config_manager.model_families.keys():
                config = config_manager.arbitrary_domains_model_specific(family)
                try:
                    run_mage_ablation(config, args.feature_group, ablated_group,
                                    features_to_remove, config_manager, global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n")
        
        elif args.testbed == 3:
            for domain in config_manager.domains:
                config = config_manager.fixed_domain_arbitrary_models(domain)
                try:
                    run_mage_ablation(config, args.feature_group, ablated_group,
                                    features_to_remove, config_manager, global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n")
        
        elif args.testbed == 4:
            config = config_manager.arbitrary_domains_arbitrary_models()
            try:
                run_mage_ablation(config, args.feature_group, ablated_group,
                                features_to_remove, config_manager, global_tracker)
            except Exception as e:
                print(f"Error: {e}\n")
        
        elif args.testbed == 5:
            for family in config_manager.model_families.keys():
                config = config_manager.unseen_models(family)
                try:
                    run_mage_ablation(config, args.feature_group, ablated_group,
                                    features_to_remove, config_manager, global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n")
        
        elif args.testbed == 6:
            for domain in config_manager.domains:
                config = config_manager.unseen_domains(domain)
                try:
                    run_mage_ablation(config, args.feature_group, ablated_group,
                                    features_to_remove, config_manager, global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n")
        
        elif args.testbed == 7:
            config = config_manager.unseen_domains_unseen_model()
            try:
                run_mage_ablation(config, args.feature_group, ablated_group,
                                features_to_remove, config_manager, global_tracker)
            except Exception as e:
                print(f"Error: {e}\n")
        
        elif args.testbed == 8:
            for domain in config_manager.domains:
                for family in config_manager.model_families.keys():
                    config = config_manager.unseen_domain_model_pair(domain, family)
                    if not config['test']['ai']:
                        continue
                    try:
                        run_mage_ablation(config, args.feature_group, ablated_group,
                                        features_to_remove, config_manager, global_tracker)
                    except Exception as e:
                        print(f"Error: {e}\n")
        
        global_tracker.save_results()
        
        print("\n" + "="*56)
        print(f"ABLATION OF {ablated_group.upper()} COMPLETE")
        print("="*56)
        
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal
    
    print("\n" + "="*56)
    print("ALL ABLATION EXPERIMENTS COMPLETE")
    print("="*56)