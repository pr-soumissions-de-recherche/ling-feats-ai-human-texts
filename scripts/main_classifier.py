from sklearnex import patch_sklearn 
patch_sklearn()
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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


class Tee:
    """Redirect stdout to both terminal and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w')
    
    def __del__(self):
        self.file.close()
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()


class TestbedClassifier:
    """SVM classifier for AI text detection testbeds."""
    
    def __init__(self, testbed_name, feature_group, output_dir='../results'):
        self.testbed_name = testbed_name
        self.feature_group = feature_group
        
        if testbed_name.startswith('1_fixed_domain_model_specific_'):
            domain = testbed_name.split('_')[5]
            self.output_dir = os.path.join(output_dir, 'testbed1_all_models', domain, testbed_name, feature_group)
        elif testbed_name.startswith('11_fixed_domain_model_family_specific_'):
            domain = testbed_name.split('_')[6]
            self.output_dir = os.path.join(output_dir, 'testbed_1_model_family', domain, testbed_name, feature_group)
        elif testbed_name.startswith('8_unseen_domain_model_pair_'):
            parts = testbed_name.split('_')
            domain = parts[5]
            family = parts[6]
            self.output_dir = os.path.join(output_dir, 'testbed8_unseen_pairs', domain, family, testbed_name, feature_group)
        else:
            self.output_dir = os.path.join(output_dir, testbed_name, feature_group)
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_features(self, df):
        """Extract features and labels from dataframe."""
        exclude_cols = ['doc_id', 'id', 'comment_id', 'post_id', 'label', 'text']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['label'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Labels: Human={sum(y==0)}, AI={sum(y==1)}")
        
        return X, y, feature_cols
    
    def train_svm(self, X_train, y_train, X_val, y_val, class_weight='balanced'):
        """Train SVM with validation monitoring."""
        print(f"\nTraining SVM with class_weight={class_weight}...")
        
        model = SVC(kernel='linear', C=1.0, random_state=42, class_weight=class_weight)
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model.fit(X_train_scaled, y_train)
        
        val_pred = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        print(f"Validation - Accuracy: {val_acc:.4f} | F1-Macro: {val_f1:.4f}")
        
        return model, scaler
    
    def evaluate(self, model, scaler, X_test, y_test):
        """Evaluate model with comprehensive metrics."""
        print(f"\nEvaluating on test set...")
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_scores = model.decision_function(X_test_scaled)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auroc': roc_auc_score(y_test, y_scores),
            'avg_precision': average_precision_score(y_test, y_scores),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_micro': f1_score(y_test, y_pred, average='micro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_human': precision[0],
            'precision_ai': precision[1],
            'recall_human': recall[0],
            'recall_ai': recall[1],
            'f1_human': f1[0],
            'f1_ai': f1[1],
            'avg_recall': (recall[0] + recall[1]) / 2,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def save_top_features(self, model, feature_names, top_k=50):
        """Save top k features with coefficients."""
        coefs = model.coef_[0]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs,
            'abs_importance': np.abs(coefs)
        })
        
        feature_importance = feature_importance.sort_values('abs_importance', ascending=False)
        
        top_features = feature_importance.head(top_k)
        filepath_top = os.path.join(self.output_dir, f'top_{top_k}_features.csv')
        top_features[['feature', 'coefficient']].to_csv(filepath_top, index=False)
        
        filepath_all = os.path.join(self.output_dir, f'all_features.csv')
        feature_importance[['feature', 'coefficient']].to_csv(filepath_all, index=False)
        
        print(f"\nTop {top_k} features saved to: {filepath_top}")
        return top_features
    
    def save_results(self, metrics):
        """Save comprehensive results to file."""
        filepath = os.path.join(self.output_dir, 'results.txt')
        
        with open(filepath, 'w') as f:
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
        
        print(f"Results saved to: {filepath}")


class GlobalResultsTracker:
    """Track results across all testbeds."""
    
    def __init__(self, output_dir='../results'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
    
    def add_result(self, testbed_name, accuracy, auroc, f1_macro):
        """Add a testbed result."""
        self.results.append({
            'testbed': testbed_name,
            'accuracy': float(accuracy),
            'auroc': float(auroc),
            'f1_macro': float(f1_macro)
        })
    
    def save_results(self):
        """Save all results to JSON."""
        filepath = os.path.join(self.output_dir, 'global_results.json')
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nGlobal results saved to: {filepath}")


class MAGETestbedConfig:
    """Configuration manager for MAGE testbeds."""
    
    def __init__(self, base_path="../data/mage"):
        self.base_path = base_path
        self.train_path = os.path.join(base_path, "train")
        self.val_path = os.path.join(base_path, "validation")
        self.test_path = os.path.join(base_path, "test")
        
        self.domains = ["cmv", "yelp", "xsum", "tldr", "eli5", "wp", 
                       "roct", "hswag", "squad", "sci_gen"]
        
        self.model_families = {
            'openai': ['gpt-3.5-trubo', 'text-davinci-003', 'text-davinci-002'],
            'llama': ['_7B', '_13B', '_30B', '_65B'],
            'glm': ['GLM130B'],
            'flan_t5': ['flan_t5_small', 'flan_t5_base', 'flan_t5_large', 
                       'flan_t5_xl', 'flan_t5_xxl'],
            'opt': ['opt_125m', 'opt_350m', 'opt_1.3b', 'opt_2.7b', 
                   'opt_6.7b', 'opt_13b', 'opt_30b', 'opt_iml_30b', 
                   'opt_iml_max_1.3b'],
            'bigscience': ['bloom_7b', 't0_3b', 't0_11b'],
            'eleuther': ['gpt_j', 'gpt_neox']
        }
    
    def fixed_domain_model_specific(self, domain, model='gpt_j'):
        """Testbed 1: Fixed-domain & Model-specific."""
        config = {
            'name': f'1_fixed_domain_model_specific_{domain}_{model}',
            'train': {
                'human': f"{self.train_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_model_path(self.train_path, domain, model)
            },
            'val': {
                'human': f"{self.val_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_model_path(self.val_path, domain, model)
            },
            'test': {
                'human': f"{self.test_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_model_path(self.test_path, domain, model)
            }
        }
        return config
    
    def fixed_domain_model_family_specific(self, domain, model_family):
        """Testbed 11: Fixed-domain & Model-family-specific."""
        config = {
            'name': f'11_fixed_domain_model_family_specific_{domain}_{model_family}',
            'train': {
                'human': f"{self.train_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_in_family_for_domain(self.train_path, domain, model_family)
            },
            'val': {
                'human': f"{self.val_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_in_family_for_domain(self.val_path, domain, model_family)
            },
            'test': {
                'human': f"{self.test_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_in_family_for_domain(self.test_path, domain, model_family)
            }
        }
        return config
    
    def arbitrary_domains_model_specific(self, model_family):
        """Testbed 2: Arbitrary-domains & Model-specific."""
        config = {
            'name': f'2_arbitrary_domains_model_specific_{model_family}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_models_in_family(self.train_path, model_family)
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_models_in_family(self.val_path, model_family)
            },
            'test': {
                'human': [f"{self.test_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_models_in_family(self.test_path, model_family)
            }
        }
        return config
    
    def fixed_domain_arbitrary_models(self, domain):
        """Testbed 3: Fixed-domain & Arbitrary-models."""
        config = {
            'name': f'3_fixed_domain_arbitrary_models_{domain}',
            'train': {
                'human': f"{self.train_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_for_domain(self.train_path, domain)
            },
            'val': {
                'human': f"{self.val_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_for_domain(self.val_path, domain)
            },
            'test': {
                'human': f"{self.test_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_for_domain(self.test_path, domain)
            }
        }
        return config
    
    def arbitrary_domains_arbitrary_models(self):
        """Testbed 4: Arbitrary-domains & Arbitrary-models."""
        config = {
            'name': '4_arbitrary_domains_arbitrary_models',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.train_path)
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.val_path)
            },
            'test': {
                'human': [f"{self.test_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.test_path)
            }
        }
        return config
    
    def unseen_models(self, excluded_family):
        """Testbed 5: Unseen Models."""
        train_families = [f for f in self.model_families.keys() if f != excluded_family]
        
        config = {
            'name': f'5_unseen_models_{excluded_family}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': []
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': []
            },
            'test': {
                'human': [f"{self.test_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_models_in_family(self.test_path, excluded_family)
            }
        }
        
        for family in train_families:
            config['train']['ai'].extend(self._find_all_models_in_family(self.train_path, family))
            config['val']['ai'].extend(self._find_all_models_in_family(self.val_path, family))
        
        return config
    
    def unseen_domains(self, excluded_domain):
        """Testbed 6: Unseen Domains."""
        train_domains = [d for d in self.domains if d != excluded_domain]
        
        config = {
            'name': f'6_unseen_domains_{excluded_domain}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in train_domains],
                'ai': []
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in train_domains],
                'ai': []
            },
            'test': {
                'human': f"{self.test_path}/Human/{excluded_domain}/samples/samples_features",
                'ai': self._find_all_models_for_domain(self.test_path, excluded_domain)
            }
        }
        
        for d in train_domains:
            config['train']['ai'].extend(self._find_all_models_for_domain(self.train_path, d))
            config['val']['ai'].extend(self._find_all_models_for_domain(self.val_path, d))
        
        return config
    
    def unseen_domains_unseen_model(self):
        """Testbed 7: Unseen-domains & Unseen-model."""
        config = {
            'name': '7_unseen_domains_unseen_model',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.train_path)
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.val_path)
            },
            'test': {
                'human': [f"{self.test_path}/unseen/Human/{d}/{d}_human/{d}_human_features" 
                         for d in ['cnn', 'dialogsum', 'imdb', 'pubmed']],
                'ai': [f"{self.test_path}/unseen/AI/{d}/{d}_gpt4/{d}_gpt4_features" 
                      for d in ['cnn', 'dialogsum', 'imdb', 'pubmed']]
            }
        }
        return config
    
    def unseen_domains_unseen_model_separate(self, test_domain):
        """Testbed 7.1: Unseen-domains & Unseen-model (per domain)."""
        config = {
            'name': f'71_unseen_domains_unseen_model_{test_domain}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.train_path)
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.val_path)
            },
            'test': {
                'human': f"{self.test_path}/unseen/Human/{test_domain}/{test_domain}_human/{test_domain}_human_features",
                'ai': f"{self.test_path}/unseen/AI/{test_domain}/{test_domain}_gpt4/{test_domain}_gpt4_features"
            }
        }
        return config
    
    def unseen_domain_model_pair(self, excluded_domain, excluded_family):
        """Testbed 8: Unseen Domain-Model Pair."""
        train_domains = [d for d in self.domains if d != excluded_domain]
        train_families = [f for f in self.model_families.keys() if f != excluded_family]
        
        config = {
            'name': f'8_unseen_domain_model_pair_{excluded_domain}_{excluded_family}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in train_domains],
                'ai': []
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in train_domains],
                'ai': []
            },
            'test': {
                'human': f"{self.test_path}/Human/{excluded_domain}/samples/samples_features",
                'ai': self._find_all_models_in_family_for_domain(self.test_path, excluded_domain, excluded_family)
            }
        }
        
        for domain in train_domains:
            for family in train_families:
                train_paths = self._find_all_models_in_family_for_domain(self.train_path, domain, family)
                config['train']['ai'].extend(train_paths)
                
                val_paths = self._find_all_models_in_family_for_domain(self.val_path, domain, family)
                config['val']['ai'].extend(val_paths)
        
        return config
    
    def get_all_available_models(self, split_path):
        """Get all unique model names across domains."""
        models = set()
        for domain in self.domains:
            domain_path = f"{split_path}/AI/{domain}"
            if os.path.exists(domain_path):
                for family in os.listdir(domain_path):
                    family_path = os.path.join(domain_path, family)
                    if os.path.isdir(family_path):
                        for model_folder in os.listdir(family_path):
                            models.add(model_folder)
        return sorted(list(models))
    
    def load_data_from_config(self, config, feature_group):
        """Load train/val/test data from config."""
        def load_from_paths(paths, label):
            dfs = []
            if isinstance(paths, str):
                paths = [paths]
            
            for path in paths:
                feature_file = os.path.join(path, f"{feature_group}_features.csv")
                if os.path.exists(feature_file):
                    df = pd.read_csv(feature_file)
                    df['label'] = 0 if 'Human'.lower() in str(path).lower() else 1
                    dfs.append(df)
            
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        
        train_human = load_from_paths(config['train']['human'], label=0)
        train_ai = load_from_paths(config['train']['ai'], label=1)
        train_df = pd.concat([train_human, train_ai], ignore_index=True)
        
        val_human = load_from_paths(config['val']['human'], label=0)
        val_ai = load_from_paths(config['val']['ai'], label=1)
        val_df = pd.concat([val_human, val_ai], ignore_index=True)
        
        test_human = load_from_paths(config['test']['human'], label=0)
        test_ai = load_from_paths(config['test']['ai'], label=1)
        test_df = pd.concat([test_human, test_ai], ignore_index=True)
        
        print(f"Train: {len(train_df)} ({sum(train_df['label']==0)} H, {sum(train_df['label']==1)} AI)")
        print(f"Val:   {len(val_df)} ({sum(val_df['label']==0)} H, {sum(val_df['label']==1)} AI)")
        print(f"Test:  {len(test_df)} ({sum(test_df['label']==0)} H, {sum(test_df['label']==1)} AI)")
        
        return train_df, val_df, test_df
    
    def _find_model_path(self, split_path, domain, model):
        """Find path for a specific model."""
        for family, models in self.model_families.items():
            for m in models:
                if model in m or m in model:
                    path = f"{split_path}/AI/{domain}/{family}/{model}/{model}_features"
                    if os.path.exists(path):
                        return path
        return None
    
    def _find_all_models_in_family_for_domain(self, split_path, domain, family):
        """Find all models in a family for a domain."""
        paths = []
        family_path = f"{split_path}/AI/{domain}/{family}"
        if os.path.exists(family_path):
            for model_folder in os.listdir(family_path):
                model_features_path = os.path.join(family_path, model_folder, f"{model_folder}_features")
                if os.path.isdir(model_features_path):
                    paths.append(model_features_path)
        return paths
    
    def _find_all_models_in_family(self, split_path, family):
        """Find all models in a family across domains."""
        paths = []
        for domain in self.domains:
            family_path = f"{split_path}/AI/{domain}/{family}"
            if os.path.exists(family_path):
                for model_folder in os.listdir(family_path):
                    model_features_path = os.path.join(family_path, model_folder, f"{model_folder}_features")
                    if os.path.isdir(model_features_path):
                        paths.append(model_features_path)
        return paths
    
    def _find_all_models_for_domain(self, split_path, domain):
        """Find all models for a domain."""
        paths = []
        domain_path = f"{split_path}/AI/{domain}"
        if os.path.exists(domain_path):
            for family in os.listdir(domain_path):
                family_path = os.path.join(domain_path, family)
                if os.path.isdir(family_path):
                    for model_folder in os.listdir(family_path):
                        model_features_path = os.path.join(family_path, model_folder, f"{model_folder}_features")
                        if os.path.exists(model_features_path):
                            paths.append(model_features_path)
        return paths
    
    def _find_all_ai_data(self, split_path):
        """Find all AI data across domains."""
        paths = []
        for domain in self.domains:
            paths.extend(self._find_all_models_for_domain(split_path, domain))
        return paths


def run_cmv_testbed(feature_group, ai_model, base_path, global_tracker=None):
    """Run CMV testbed."""
    print("\n" + "="*56)
    print(f"CMV TESTBED - {feature_group} - {ai_model}")
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
    
    classifier = TestbedClassifier(f'cmv_{ai_model}', feature_group)
    X, y, feature_names = classifier.prepare_features(df)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    print(f"\nSplits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val)
    metrics = classifier.evaluate(model, scaler, X_test, y_test)
    
    if global_tracker:
        global_tracker.add_result(f'cmv_{ai_model}', metrics['accuracy'], metrics['auroc'], metrics['f1_macro'])
    
    print(f"\nRESULTS:")
    print(f"Accuracy: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    classifier.save_top_features(model, feature_names, top_k=50)
    classifier.save_results(metrics)


def run_mage_testbed(config, feature_group, config_manager, global_tracker=None):
    """Run MAGE testbed."""
    print("\n" + "="*56)
    print(f"MAGE TESTBED: {config['name']} - {feature_group}")
    print("="*56)
    
    train_df, val_df, test_df = config_manager.load_data_from_config(config, feature_group)
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Skipping - insufficient data")
        return
    
    classifier = TestbedClassifier(config['name'], feature_group)
    X_train, y_train, feature_names = classifier.prepare_features(train_df)
    X_val, y_val, _ = classifier.prepare_features(val_df)
    X_test, y_test, _ = classifier.prepare_features(test_df)
    
    model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val)
    metrics = classifier.evaluate(model, scaler, X_test, y_test)
    
    if global_tracker:
        global_tracker.add_result(config['name'], metrics['accuracy'], metrics['auroc'], metrics['f1_macro'])
    
    print(f"\nRESULTS:")
    print(f"Accuracy: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    classifier.save_top_features(model, feature_names, top_k=50)
    classifier.save_results(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AI text detection experiments')
    parser.add_argument('--testbed', type=int, required=True,
                        help='Testbed number (1-8)')
    parser.add_argument('--feature-group', type=str, default='combined',
                        help='Feature group to use')
    parser.add_argument('--data-path', type=str, default='../data/mage',
                        help='Path to MAGE data')
    parser.add_argument('--cmv-path', type=str, default='../data/cmv',
                        help='Path to CMV data')
    parser.add_argument('--output-dir', type=str, default='../results',
                        help='Output directory for results')
    parser.add_argument('--log-dir', type=str, default='../logs',
                        help='Directory for log files')
    parser.add_argument('--run-cmv', action='store_true',
                        help='Also run CMV testbeds')
    
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.log_dir, f'testbed{args.testbed}.txt')
    sys.stdout = Tee(log_file)
    
    global_tracker = GlobalResultsTracker(output_dir=args.output_dir)
    config_manager = MAGETestbedConfig(base_path=args.data_path)
    
    print("="*56)
    print(f"### TESTBED {args.testbed} EXPERIMENT ###")
    print(f">> Feature Group: {args.feature_group}")
    print("="*56)
    
    if args.run_cmv:
        print("\n" + "#"*56)
        print("# RUNNING CMV TESTBEDS")
        print("#"*56)
        
        for ai_model in ['combined', 'gpt', 'llama', 'mistral']:
            try:
                run_cmv_testbed(args.feature_group, ai_model, args.cmv_path, global_tracker)
            except Exception as e:
                print(f"Error in CMV {ai_model}: {e}\n")
    
    print("\n" + "#"*56)
    print("# RUNNING MAGE TESTBEDS")
    print("#"*56)
    
    if args.testbed == 1:
        all_models = config_manager.get_all_available_models(config_manager.train_path)
        for model in all_models:
            for domain in config_manager.domains:
                config = config_manager.fixed_domain_model_specific(domain, model)
                if config['train']['ai'] is None:
                    continue
                try:
                    run_mage_testbed(config, args.feature_group, config_manager, global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n") 

    elif args.testbed == 11: #Testbed 11: Fixed-domain & Model-family-specific
        for family in config_manager.model_families.keys():
            for domain in config_manager.domains:
                config = config_manager.fixed_domain_model_family_specific(domain, family)
                # Check if family exists for this domain
                if not config['train']['ai']:
                    print(f"Skipping {family} for {domain} - family not found")
                    continue
                try:
                    run_mage_testbed(config, args.feature_group, config_manager, global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n")

    elif args.testbed == 2:
        for family in config_manager.model_families.keys():
            config = config_manager.arbitrary_domains_model_specific(family)
            try:
                run_mage_testbed(config, args.feature_group, config_manager, global_tracker)
            except Exception as e:
                print(f"Error: {e}\n")
    
    elif args.testbed == 3:
        for domain in config_manager.domains:
            config = config_manager.fixed_domain_arbitrary_models(domain)
            try:
                run_mage_testbed(config, args.feature_group, config_manager, global_tracker)
            except Exception as e:
                print(f"Error: {e}\n")
    
    elif args.testbed == 4:
        config = config_manager.arbitrary_domains_arbitrary_models()
        try:
            run_mage_testbed(config, args.feature_group, config_manager, global_tracker)
        except Exception as e:
            print(f"Error: {e}\n")
    
    elif args.testbed == 5:
        for family in config_manager.model_families.keys():
            config = config_manager.unseen_models(family)
            try:
                run_mage_testbed(config, args.feature_group, config_manager, global_tracker)
            except Exception as e:
                print(f"Error: {e}\n")
    
    elif args.testbed == 6:
        for domain in config_manager.domains:
            config = config_manager.unseen_domains(domain)
            try:
                run_mage_testbed(config, args.feature_group, config_manager, global_tracker)
            except Exception as e:
                print(f"Error: {e}\n")
    
    elif args.testbed == 7:
        config = config_manager.unseen_domains_unseen_model()
        try:
            run_mage_testbed(config, args.feature_group, config_manager, global_tracker)
        except Exception as e:
            print(f"Error: {e}\n")


    elif args.testbed == 71:
        unseen_test_domains = ['cnn', 'dialogsum', 'imdb', 'pubmed']
        for test_domain in unseen_test_domains:
            config = config_manager.unseen_domains_unseen_model_separate(test_domain)
            try:
                run_mage_testbed(config, args.feature_group, config_manager, global_tracker)
            except Exception as e:
                print(f"Error: {e}\n")

    elif args.testbed == 8:
        for domain in config_manager.domains:
            for family in config_manager.model_families.keys():
                config = config_manager.unseen_domain_model_pair(domain, family)
                if not config['test']['ai']:
                    continue
                try:
                    run_mage_testbed(config, args.feature_group, config_manager, global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n")
    
    global_tracker.save_results()
    
    print("\n" + "="*56)
    print("EXPERIMENT COMPLETE")
    print("="*56)
    
    sys.stdout.file.close()
    sys.stdout = sys.stdout.terminal