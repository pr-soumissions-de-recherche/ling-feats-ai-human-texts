import polars as pl
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def suggest_replacement_method(df, column):
    """Suggest replacement method based on column characteristics."""
    valid_values = df.filter(
        pl.col(column).is_not_null() & 
        pl.col(column).is_finite()
    )[column]
    
    if len(valid_values) == 0:
        return ("drop_column", "All values are invalid")
    
    unique_vals = valid_values.unique().to_list()
    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        return ("mode", "Binary feature")
    
    if valid_values.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
        return ("median", "Integer feature")
    
    if valid_values.dtype in [pl.Float32, pl.Float64]:
        mean_val = valid_values.mean()
        median_val = valid_values.median()
        std_val = valid_values.std()
        
        if std_val < 0.01:
            return ("mean", "Low variance")
        
        if abs(mean_val - median_val) > std_val:
            return ("median", "Skewed distribution")
        else:
            return ("mean", "Normal distribution")
    
    return ("median", "Default")


def check_error_values(root_dir, save_report=False):
    """Check all CSV files for NaN, Inf, and other problematic values."""
    report = {}
    
    feature_areas = ['surface', 'pos', 'lexical_richness', 'readability', 
                     'information', 'entities', 'semantic', 'emotion',
                     'psycholinguistic', 'morphological', 'dependency']
    
    print("\n" + "="*80)
    print("CHECKING FOR NaN AND Inf VALUES")
    print("="*80)
    print(f"Directory: {root_dir}\n")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_features.csv'):
                area = file.replace('_features.csv', '')
                if area not in feature_areas:
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                try:
                    df = pl.read_csv(file_path)
                    
                    if len(df.columns) <= 1:
                        continue
                    
                    error_features = {
                        'nan': [],
                        'inf': [],
                        'negative_inf': []
                    }
                    
                    total_errors = 0
                    total_cells = 0
                    
                    for col in df.columns:
                        if col in ['doc_id', 'id', 'comment_id']:
                            continue
                        
                        total_cells += len(df)
                        
                        nan_count = df[col].is_null().sum()
                        if nan_count > 0:
                            error_features['nan'].append({
                                'feature': col,
                                'count': nan_count,
                                'replacement': suggest_replacement_method(df, col)
                            })
                            total_errors += nan_count
                        
                        if df[col].dtype in [pl.Float32, pl.Float64]:
                            inf_count = df[col].is_infinite().sum()
                            if inf_count > 0:
                                pos_inf = (df[col] == float('inf')).sum()
                                neg_inf = (df[col] == float('-inf')).sum()
                                
                                if pos_inf > 0:
                                    error_features['inf'].append({
                                        'feature': col,
                                        'count': pos_inf,
                                        'replacement': suggest_replacement_method(df, col)
                                    })
                                    total_errors += pos_inf
                                
                                if neg_inf > 0:
                                    error_features['negative_inf'].append({
                                        'feature': col,
                                        'count': neg_inf,
                                        'replacement': suggest_replacement_method(df, col)
                                    })
                                    total_errors += neg_inf
                    
                    if total_errors > 0:
                        if area not in report:
                            report[area] = []
                        
                        error_details = {
                            'file_name': os.path.basename(file_path),
                            'file_path': rel_path,
                            'total_error_values': total_errors,
                            'total_cells': total_cells,
                            'error_percentage': round((total_errors / total_cells) * 100, 2),
                            'error_types': {}
                        }
                        
                        for error_type, features in error_features.items():
                            if features:
                                error_details['error_types'][error_type] = {
                                    'count': sum(f['count'] for f in features),
                                    'affected_features': [
                                        {
                                            'name': f['feature'],
                                            'error_count': f['count'],
                                            'replacement_method': f['replacement'][0],
                                            'replacement_reason': f['replacement'][1]
                                        }
                                        for f in features
                                    ]
                                }
                        
                        report[area].append(error_details)
                        print(f"  {rel_path}: {total_errors} errors found")
                
                except Exception as e:
                    print(f"  Error reading {rel_path}: {str(e)}")
    
    for area in report:
        report[area] = sorted(report[area], 
                            key=lambda x: x['total_error_values'], 
                            reverse=True)
    
    total_files_with_errors = sum(len(files) for files in report.values())
    total_errors_all = sum(
        f['total_error_values'] 
        for files in report.values() 
        for f in files
    )
    
    print(f"\nSUMMARY:")
    print(f"  Files with errors: {total_files_with_errors}")
    print(f"  Total error values: {total_errors_all}")
    print(f"\n  Errors by feature area:")
    for area, files in sorted(report.items(), 
                              key=lambda x: sum(f['total_error_values'] for f in x[1]), 
                              reverse=True):
        area_errors = sum(f['total_error_values'] for f in files)
        print(f"    {area:20s}: {area_errors:8d} errors in {len(files)} files")
    
    if save_report and report:
        report_path = os.path.join(root_dir, 'error_value_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Error report saved to: {report_path}")
    
    return report


def get_features_from_dir(root_dir):
    """Collect all unique features from a directory."""
    features = set()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_features.csv'):
                df = pl.read_csv(os.path.join(root, file))
                features.update(df.columns)
    return features


def process_dataset(input_dir, output_dir, features_to_keep):
    """Process dataset: filter features, handle errors, combine files."""
    
    for root, dirs, files in os.walk(input_dir):
        feature_files = [f for f in files if f.endswith('_features.csv')]
        if not feature_files:
            continue
        
        rel_path = os.path.relpath(root, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(out_path, exist_ok=True)
        
        all_group_data = {}
        
        for file in feature_files:
            file_path = os.path.join(root, file)
            area = file.replace('_features.csv', '')
            
            df = pl.read_csv(file_path)
            
            cols_to_keep = ['doc_id'] + [
                col for col in df.columns 
                if col in features_to_keep
            ]
            df_filtered = df.select(cols_to_keep)
            
            for col in df_filtered.columns:
                if col == 'doc_id':
                    continue
                
                if df_filtered[col].dtype in [pl.Float32, pl.Float64]:
                    df_filtered = df_filtered.with_columns(
                        pl.when(pl.col(col).is_infinite())
                        .then(None)
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
                
                if df_filtered[col].null_count() > 0:
                    method, reason = suggest_replacement_method(df_filtered, col)
                    
                    if method == "mean":
                        fill_value = df_filtered[col].mean()
                    elif method == "median":
                        fill_value = df_filtered[col].median()
                    elif method == "mode":
                        fill_value = df_filtered[col].mode().first()
                    else:
                        fill_value = df_filtered[col].median()
                    
                    if fill_value is None:
                        fill_value = 0
                    
                    df_filtered = df_filtered.with_columns(
                        pl.col(col).fill_null(fill_value)
                    )
            
            df_filtered.write_csv(os.path.join(out_path, file))
            all_group_data[area] = df_filtered
        
        if all_group_data:
            combined = None
            for area, df in all_group_data.items():
                if combined is None:
                    combined = df
                else:
                    combined = combined.join(df, on='doc_id', how='left')
            
            combined.write_csv(os.path.join(out_path, 'combined_features.csv'))
            print(f"  {rel_path}: {len(feature_files)} groups -> combined ({len(combined.columns)-1} features)")


def prepare_features_for_classification(mage_dir, cmv_dir, output_dir):
    """Preprocess features: remove zero-variance, handle errors, align features."""
    
    print("\n" + "="*80)
    print("PREPARING FEATURES FOR CLASSIFICATION")
    print("="*80)
    
    print("\n[1/5] Loading zero-variance features...")
    mage_zero = json.load(open(os.path.join(mage_dir, 'zero_variance_features.json')))
    cmv_zero = json.load(open(os.path.join(cmv_dir, 'zero_variance_features.json')))
    
    all_zero_features = set()
    for features in list(mage_zero.values()) + list(cmv_zero.values()):
        all_zero_features.update(features)
    
    print(f"  Features to remove (zero variance): {len(all_zero_features)}")
    
    print("\n[2/5] Collecting features from both datasets...")
    mage_features = get_features_from_dir(mage_dir)
    cmv_features = get_features_from_dir(cmv_dir)
    
    common_features = mage_features & cmv_features
    
    print(f"  MAGE features: {len(mage_features)}")
    print(f"  CMV features: {len(cmv_features)}")
    print(f"  Common features: {len(common_features)}")
    
    non_informative = {'doc_id', 'id', 'comment_id', 'post_id', 'label', 'text', 'nlp'}
    features_to_keep = common_features - all_zero_features - non_informative
    
    print(f"  Final features to keep: {len(features_to_keep)}")
    
    print("\n[3/5] Processing MAGE dataset...")
    process_dataset(mage_dir, os.path.join(output_dir, 'mage'), features_to_keep)
    
    print("\n[4/5] Processing CMV dataset...")
    process_dataset(cmv_dir, os.path.join(output_dir, 'cmv'), features_to_keep)
    
    print("\n[5/5] Done!")
    print(f"  Prepared data saved to: {output_dir}/")


def verify_feature_consistency(root_dir):
    """Verify all feature files have consistent columns."""
    
    print("\n" + "="*80)
    print("VERIFYING FEATURE CONSISTENCY")
    print("="*80)
    
    feature_groups = defaultdict(lambda: {'files': [], 'features': set(), 'all_same': True})
    
    areas = ['surface', 'pos', 'lexical_richness', 'readability', 
             'information', 'entities', 'semantic', 'emotion',
             'psycholinguistic', 'morphological', 'dependency', 'combined']
    
    print("\n[1/2] Scanning files...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'combined_features.csv':
                area = 'combined'
            elif file.endswith('_features.csv'):
                area = file.replace('_features.csv', '')
            else:
                continue
            
            if area not in areas:
                continue
            
            file_path = os.path.join(root, file)
            df = pl.read_csv(file_path)
            
            features = set(df.columns) - {'doc_id'}
            
            feature_groups[area]['files'].append(file_path)
            
            if not feature_groups[area]['features']:
                feature_groups[area]['features'] = features
            else:
                if features != feature_groups[area]['features']:
                    feature_groups[area]['all_same'] = False
                    print(f"  Mismatch in {area}: {os.path.relpath(file_path, root_dir)}")
    
    print("\n[2/2] Checking consistency...")
    report = {}
    all_consistent = True
    
    for area in sorted(feature_groups.keys()):
        data = feature_groups[area]
        
        report[area] = {
            'features': sorted(list(data['features'])),
            'number': len(data['features']),
            'files_checked': len(data['files']),
            'all_consistent': data['all_same']
        }
        
        if data['all_same']:
            print(f"  {area:20s}: {len(data['features']):3d} features - Consistent across {len(data['files'])} files")
        else:
            print(f"  {area:20s}: INCONSISTENT!")
            all_consistent = False
    
    report_path = os.path.join(root_dir, 'feature_consistency_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved: {report_path}")
    
    if all_consistent:
        print("\n  All feature groups are consistent!")
    else:
        print("\n  Some feature groups have inconsistencies!")
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare and validate feature data for classification')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['check', 'prepare', 'verify', 'all'],
                        help='Mode: check (NaN/Inf), prepare (preprocess), verify (consistency), all (all three)')
    parser.add_argument('--data-dir', type=str, default='../data/features',
                        help='Directory for checking or verifying')
    parser.add_argument('--mage-dir', type=str, default='../data/mage_features_raw',
                        help='Raw MAGE features directory (for prepare mode)')
    parser.add_argument('--cmv-dir', type=str, default='../data/cmv_features_raw',
                        help='Raw CMV features directory (for prepare mode)')
    parser.add_argument('--output-dir', type=str, default='../data/features',
                        help='Output directory for prepared features')
    parser.add_argument('--save-report', action='store_true',
                        help='Save error report to JSON file')
    
    args = parser.parse_args()
    
    if args.mode == 'check':
        check_error_values(args.data_dir, save_report=args.save_report)
    
    elif args.mode == 'prepare':
        prepare_features_for_classification(args.mage_dir, args.cmv_dir, args.output_dir)
    
    elif args.mode == 'verify':
        verify_feature_consistency(args.data_dir)
    
    elif args.mode == 'all':
        print("\n" + "#"*80)
        print("# RUNNING ALL DATA PREPARATION STEPS")
        print("#"*80)
        
        print("\n\nSTEP 1: Checking raw data for errors...")
        print("-"*80)
        check_error_values(args.mage_dir, save_report=args.save_report)
        check_error_values(args.cmv_dir, save_report=args.save_report)
        
        print("\n\nSTEP 2: Preparing features...")
        print("-"*80)
        prepare_features_for_classification(args.mage_dir, args.cmv_dir, args.output_dir)
        
        print("\n\nSTEP 3: Verifying consistency...")
        print("-"*80)
        verify_feature_consistency(args.output_dir)
        
        print("\n\n" + "#"*80)
        print("# ALL STEPS COMPLETE")
        print("#"*80)