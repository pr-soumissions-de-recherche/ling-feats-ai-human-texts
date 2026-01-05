import polars as pl
import os
import argparse
from elfen import Extractor


def extract_features(input_file, output_folder, create_combined=False, normalize_method='none'):
    """Extract linguistic features using ELFEN toolkit."""
    
    feature_groups = [
        "surface", "lexical_richness", "emotion", "psycholinguistic",
        "readability", "morphological", "pos", "dependency",
        "semantic", "entities", "information"
    ]
    
    print(f"\n=== Processing dataset: {input_file} ===")
    print(f"Normalization method: {normalize_method}")
    
    df = pl.read_csv(input_file)
    
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    dataset_output_folder = os.path.join(output_folder, f"{dataset_name}_features")
    os.makedirs(dataset_output_folder, exist_ok=True)
    
    exclude_columns = {'nlp', 'id', 'comment_id', 'post_id'}
    all_previous_features = set()
    
    extractor = Extractor(data=df, model='en_core_web_lg')
    combined_features_data = None
    
    for group in feature_groups:
        print(f"Extracting {group} features...")
        
        extractor.extract_feature_group(feature_group=group)
        
        if normalize_method == 'token' and 'n_tokens' not in extractor.data.columns:
            print(f"Skipping token normalization for {group}, using standard normalization instead.")
            extractor.normalize("all")
        else:
            if normalize_method == 'token':
                extractor.token_normalize("all")
            elif normalize_method == 'standard':
                extractor.normalize("all")
            elif normalize_method == 'ratio':
                extractor.ratio_normalize("all", "token")
            elif normalize_method == 'rescale':
                extractor.rescale("all")
        
        print(f"Extracted features for {group}:")
        print(extractor.data.columns)
        
        safe_columns = []
        for col in extractor.data.columns:
            dtype = extractor.data[col].dtype
            if dtype in [pl.String, pl.Utf8, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64, pl.Boolean]:
                safe_columns.append(col)
        
        columns_to_save = []
        for col in safe_columns:
            if col in ['id', 'comment_id', 'doc_id']:
                columns_to_save.append(col)
            elif col not in exclude_columns and col not in all_previous_features:
                columns_to_save.append(col)
                all_previous_features.add(col)
        
        data_to_save = extractor.data.select(columns_to_save)
        
        output_file = os.path.join(dataset_output_folder, f"{group}_features.csv")
        data_to_save.write_csv(output_file)
        print(f"Saved {len(columns_to_save)} columns to {group}_features.csv")
        
        if create_combined:
            if combined_features_data is None:
                combined_features_data = data_to_save
            else:
                combined_features_data = combined_features_data.join(data_to_save, on='doc_id', how='left')
    
    if create_combined and combined_features_data is not None:
        combined_output_file = os.path.join(dataset_output_folder, "all_features_combined.csv")
        combined_features_data.write_csv(combined_output_file)
        print(f"\nSaved combined features file: all_features_combined.csv")
        print(f"Combined file contains {len(combined_features_data.columns)} columns")
    
    print(f"All feature groups extracted and saved for {dataset_name}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract linguistic features using ELFEN toolkit')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, default='../data/features',
                        help='Output directory path')
    parser.add_argument('--combined', '-c', action='store_true',
                        help='Create combined features file')
    parser.add_argument('--normalize', '-n', choices=['none', 'token', 'standard', 'ratio', 'rescale'],
                        default='none', help='Normalization method')
    
    args = parser.parse_args()
    
    extract_features(args.input, args.output, args.combined, args.normalize)