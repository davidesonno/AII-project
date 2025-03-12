def merge_csv_to_dataframe(input_folder, **kwargs):
    import pandas as pd
    import os
    
    """Merge all CSV files in a folder into a single pandas DataFrame."""
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the folder.")
        return None

    dataframes = [pd.read_csv(os.path.join(input_folder, file), **kwargs) for file in csv_files]
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"Merged {len(csv_files)} CSV files")
    
    return merged_df