import os
import pandas as pd

# Path to the results directory
results_dir = 'results'
nas_dir = os.path.join(results_dir, 'nas')
os.makedirs(nas_dir, exist_ok=True)

mean_scores = []

# Loop through folders in results directory
for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    if os.path.isdir(folder_path) and folder.startswith('Archticure'):
        # Find the first CSV file in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            continue
        csv_path = os.path.join(folder_path, csv_files[0])
        df = pd.read_csv(csv_path, index_col=0)
        if 'mean' in df.index and 'Normalized Expressivity Score' in df.columns:
            mean_value = df.loc['mean', 'Normalized Expressivity Score']
            mean_scores.append({'Architecture': folder, 'Mean Normalized Expressivity Score': mean_value})

# Save to DataFrame and CSV
mean_df = pd.DataFrame(mean_scores)
output_path = os.path.join(nas_dir, 'architecture_mean_scores.csv')
mean_df.to_csv(output_path, index=False)
print(f"Saved mean scores to {output_path}")
