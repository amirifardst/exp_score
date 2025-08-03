import pandas as pd
from scipy.stats import kendalltau

# Load the CSV files
def get_rank(acc_df_dir, df_exp_dir):
    df_acc = pd.read_csv(acc_df_dir)  # Columns: model_name, accuracy (sorted by accuracy)
    df_exp = pd.read_csv(df_exp_dir)  # Columns: model_name, exp_score (sorted by exp_score)

    # Merge on model_name to align models present in both files
    merged_df = pd.merge(df_acc, df_exp, on='Architecture')

    # Since the CSVs are sorted by accuracy and exp_score respectively,
    # we need to assign ranks based on their order in each CSV.

    # Assign ranks based on order in each CSV (starting from 1)
    merged_df['rank_accuracy'] = merged_df['Accuracy'].rank(ascending=False, method='min')
    merged_df['rank_exp_score'] = merged_df['Mean Normalized Expressivity Score'].rank(ascending=False, method='min')

    # Alternatively, if you trust the CSV order, you can assign ranks by position:
    # merged_df['rank_accuracy'] = merged_df['accuracy'].rank(ascending=False, method='min')
    # merged_df['rank_exp_score'] = merged_df['exp_score'].rank(ascending=False, method='min')

    # Compute Kendall's Tau correlation between the two ranks
    tau, p_value = kendalltau(merged_df['rank_accuracy'], merged_df['rank_exp_score'])
    merged_df.to_csv('results/cifar10/ranked_models.csv', index=False)
    print(f"Kendall's Tau correlation: {tau}, p-value: {p_value}")
    return tau, p_value, merged_df
