import pandas as pd
import matplotlib.pyplot as plt
import pdb
import scienceplots

# === Research Question 1 ===
# Overall comparison: baseline vs bandit retrieval.
# This function compares a selected metric (e.g., "precision@10") as a function of top_k_passages.
def plot_baseline_vs_bandit(metric, top_k_col='top_k_passages'):
    plt.figure(figsize=(8,6))
    # For bandit retrieval, aggregate the metric by top_k_passages (in case of multiple runs)
    bandit_agg = bandit_df.groupby(top_k_col)[metric].mean().reset_index()
    baseline_agg = baseline_df.groupby(top_k_col)[metric].mean().reset_index()
    
    plt.plot(bandit_agg[top_k_col], bandit_agg[metric], marker='o', label='Bandit Retrieval')
    plt.plot(baseline_agg[top_k_col], baseline_agg[metric], marker='s', label='Baseline')
    plt.xlabel(top_k_col)
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison: Baseline vs Bandit Retrieval')
    plt.legend()
    plt.grid(True)
    plt.show()


# === Research Question 2 ===
# Impact of different hyperparameter settings within bandit retrieval.
# This function groups the bandit data by a selected hyperparameter (e.g., beta)
def plot_hyperparameter_impact(data, metrics, hyperparameter, settings):
    plt.figure(figsize=(8,6))
    # sort the unique values in ascending order
    unique_values = sorted(data[hyperparameter].unique())
    # remove hyperparameter from settings
    new_settings = {k: v for k, v in settings.items() if k != hyperparameter}
    # only hyperparameter is different, other settings are the same
    results = {}
    for metric in metrics:
        result = {}
        for val in unique_values:
            # Start with rows that match the current hyperparameter value
            subset = data[data[hyperparameter] == val]
            
            # Filter rows where all other settings match the specified values
            for col, value in new_settings.items():
                subset = subset[subset[col] == value]

            # If we have matching data points after filtering
            if not subset.empty:
                # find the mean of the metric
                mean_metric = subset[metric].mean()
                result[val] = mean_metric
            else:
                # No data points match the criteria
                result[val] = float('nan')  # Use NaN to indicate missing data
                
        results[metric] = result

    # plot the results
    # x is the hyperparameter values
    # y is the metric scores, with each line representing a different metric
    # remove nan values
    unique_values = [val for val in unique_values if not pd.isna(results[metrics[0]][val])]
    
    # Define markers for each metric prefix
    marker_map = {
        'precision': 'o',   # circle
        'recall': 's',      # square
        'map': '^'          # triangle
    }

    # caption map
    caption_map = {
        'beta': 'Beta',
        'gpucb_percentage': 'GPUCB Percentage',
        'k_retrieval': 'Passages Retrieved',
        'llm_budget': 'LLM Budget',
        'top_k_passages': 'Top K Passages Used',
        'batch_size': 'Batch Size'
    }
    plt.style.use(['science', 'no-latex'])
    for metric in metrics:
        metric_values = [results[metric][val] for val in unique_values]
        # plt.plot(unique_values, metric_values, marker=marker_map[metric.split('@')[0]], label=f'{metric}')
        plt.plot(unique_values, metric_values, label=f'{metric}')
    
    plt.xlabel(caption_map[hyperparameter], fontsize=22)
    plt.ylabel('Score', fontsize=22)    
    plt.title(f'Performance Metrics vs {caption_map[hyperparameter]}', fontsize=22)
   # Move legend outside the plot at the bottom with horizontal layout
    plt.legend(fontsize=18, 
               loc='lower center',     # Position the legend at upper center...
               bbox_to_anchor=(0.5, -0.25),  # ...but offset below the plot
               ncol=len(metrics) # display items in a row
               )           # Add a frame around the legend

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.grid(True)
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Get metric type (map, precision, recall) for the filename
    metric_type = metrics[0].split('@')[0]
    
    # Save the plot with a more descriptive filename
    plt.savefig(f'plots/bandit_llm_budget{settings["llm_budget"]}_k_retrieval{settings["k_retrieval"]}_top_k_passages{settings["top_k_passages"]}_batch_size{settings["batch_size"]}_{hyperparameter}_{metric_type}_plot.pdf', bbox_inches='tight', dpi=300)
    plt.show()

# === Research Question 3 ===
# Compare bandit retrieval and LLM reranking based on the same LLM budgets,
# and estimate the budget needed to achieve a fixed performance threshold.
def plot_method_comparison(metric, budget_col_bandit='llm_budget', budget_col_llm='budget'):
    plt.figure(figsize=(8,6))
    # Aggregate metrics by budget for each method
    bandit_agg = bandit_df.groupby(budget_col_bandit)[metric].mean().reset_index()
    llm_agg = llm_reranking_df.groupby(budget_col_llm)[metric].mean().reset_index()
    
    plt.plot(bandit_agg[budget_col_bandit], bandit_agg[metric], marker='o', label='Bandit Retrieval')
    plt.plot(llm_agg[budget_col_llm], llm_agg[metric], marker='s', label='LLM Reranking')
    plt.xlabel('LLM Budget')
    plt.ylabel(metric)
    plt.title(f'{metric} vs LLM Budget: Bandit Retrieval vs LLM Reranking')
    plt.legend()
    plt.grid(True)
    plt.show()



# --- Estimating Budgets for a Fixed Performance Threshold ---
def find_budget_for_threshold(df, budget_col, metric, threshold):
    """
    Finds the smallest budget where the specified metric reaches or exceeds the threshold.
    Assumes that the data is monotonic or that the first occurrence is representative.
    """
    df_sorted = df.sort_values(budget_col)
    for _, row in df_sorted.iterrows():
        if row[metric] >= threshold:
            return row[budget_col]
    return None

def compare_budget_threshold(metric, threshold, budget_col_bandit='llm_budget', budget_col_llm='budget'):
    bandit_budget = find_budget_for_threshold(bandit_df, budget_col_bandit, metric, threshold)
    llm_budget = find_budget_for_threshold(llm_reranking_df, budget_col_llm, metric, threshold)
    
    print(f"For {metric} reaching the threshold {threshold}:")
    print(f"  Bandit Retrieval requires budget: {bandit_budget}")
    print(f"  LLM Reranking requires budget: {llm_budget}")


if __name__ == "__main__":# === Load Data ===
    # Make sure your CSV files are in the same directory or adjust the paths accordingly.
    bandit_df = pd.read_csv('travel_dest_all-MiniLM-L6-v2_bandit_results.csv')
    llm_reranking_df = pd.read_csv('travel_dest_all-MiniLM-L6-v2_llm_reranking_results.csv')
    # baseline_df = pd.read_csv('travel_dest_all-MiniLM-L6-v2_baseline_results.csv')

    # metrics = ['precision@10','recall@10','map@10','precision@30','recall@30','map@30','precision@50','recall@50','map@50']
    map_metrics = ['map@10','map@30','map@50']
    precision_metrics = ['precision@10','precision@30','precision@50']
    recall_metrics = ['recall@10','recall@30','recall@50']
    settings = {
        'llm_budget': 100,
        'gpucb_percentage': 0.5,
        'kernel': 'rbf',
        'k_retrieval': 1000,
        'top_k_passages': 5,
        'batch_size': 5,
        'beta': 3.0,
    }
    # hyperparameter = 'gpucb_percentage'
    hyperparameter = 'beta'
    plot_hyperparameter_impact(bandit_df, map_metrics, hyperparameter, settings)
    plot_hyperparameter_impact(bandit_df, precision_metrics, hyperparameter, settings)
    plot_hyperparameter_impact(bandit_df, recall_metrics, hyperparameter, settings)
    