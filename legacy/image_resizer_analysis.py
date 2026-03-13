import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import os

# Load Results from CSV
def load_results(csv_file):
    return pd.read_csv(csv_file)

# Compute Summary Statistics for Each Method
def summary_stats(df):
    methods = df['Method'].unique()
    stats = []

    for method in methods:
        method_data = df[df['Method'] == method]
        mean_ssim = method_data['SSIM'].mean()
        std_ssim = method_data['SSIM'].std()
        mean_mse = method_data['MSE'].mean()
        std_mse = method_data['MSE'].std()
        stats.append({'Method': method, 'Mean SSIM': mean_ssim, 'Std SSIM': std_ssim, 
                      'Mean MSE': mean_mse, 'Std MSE': std_mse})
    
    summary_df = pd.DataFrame(stats)
    return summary_df

# Conduct Statistical Tests (ANOVA and Pairwise t-tests)
def statistical_tests(df):
    methods = df['Method'].unique()
    ssim_data = [df[df['Method'] == method]['SSIM'] for method in methods]
    mse_data = [df[df['Method'] == method]['MSE'] for method in methods]

    # ANOVA Test
    anova_ssim = f_oneway(*ssim_data)
    anova_mse = f_oneway(*mse_data)

    # Pairwise t-tests for SSIM
    ttest_results = []
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                ttest_ssim = ttest_ind(ssim_data[i], ssim_data[j])
                ttest_mse = ttest_ind(mse_data[i], mse_data[j])
                ttest_results.append({
                    'Method 1': method1,
                    'Method 2': method2,
                    'SSIM p-value': ttest_ssim.pvalue,
                    'MSE p-value': ttest_mse.pvalue
                })
    
    ttest_df = pd.DataFrame(ttest_results)
    return {'anova_ssim': anova_ssim, 'anova_mse': anova_mse, 'pairwise_ttests': ttest_df}

# Visualization: Boxplots, Histograms, and Scatter Plots
def plot_distributions(df, output_dir):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Method', y='SSIM', data=df)
    plt.title('SSIM Comparison Across Methods')
    plt.savefig(os.path.join(output_dir, 'ssim_boxplot.png'))
    plt.clf()
    
    sns.boxplot(x='Method', y='MSE', data=df)
    plt.title('MSE Comparison Across Methods')
    plt.savefig(os.path.join(output_dir, 'mse_boxplot.png'))
    plt.clf()
    
    for method in df['Method'].unique():
        sns.histplot(df[df['Method'] == method]['SSIM'], label=method, kde=True, element="step")
    plt.title('SSIM Distribution Across Methods')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'ssim_distribution.png'))
    plt.clf()

    for method in df['Method'].unique():
        sns.histplot(df[df['Method'] == method]['MSE'], label=method, kde=True, element="step")
    plt.title('MSE Distribution Across Methods')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mse_distribution.png'))
    plt.clf()
    
    sns.scatterplot(x='SSIM', y='MSE', hue='Method', data=df)
    plt.title('SSIM vs. MSE Across Methods')
    plt.savefig(os.path.join(output_dir, 'ssim_vs_mse.png'))
    plt.clf()

# Detect Outliers Using IQR Method
def detect_outliers(df):
    outlier_summary = []
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        q1_ssim = method_data['SSIM'].quantile(0.25)
        q3_ssim = method_data['SSIM'].quantile(0.75)
        iqr_ssim = q3_ssim - q1_ssim
        lower_bound_ssim = q1_ssim - 1.5 * iqr_ssim
        upper_bound_ssim = q3_ssim + 1.5 * iqr_ssim
        outliers_ssim = method_data[(method_data['SSIM'] < lower_bound_ssim) | (method_data['SSIM'] > upper_bound_ssim)]
        
        outlier_summary.append({'Method': method, 'SSIM Outliers': len(outliers_ssim)})
    return pd.DataFrame(outlier_summary)

# Save Analysis Output
def save_output(summary_df, test_results, outlier_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
    test_results['pairwise_ttests'].to_csv(os.path.join(output_dir, 'pairwise_ttests.csv'), index=False)
    outlier_df.to_csv(os.path.join(output_dir, 'outlier_summary.csv'), index=False)
    
    # Save ANOVA Results as text files
    with open(os.path.join(output_dir, 'anova_ssim.txt'), 'w') as f:
        f.write(str(test_results['anova_ssim']))
    with open(os.path.join(output_dir, 'anova_mse.txt'), 'w') as f:
        f.write(str(test_results['anova_mse']))

# Main Function to Run All Analysis
def main():
    # Load results CSV
    csv_file = csv_file = '/Users/jameswinslow/Documents/Projects/DataScience/IGPhotoResizer/resizing_comparison_results.csv'
    output_dir = 'analysis_output'
    df = load_results(csv_file)
    
    # Compute Summary Stats
    summary_df = summary_stats(df)
    print("Summary Statistics:\n", summary_df)
    
    # Perform Statistical Tests
    test_results = statistical_tests(df)
    print("ANOVA Results (SSIM):\n", test_results['anova_ssim'])
    print("ANOVA Results (MSE):\n", test_results['anova_mse'])
    
    # Plot and Save Visualizations
    plot_distributions(df, output_dir)
    
    # Detect and Save Outliers
    outlier_df = detect_outliers(df)
    print("Outlier Summary:\n", outlier_df)
    
    # Save All Results
    save_output(summary_df, test_results, outlier_df, output_dir)

if __name__ == "__main__":
    main()
