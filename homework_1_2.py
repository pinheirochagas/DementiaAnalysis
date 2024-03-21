#%%
from DementiaAnalysis.importer import import_dataframe
from DementiaAnalysis.vis import *
from DementiaAnalysis.stats import perform_t_test, perform_correlation, perform_anova, perform_multiple_regression
from DementiaAnalysis.ml import classify_and_evaluate
import DementiaAnalysis.neuroimaging as nimg

import pandas as pd

#%% Import data from csv file
df = import_dataframe('/Users/pinheirochagas/Library/CloudStorage/Box-Box/Pedro/Stanford/code/mac_ai/guides/simulated_cognitive_tests_dataset.csv')

#%% Plot distribution of scores for each cognitive domain
plot_histograms(df)

#%% Plot boxplots of scores for each cognitive domain grouped by diagnosis
plot_boxplots(df, 'Diagnosis')  # or 'Diagnosis'

#%% Plot correlation matrix
plot_correlation_matrix(df, vmin=-.8, vmax=.8, cmap='coolwarm')

#%% Plot pairwise correlations with regression lines
plot_pairwise_correlations_with_regression(df, include_columns=['Mini-Mental State Examination Total Score', 'Global Cognition',
       'Trail Making Test', 'Digit Task'])

#%% Scatter plot with a regression line
scatter_plot_with_regression(df, 'Mini-Mental State Examination Total Score', 'Global Cognition')

#%% Perform t-test between groups
result = perform_t_test(df, x='Mini-Mental State Examination Total Score', 
               grouping_variable='Diagnosis', 
               group1='Logopenic Variant Primary Progressive Aphasia', 
               group2='Semantic Variant Primary Progressive Aphasia')

plot_boxplots(df, grouping_variable='Diagnosis', 
              variables=['Mini-Mental State Examination Total Score'], 
              groups=['Logopenic Variant Primary Progressive Aphasia', 
                      'Semantic Variant Primary Progressive Aphasia'])  # or 'Diagnosis'
result.head()

#%% Perform t-test within group
result = perform_t_test(df, x=['Mini-Mental State Examination Total Score', 'Digit Task'],
                grouping_variable='Diagnosis', 
                group1='Logopenic Variant Primary Progressive Aphasia')

plot_boxplots(df, grouping_variable='Diagnosis', 
              variables=['Mini-Mental State Examination Total Score', 'Digit Task'], 
              groups=['Logopenic Variant Primary Progressive Aphasia'])  # or 'Diagnosis'

result.head()

# %% Perform correlation
result = perform_correlation(df, x='Mini-Mental State Examination Total Score', y='Global Cognition')
result.head()

# %%
# Example usage:
results, posthoc_results = perform_anova(df, 
                                         continuous_variable='Mini-Mental State Examination Total Score', 
                                         grouping_variable='Diagnosis', alpha=0.05, posthoc=True)


# %%
results = perform_anova(df, continuous_variable='Boston Naming Test Correct', 
                        grouping_variable='Diagnosis',
                        groups=['Logopenic Variant Primary Progressive Aphasia', 
                                'Semantic Variant Primary Progressive Aphasia',
                                'Nonfluent Variant Primary Progressive Aphasia'], 
                        alpha=0.05, posthoc=True)

plot_boxplots(df, grouping_variable='Diagnosis', 
              variables=['Boston Naming Test Correct'], 
              groups=['Logopenic Variant Primary Progressive Aphasia', 
                      'Semantic Variant Primary Progressive Aphasia',
                      'Nonfluent Variant Primary Progressive Aphasia'])  # or 'Diagnosis'
print(results)

# %% Perform multiple regression
results = perform_multiple_regression(df, 
                                      dependent_variable = 'Boston Naming Test Correct', 
                                      independent_variables=['Age', 'Attention', 'Digit Task'])
print(results)


# %% Classify and evaluate
# Example usage
grouping_variable = 'Diagnosis'
classes=['Logopenic Variant Primary Progressive Aphasia', 
        'Semantic Variant Primary Progressive Aphasia',
        'Nonfluent Variant Primary Progressive Aphasia']  # Example groups
features = ['Mini-Mental State Examination Total Score', 'Global Cognition',
            'Trail Making Test', 'Digit Task', 'Verbal Memory Recall (30-item)',
            'Verbal Memory Recall (10-item)', 'Attention',
            'Rey-Osterrieth Complex Figure Test', 'Memory and Naming',
            'Boston Naming Test Correct', 'Digit Forward Correct', 'Digit Backward',
            'Auditory Naming Correct',
            'Rey-Osterrieth Complex Figure Test (10-min delay)',
            'Rey-Osterrieth Complex Figure Test Recognition',
            'Geriatric Depression Scale Total Score',
            'Clinical Dementia Rating Total Score']  # Example features
#results = classify_and_evaluate(df, grouping_variable, classes, features, estimator='svm', scoring='accuracy', n_folds=5)
#results = classify_and_evaluate(df, grouping_variable, classes, features, estimator='random_forest', scoring='accuracy', cross_validation='stratified_kfold', n_folds=5, feature_importance=True)

results = classify_and_evaluate(df, grouping_variable, classes, features, estimator='random_forest', scoring='roc_auc', cross_validation='stratified_kfold', n_folds=5, feature_importance=True)


# %% Plot multiclass ROC
plot_multiclass_roc(results['true_labels'], results['probabilities'], classes=classes)

# %% Plot confusion matrix
plot_confusion_matrix(results['true_labels'], results['predicted_labels'], classes=classes)

# %% Plot feature importance 
plot_feature_importance(results['feature_importance'])

# %%
plot_boxplots(df, grouping_variable='Diagnosis', 
              variables=['Digit Backward'], 
              groups=['Logopenic Variant Primary Progressive Aphasia', 
                      'Semantic Variant Primary Progressive Aphasia',
                       'Nonfluent Variant Primary Progressive Aphasia'])  # or 'Diagnosis'
# %%
plot_boxplots(df, grouping_variable='Diagnosis', 
              variables=['Mini-Mental State Examination Total Score'], 
              groups=['Logopenic Variant Primary Progressive Aphasia', 
                      'Semantic Variant Primary Progressive Aphasia',
                       'Nonfluent Variant Primary Progressive Aphasia'])  # or 'Diagnosis'
