#%%
from DementiaAnalysis.importer import import_dataframe
from DementiaAnalysis.viz import plot_histograms, plot_boxplots, plot_correlation_matrix, plot_pairwise_correlations_with_regression, scatter_plot_with_regression, plot_multiclass_roc, plot_confusion_matrix, plot_feature_importance
from DementiaAnalysis.stats import perform_t_test, perform_correlation, perform_anova, perform_multiple_regression
from DementiaAnalysis.ml import classify_and_evaluate


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



# %%
from himalaya.ridge import RidgeCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance
import numpy as np
def classify_and_evaluate(df, grouping_variable, classes, features, estimator='svm', scoring='roc_auc', cross_validation='stratified_kfold', n_folds=5, feature_importance=False, model=None):
    """
    Classifies clinical diagnoses based on scores, optionally calculates feature importance, and returns objects for confusion matrix, ROC curve, etc.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    grouping_variable (str): The name of the categorical variable for classification.
    classes (list): The classes within the grouping variable to include in the classification.
    features (list): The list of features for classification.
    estimator (str): The classifier to use ('svm', 'random_forest', 'ridge', etc.).
    scoring (str): The scoring method ('roc_auc', 'accuracy', etc.).
    cross_validation (str): The cross-validation strategy ('stratified_kfold').
    n_folds (int): The number of folds for cross-validation.
    feature_importance (bool): Whether to calculate and return feature importance scores (default False).
    model (sklearn estimator, optional): An optional model to fit and evaluate. If provided, this model will be used instead of the estimator parameter.
    
    Returns:
    dict: A dictionary containing true labels, predicted labels, probabilities, scores, and optionally feature importances.
    """
    # Filter the dataframe for specified classes
    df_filtered = df[df[grouping_variable].isin(classes)]
    
    # Prepare data
    X = df_filtered[features]
    y = df_filtered[grouping_variable]
    y = label_binarize(y, classes=classes)  # Binarize labels for multi-class ROC analysis
    n_classes = y.shape[1]
    
    # Define classifier
    if model is not None:
        classifier = model
    elif estimator == 'svm':
        classifier = SVC(probability=True)
    elif estimator == 'random_forest':
        classifier = RandomForestClassifier()
    elif estimator == 'ridge':
        classifier = RidgeCV()
    else:
        raise ValueError("Unsupported estimator. Please add the estimator to the function.")
    
    # Cross-validation
    if cross_validation == 'stratified_kfold':
        cv = StratifiedKFold(n_splits=n_folds)
    else:
        raise ValueError("Unsupported cross-validation strategy. Please add it to the function.")

    if estimator == 'ridge':
        classifier.fit(X,y)
    else:
         # Perform cross-validated predictions
        y_pred = cross_val_predict(classifier, X, y.argmax(axis=1), cv=cv, method='predict')
        y_proba = cross_val_predict(classifier, X, y.argmax(axis=1), cv=cv, method='predict_proba')
        scores = cross_val_score(classifier, X, y.argmax(axis=1), cv=cv, scoring=scoring)
        
        results = {
                'true_labels': y,
                'predicted_labels': y_pred,
                'probabilities': y_proba,
                'scores': scores,
                'mean_score': np.mean(scores),
                'std_dev_score': np.std(scores),
                'Grouping Variable': grouping_variable,
                'Classes': classes,
                'Features': features,
                'Model': estimator,
                'Cross-Validation': cross_validation,
                'N Folds': n_folds,
                'Scoring': scoring
        }

        # Calculate feature importance if requested
        if feature_importance:
                feature_importances = np.zeros(len(features))

                for train_index, test_index in cv.split(X, y.argmax(axis=1)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    classifier.fit(X_train, y_train.argmax(axis=1))
                    
                    if estimator == 'random_forest':
                            # Direct method for models with built-in feature importance
                            feature_importances += classifier.feature_importances_
                    elif estimator == 'svm':
                            # Permutation importance for models without built-in feature importance
                            r = permutation_importance(classifier, X_test, y_test.argmax(axis=1), n_repeats=30)
                            feature_importances += r.importances_mean

                # Average feature importance over folds
                feature_importances /= n_folds
                feature_importance_dict = {features[i]: feature_importances[i] for i in range(len(features))}
                results['feature_importance'] = feature_importance_dict

    # Save best alphas if model is RidgeCV
    if isinstance(classifier, RidgeCV):
        results = {
                'best_alphas': classifier.best_alphas_}
        
    return results, classifier

#%%
import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(0)

# Create a DataFrame with 246 rows and 558 columns
# df = pd.DataFrame(np.random.rand(246, 558))
df = pd.DataFrame(np.random.rand(20, 50))


# Print the first few rows of the DataFrame
print(df.head())

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_time_series_and_correlation(df):
    """
    Plots time series for each ROI and a correlation matrix with circles representing effect size.

    Parameters:
    df (pd.DataFrame): The dataframe containing the time series data. Each column represents an ROI and each row represents a time point.
    """
    # Time series plot
    plt.figure(figsize=(10, 10))
    for column in df.columns:
        plt.plot(df[column], label=column)
    plt.legend()
    plt.title('Time Series for Each ROI')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

    # Correlation matrix
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask for the upper triangle
    size = np.abs(corr).values  # Absolute value of correlation coefficients for circle sizes

    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, mask=mask, cbar=False)
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            plt.scatter(i+0.5, j+0.5, s=size[i][j]*1000, c=[corr.values[i][j]], cmap='coolwarm', vmin=-1, vmax=1, marker='o')
    plt.title('Correlation Matrix with Circle Size Representing Effect Size')
    plt.xticks(ticks=np.arange(0.5, len(df.columns)), labels=df.columns, rotation=90)
    plt.yticks(ticks=np.arange(0.5, len(df.columns)), labels=df.columns, rotation=0)
    plt.gca().invert_yaxis()
    plt.show()
# %%
plot_time_series_and_correlation(df)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation(df):
    """
    Plots a correlation matrix with circles representing the correlation coefficients.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data. Each column represents a variable.
    """
    # Correlation matrix
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask for the upper triangle
    size = np.abs(corr).values  # Absolute value of correlation coefficients for circle sizes

    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5, mask=mask, cbar=False)
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            plt.scatter(i+0.5, j+0.5, s=size[i][j]*1000, c=[corr.values[i][j]], cmap='coolwarm', vmin=-1, vmax=1, marker='o')
    plt.title('Correlation Matrix with Circle Size Representing Correlation Coefficient')
    plt.xticks(ticks=np.arange(0.5, len(df.columns)), labels=df.columns, rotation=90)
    plt.yticks(ticks=np.arange(0.5, len(df.columns)), labels=df.columns, rotation=0)
    plt.gca().invert_yaxis()
    plt.show()


#%% Plot the correlation matrix
plot_correlation(df)
# %%
plot_correlation_matrix(df)
# %%
