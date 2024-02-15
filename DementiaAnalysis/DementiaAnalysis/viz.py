import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import interp
from itertools import cycle

def plot_histograms(dataframe, exclude_columns = ['ID', 'Diagnosis', 'Gender']):
    """
    Plots histograms for numerical columns in the given dataframe, excluding specified columns.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to plot histograms for.
    exclude_columns (list, optional): A list of column names to exclude from plotting. 
                                      Defaults to ['ID', 'Diagnosis', 'Gender'].

    Returns:
    None

    Example:
    >>> plot_histograms(df, exclude_columns=['ID', 'Age'])

    This will plot histograms for all numerical columns in df, excluding 'ID' and 'Age'.
    """
    
    numerical_columns = dataframe.select_dtypes(include=['number']).columns
    columns_to_plot = [col for col in numerical_columns if col not in exclude_columns]
    
    for column in columns_to_plot:
        plt.figure(figsize=(10, 6))
        sns.histplot(dataframe[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


def plot_boxplots(dataframe, grouping_variable, variables=None, groups=None, exclude_columns=['ID']):
    """
    Plots boxplots for specified variables in the dataframe, grouped by a specified categorical column.
    When variables and a single group are specified, combines boxplots in the same figure for the specified group.
    Otherwise, plots separate figures for each variable.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to plot boxplots for.
    grouping_variable (str): The name of the categorical column to group by.
    variables (list, optional): A list of variable names to plot. If None, all numerical columns are plotted.
    groups (list, optional): A list or a single group name within the grouping_variable column to include in the plot.
                             If None, all groups are included.
    exclude_columns (list, optional): A list of column names to exclude from plotting. Defaults to ['ID'].

    Example:
    >>> plot_boxplots(df, grouping_variable='Diagnosis', 
                      variables=['Mini-Mental State Examination Total Score', 'Digit Task'], 
                      groups=['Logopenic Variant Primary Progressive Aphasia'])
    """
    
    if grouping_variable not in dataframe.columns or dataframe[grouping_variable].dtype not in ['object', 'category']:
        raise ValueError(f"{grouping_variable} must be a categorical column.")

    # Filter dataframe for specified groups if provided
    if groups is not None:
        dataframe = dataframe[dataframe[grouping_variable].isin(groups)]
    
    if variables is not None:
        # Convert a single variable string to a list
        if isinstance(variables, str):
            variables = [variables]
        # Ensure specified variables are not in exclude_columns and exist in dataframe
        variables = [var for var in variables if var not in exclude_columns and var in dataframe.columns]

        # Melt the dataframe for side-by-side plotting
        melted_df = dataframe.melt(id_vars=[grouping_variable], value_vars=variables, var_name='Variable', value_name='Score')
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Variable', y='Score', hue=grouping_variable, data=melted_df, palette="Set3")
        plt.title(f'Boxplot of specified variables grouped by {grouping_variable}')
        plt.xlabel('Variable')
        plt.ylabel('Score')
        plt.legend(title=grouping_variable)
        plt.show()

    else:
        # If no variables are specified, plot all numerical columns not in exclude_columns
        numerical_columns = dataframe.select_dtypes(include=['number']).columns
        variables = [col for col in numerical_columns if col not in exclude_columns]

        for variable in variables:
            plt.figure(figsize=(12, 8))
            sns.boxplot(x=grouping_variable, y=variable, data=dataframe)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Boxplot of {variable} grouped by {grouping_variable}')
            plt.xlabel(grouping_variable)
            plt.ylabel(variable)
            plt.show()


def plot_correlation_matrix(dataframe, exclude_columns=['ID'], cmap='coolwarm', vmin=-1, vmax=1):
    """
    Plots a correlation matrix heatmap for numerical columns in the given dataframe, excluding specified columns.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to plot the correlation matrix for.
    exclude_columns (list, optional): A list of column names to exclude from the correlation matrix. 
                                      Defaults to ['ID'].

    Returns:
    None

    Example:
    >>> plot_correlation_matrix(df)

    This will plot a correlation matrix heatmap for all numerical columns in df, excluding 'ID'.
    """
    
    # Select all numerical columns
    numerical_columns = dataframe.select_dtypes(include=['number']).columns
    columns_to_plot = [col for col in numerical_columns if col not in exclude_columns]

    # Calculate the correlation matrix for the selected columns
    correlation_matrix = dataframe[columns_to_plot].corr()

    # Set the diagonal of the correlation matrix to zero
    np.fill_diagonal(correlation_matrix.values, 0)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Generate a heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap,
                cbar=True, square=True, linewidths=.5, vmin=vmin, vmax=vmax)
    
    plt.title('Correlation Matrix Heatmap')
    plt.show()



def plot_pairwise_correlations_with_regression(dataframe, exclude_columns=['ID', 'Diagnosis', 'Gender'], include_columns=None):
    """
    Plots pairwise correlations with regression lines for numerical columns in the given dataframe, excluding specified columns.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to plot the pairwise correlations for.
    exclude_columns (list, optional): A list of column names to exclude from the plots. 
                                      Defaults to ['ID', 'Diagnosis', 'Gender'].
    include_columns (list, optional): A list of column names to include in the plots. 
                                      If not provided, all numerical columns except those in exclude_columns will be used.

    Returns:
    None

    Example:
    >>> plot_pairwise_correlations_with_regression(df)

    This will plot pairwise correlations with regression lines for all numerical columns in df, excluding 'ID', 'Diagnosis', and 'Gender'.
    """
    
    # Select only numerical columns
    numerical_columns = dataframe.select_dtypes(include=['number']).columns

    if include_columns is None:
        columns_to_plot = [col for col in numerical_columns if col not in exclude_columns]
    else:
        columns_to_plot = [col for col in numerical_columns if col in include_columns]

    numerical_dataframe = dataframe[columns_to_plot]
    
    # Use Seaborn's pairplot function to plot pairwise relationships
    # Set kind='reg' to draw a regression line for each pair
    g = sns.pairplot(numerical_dataframe, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})
    
    # Adjust the title of the plot
    g.fig.suptitle("Pairwise Correlations with Regression Line", y=1.08) # Adjust the title position
    
    plt.show()



def scatter_plot_with_regression(dataframe, x, y, add_regression_line=True):
    """
    Plots a scatter plot for the specified x and y variables in the given dataframe. 
    Optionally, adds a regression line to the plot.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the data to plot.
    x (str): The name of the column to use as the x variable in the scatter plot.
    y (str): The name of the column to use as the y variable in the scatter plot.
    add_regression_line (bool, optional): Whether to add a regression line to the scatter plot. 
                                          Defaults to True.

    Returns:
    None

    Example:
    >>> scatter_plot_with_regression(df, 'Age', 'Score', add_regression_line=True)

    This will plot a scatter plot for 'Age' and 'Score' in df, and add a regression line to the plot.
    """
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(dataframe[x], dataframe[y], alpha=0.5)
    
    # Optional: Add regression line
    if add_regression_line:
        sns.regplot(x=x, y=y, data=dataframe, scatter=False, color="red")
    
    plt.title(f'Relationship between {x} and {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def plot_multiclass_roc(true_labels, probabilities, classes):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for multiclass classification.

    Parameters:
    true_labels (numpy.ndarray): A 2D array where each row represents a sample and each column represents a class. 
                                 Each element should be 1 if the sample belongs to the corresponding class and 0 otherwise.
    probabilities (numpy.ndarray): A 2D array where each row represents a sample and each column represents a class. 
                                   Each element should be the probability that the sample belongs to the corresponding class.
    n_classes (int): The number of classes.
    classes (list): A list of class names. The order should correspond to the columns of true_labels and probabilities.

    Returns:
    None

    This function computes and plots the ROC curve and ROC area for each class. It also computes and plots the micro-average 
    ROC curve and ROC area. The ROC curves are plotted with different colors for each class. The plot includes a legend 
    showing the area under the ROC curve for each class.

    Example:
    >>> plot_multiclass_roc(y_true, y_prob, 3, ['class1', 'class2', 'class3'])

    This will plot the ROC curves for a 3-class classification problem with classes 'class1', 'class2', and 'class3'.
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC')
    plt.legend(loc="lower right")
    plt.show()



def plot_confusion_matrix(true_labels, predicted_labels, classes):
    """
    Plots a confusion matrix for the given true and predicted labels.

    Parameters:
    true_labels (numpy.ndarray): A 2D array where each row represents a sample and each column represents a class. 
                                 Each element should be 1 if the sample belongs to the corresponding class and 0 otherwise.
    predicted_labels (numpy.ndarray): A 1D array of predicted class indices for each sample.
    classes (list): A list of class names. The order should correspond to the columns of true_labels.

    Returns:
    None

    This function computes the confusion matrix from the true and predicted labels, and then plots it using a heatmap. 
    The x-axis represents the predicted labels and the y-axis represents the true labels. The color of each cell in the 
    heatmap corresponds to the number of samples with a particular pair of true and predicted labels.

    Example:
    >>> plot_confusion_matrix(y_true, y_pred, ['class1', 'class2', 'class3'])

    This will plot the confusion matrix for a 3-class classification problem with classes 'class1', 'class2', and 'class3'.
    """
    cm = confusion_matrix(true_labels.argmax(axis=1), predicted_labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, ax=ax, xticklabels=classes, yticklabels=classes)
    
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Rotate the tick labels for clarity
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)

    plt.show()


def plot_feature_importance(feature_importance, title='Feature Importance', ylabel='Feature Importance'):
    """
    This function plots the feature importance in a bar chart format.

    Parameters:
    feature_importance (dict): A dictionary where keys are feature names and values are their corresponding importance.
    title (str, optional): The title of the plot. Default is 'Feature Importance'.
    xlabel (str, optional): The label for the x-axis. Default is 'Feature'.
    ylabel (str, optional): The label for the y-axis. Default is 'Importance'.

    Returns:
    None: This function doesn't return anything; it shows a plot.

    Note:
    The function sorts the features based on their importance in descending order before plotting.

    Example:
    >>> plot_feature_importance({'feature1': 0.2, 'feature2': 0.1, 'feature3': 0.7})
    """
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
    plt.bar(feature_importance.keys(), feature_importance.values())
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

