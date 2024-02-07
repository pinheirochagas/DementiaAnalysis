import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


def plot_boxplots(dataframe, group_by, exclude_columns=['ID']):
    """
    Plots boxplots for numerical columns in the given dataframe, grouped by a specified categorical column.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to plot boxplots for.
    group_by (str): The name of the categorical column to group by.
    exclude_columns (list, optional): A list of column names to exclude from plotting. 
                                      Defaults to ['ID'].

    Returns:
    None

    Raises:
    ValueError: If the group_by column is not found in the dataframe or is not categorical.

    Example:
    >>> plot_boxplots(df, group_by='Gender')

    This will plot boxplots for all numerical columns in df, grouped by 'Gender', excluding 'ID'.
    """
    
    # Check if the group_by column is categorical
    if group_by not in dataframe.columns or dataframe[group_by].dtype not in ['object', 'category']:
        print(f"{group_by} must be a categorical column.")
        return

    # Select all numerical columns
    numerical_columns = dataframe.select_dtypes(include=['number']).columns
    columns_to_plot = [col for col in numerical_columns if col not in exclude_columns]

    for column in columns_to_plot:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=group_by, y=column, data=dataframe)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Boxplot of {column} grouped by {group_by}')
        plt.xlabel(group_by)
        plt.ylabel(column)
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
