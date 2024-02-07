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