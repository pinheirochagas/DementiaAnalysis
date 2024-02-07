import numpy as np
import scipy.stats as stats
import pandas as pd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

def perform_t_test(dataframe, x, grouping_variable=None, group1=None, group2=None, alpha=0.05, tails="two-tailed"):
    if grouping_variable and group1 and group2:
        # Comparison between two groups for a single variable
        data1 = dataframe[dataframe[grouping_variable] == group1][x]
        data2 = dataframe[dataframe[grouping_variable] == group2][x]
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        paired = False
    elif grouping_variable and group1:
        # Paired comparison within a single group for two different variables
        if not isinstance(x, list) or len(x) != 2:
            return "Error: For paired comparison, x must be a list of two variable names."
        data1 = dataframe[dataframe[grouping_variable] == group1][x[0]]
        data2 = dataframe[dataframe[grouping_variable] == group1][x[1]]
        t_stat, p_value = stats.ttest_rel(data1, data2)
        paired = True
    else:
        return "Error: Invalid input parameters."
    
    # Adjust p-value for one-tailed test if specified
    if tails == "one-tailed":
        p_value /= 2
    
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(data1), np.mean(data2)
    sd1, sd2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    
    # Calculate effect size (Cohen's d)
    if not paired:
        pooled_sd = np.sqrt(((len(data1) - 1) * sd1**2 + (len(data2) - 1) * sd2**2) / (len(data1) + len(data2) - 2))
    else:
        pooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)
    effect_size = np.abs(mean1 - mean2) / pooled_sd
    
    # Confidence interval calculation
    if not paired:
        se = np.sqrt(sd1**2/len(data1) + sd2**2/len(data2))
        df = len(data1) + len(data2) - 2
        ci_range = stats.t.ppf(1 - alpha/2, df) * se  # For two-tailed
        if tails == "one-tailed":
            ci_range = stats.t.ppf(1 - alpha, df) * se  # Adjust for one-tailed
    else:
        diff = np.array(data1) - np.array(data2)
        se_diff = np.sqrt(np.var(diff) / len(diff))
        df = len(diff) - 1
        ci_range = stats.t.ppf(1 - alpha/2, df) * se_diff  # For two-tailed
        if tails == "one-tailed":
            ci_range = stats.t.ppf(1 - alpha, df) * se_diff  # Adjust for one-tailed
    
    ci_lower = (mean1 - mean2) - ci_range
    ci_upper = (mean1 - mean2) + ci_range
    
    # Organize output in a table
    results_table = pd.DataFrame({
        'Mean Group 1': [mean1],
        'SD Group 1': [sd1],
        'Mean Group 2': [mean2],
        'SD Group 2': [sd2],
        'T Value': [t_stat],
        'P Value': [p_value],
        'CI Lower': [ci_lower],
        'CI Upper': [ci_upper],
        'Effect Size (Cohen\'s d)': [effect_size]
    })
    
    return results_table



def perform_correlation(dataframe, x, y=None, grouping_variable=None, group=None, correlation_type='pearson'):
    """
    Calculates the correlation between two variables, with an option to calculate within a specific group.
    
    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the data.
    x (str): The name of the first variable.
    y (str, optional): The name of the second variable. If None and grouping_variable is specified, x is compared within the group.
    grouping_variable (str, optional): The column name for grouping the data. Used only if y is None.
    group (str, optional): The specific group within the grouping_variable column to include in the correlation. Used only if y is None.
    correlation_type (str, optional): The type of correlation to calculate ('pearson', 'spearman', 'kendall'). Defaults to 'pearson'.
    
    Returns:
    pd.DataFrame: A dataframe with the correlation result, including correlation coefficient and p-value.
    """
    
    if y is not None:
        # Calculate correlation between x and y
        data1 = dataframe[x]
        data2 = dataframe[y]
    elif grouping_variable and group:
        # Calculate correlation within a single group for variable x
        if not isinstance(x, list) or len(x) != 2:
            return "Error: For within-group comparison, x must be a list of two variable names."
        data1 = dataframe[dataframe[grouping_variable] == group][x[0]]
        data2 = dataframe[dataframe[grouping_variable] == group][x[1]]
    else:
        return "Error: Invalid input parameters."
    
    # Choose correlation method
    if correlation_type == 'pearson':
        corr_coef, p_value = stats.pearsonr(data1, data2)
    elif correlation_type == 'spearman':
        corr_coef, p_value = stats.spearmanr(data1, data2)
    elif correlation_type == 'kendall':
        corr_coef, p_value = stats.kendalltau(data1, data2)
    else:
        return "Error: Unknown correlation type specified."
    
    # Organize output in a table
    results_table = pd.DataFrame({
        'Variable 1': [x if isinstance(x, str) else x[0]],
        'Variable 2': [y if y is not None else x[1]],
        'Correlation Coefficient': [corr_coef],
        'P Value': [p_value],
        'Correlation Type': [correlation_type]
    })
    
    return results_table


import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def perform_anova(dataframe, continuous_variable, grouping_variable, groups=None, alpha=0.05, posthoc=False, correlation_type='pearson'):
    """
    Performs one-way ANOVA with options for post-hoc analysis, specifying groups, and calculating effect size.
    
    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the data.
    continuous_variable (str): The name of the continuous variable.
    grouping_variable (str): The name of the categorical grouping variable.
    groups (list, optional): Specific groups within the grouping variable to include in the analysis.
    alpha (float, optional): Significance level for the post-hoc test.
    posthoc (bool, optional): Whether to perform post-hoc analysis.
    correlation_type (str, optional): Type of correlation for effect size calculation.
    
    Returns:
    pd.DataFrame: A dataframe with ANOVA and optionally post-hoc test results, including effect size.
    """
    
    if groups is not None:
        # Filter the dataframe to include only the specified groups
        dataframe = dataframe[dataframe[grouping_variable].isin(groups)]
    
    # Prepare formula
    formula = f'Q("{continuous_variable}") ~ C(Q("{grouping_variable}"))'
    
    # Fit the model
    model = ols(formula, data=dataframe).fit()
    
    # Perform one-way ANOVA
    anova_results = anova_lm(model, typ=2)
    
    # Calculate effect size (Eta Squared)
    eta_squared = anova_results['sum_sq'][0] / (anova_results['sum_sq'][0] + anova_results['sum_sq'][1])
    
    # Extract F-value and P-value
    f_value = anova_results['F'][0]
    p_value = anova_results['PR(>F)'][0]
    
    # Perform post-hoc test if requested
    posthoc_results = None
    if posthoc:
        tukey = pairwise_tukeyhsd(endog=dataframe[continuous_variable], groups=dataframe[grouping_variable], alpha=alpha)
        posthoc_results = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    
    # Organize ANOVA output in a table
    results_table = pd.DataFrame({
        'F Value': [f_value],
        'P Value': [p_value],
        'Effect Size (Eta Squared)': [eta_squared]
    })
    
    if posthoc:
        return results_table, posthoc_results
    else:
        return results_table



def perform_multiple_regression(dataframe, dependent_variable, independent_variables):
    """
    Performs multiple regression and reports the results, including coefficients, p-values, and effect size (R^2).
    
    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the data.
    dependent_variable (str): The name of the dependent variable.
    independent_variables (list): A list of names of the independent variables.
    
    Returns:
    pd.DataFrame: A dataframe with regression results, including coefficients, p-values, and R^2.
    """
    
    # Prepare the data
    X = dataframe[independent_variables]  # Independent variables
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    y = dataframe[dependent_variable]  # Dependent variable
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Get the summary of the regression
    summary = model.summary2().tables[1]  # Coefficients table
    
    # Convert summary to DataFrame for easier manipulation
    results_df = pd.DataFrame(summary)
    
    # Add model R^2 for effect size
    r_squared = model.rsquared
    results_df = results_df.append(pd.DataFrame({'Coef.': [r_squared], 'P>|t|': [np.nan]}, index=['R^2']))
    
    return results_df[['Coef.', 'P>|t|']]


