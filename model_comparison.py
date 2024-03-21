#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm

#%% Model comparison
# Model Comparison: To determine if NFL adds unique variance in predicting dementia types beyond what is accounted for by MRI data alone, you can compare the two models using one of the following methods:
# Likelihood Ratio Test (LRT): This test compares the fit of two nested models (i.e., one model is a special case of the other). If adding NFL significantly improves the model fit, the test will show a significant p-value.
# Nagelkerke R^2: This is a pseudo R^2 measure adapted for logistic regression that can provide an indication of the variance explained by the model. Comparing the Nagelkerke R^2 values between the two models can give you insight into whether the addition of NFL contributes to explaining more variance in dementia prediction.
# AIC (Akaike Information Criterion): Lower AIC values indicate a better-fitting model, taking into account the number of predictors. Comparing the AICs of both models can help you assess whether the inclusion of NFL improves model fit enough to justify the additional complexity.
# Interpretation: If the model with NFL demonstrates significantly better fit (via LRT), higher explanatory power (via Nagelkerke R^2), or lower AIC compared to the model with MRI data alone, then you can conclude that NFL adds unique variance in predicting dementia types beyond what is captured by MRI alone.
# This approach will allow you to quantitatively assess the contribution of NFL levels in the context of dementia prediction, complementing MRI data.


#%% Simulate data (replace this with your own data)
np.random.seed(42)
n = 100
mri = np.random.normal(0, 1, n)
nfl = np.random.normal(0, 1, n)
# Simulating a binary outcome based on mri and nfl, adding some noise
dementia = np.random.binomial(1, p=np.clip((0.5 + 0.1 * mri - 0.1 * nfl + 0.05 * mri * nfl) / 2, 0, 1))
df = pd.DataFrame({'mri': mri, 'nfl': nfl, 'dementia': dementia})

#%%
# Model 1: dementia ~ mri
model1 = sm.Logit(df['dementia'], add_constant(df[['mri']])).fit()
# Model 2: dementia ~ mri + nfl
model2 = sm.Logit(df['dementia'], add_constant(df[['mri', 'nfl']])).fit()

#%%
# Model comparison
print("Model 1 Summary:")
print(model1.summary())
print("\nModel 2 Summary:")
print(model2.summary())

# Perform the Likelihood Ratio Test manually
import scipy.stats

def likelihood_ratio_test(model1, model2):
    """
    Perform a Likelihood Ratio Test between two nested models.

    Parameters:
    - model1: The simpler model (with fewer parameters).
    - model2: The more complex model (with more parameters).

    Returns:
    - lr_stat: The test statistic for the LRT.
    - p_value: The p-value for the test.
    """
    lr_stat = 2 * (model2.llf - model1.llf)
    p_value = scipy.stats.chi2.sf(lr_stat, df=model2.df_model - model1.df_model)
    return lr_stat, p_value

# Calculate the LRT statistics
lr_stat, p_value = likelihood_ratio_test(model1, model2)

print("Likelihood Ratio Test Statistic:", lr_stat)
print("P-value:", p_value)


# Nagelkerke R^2 comparison
def nagelkerke_r2(model, df):
    # Calculate the likelihood of the null model (intercept only)
    null_model = sm.Logit(df['dementia'], np.ones(len(df))).fit(disp=0)
    null_ll = null_model.llf
    # Calculate the Nagelkerke R^2
    r2 = 1 - (model.llf / null_ll)
    r2_nagelkerke = r2 / (1 - null_ll/len(df))
    return r2_nagelkerke

print("\nNagelkerke R^2 for Model 1:", nagelkerke_r2(model1, df))
print("Nagelkerke R^2 for Model 2:", nagelkerke_r2(model2, df))

# AIC comparison
print("\nAIC for Model 1:", model1.aic)
print("AIC for Model 2:", model2.aic)

# %%
