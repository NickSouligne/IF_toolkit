import pandas as pd
import numpy as np
from scipy.stats import norm


def cond_round_3(table):
    def round_numeric_columns(df):
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].round(3)
        return df
    
    def replace_zeros(df):
        return df.apply(lambda x: x.replace(0, "<0.001") if x.name in df.select_dtypes(include=np.number).columns else x)
    
    return table.pipe(round_numeric_columns).pipe(replace_zeros)

def select_coef(model, lambda_val=None, xvars=None, outcome=None):
    if lambda_val is None:
        l = model['lambda_1se']
    else:
        l = lambda_val
        
    coef_names = model['coef'][0]
    coef_values = model['coef'][1]
    
    coef_df = pd.DataFrame({'coef.name': coef_names, 'coef.value': coef_values})
    coef_df = coef_df[coef_df['coef.value'] != 0]
    coef_vars = coef_df['coef.name'].str.extractall("(" + '|'.join(xvars) + ")").unstack().dropna().stack().tolist()
    
    formula = f"{outcome} ~ {' + '.join(coef_vars)}"
    
    return {'coef': coef_df, 'vars_names': coef_vars, 'formula': formula}

def get_bs_rescaled(bs_table, est_vals, sampsize, m_factor):
    m = sampsize ** m_factor
    sqrt_m = np.sqrt(m)
    est_vals_array = np.array(est_vals)
    rescaled_bs_table = sqrt_m * (bs_table - est_vals_array)
    return rescaled_bs_table

# Calculate normal approximation confidence interval.
#
# @param bs_rescaled Rescaled bootstrap estimate table (result of get_bs_rescaled).
# @param est_named Vector of metrics ('defs' element of'analysis_estimation' function output).
# @param parameter Name of the metric being estimated (string).
# @param sampsize Sample size of estimation data set.
# @param alpha Size of confidence interval.
#
# @returns Dataframe with point estimate, SE, variance, 1-alpha confidence interval.
def ci_norm(bs_rescaled, est_named, parameter, sampsize, alpha):
    #Calculate the variance of the bootstrap estimates
    var_est = np.var(bs_rescaled[parameter], ddof=1)  #ddof=1 for sample variance
    
    #Point estimate
    point_est = est_named[parameter]
    
    #Standard error of the estimate
    se_est = np.sqrt(var_est / sampsize)
    
    #Confidence interval lower and upper bounds
    ci_low = point_est - norm.ppf(1 - alpha / 2) * se_est
    ci_high = point_est + norm.ppf(1 - alpha / 2) * se_est
    
    #Create a DataFrame to hold the results
    result = pd.DataFrame({
        'var_est': [var_est],
        'point_est': [point_est],
        'se_est': [se_est],
        'ci_low': [ci_low],
        'ci_high': [ci_high]
    })
    
    return result


# Calculate t-interval.
#
# @inheritParams ci_norm
# @param m_factor Fractional power for calculating resample size.
#
# @returns Dataframe with point estimate, SE, variance, 1-alpha confidence interval.
def ci_tint(bs_rescaled, est_named, parameter, sampsize, alpha, m_factor):
    #Calculate the normal approximation confidence interval first
    se_table = ci_norm(bs_rescaled, est_named, parameter, sampsize, alpha)
    se_scalar = se_table['se_est'].values[0]
    t_values = np.zeros_like(bs_rescaled[parameter])  #Initialize t_values array
    t_values = bs_rescaled[parameter] / (np.sqrt(sampsize ** m_factor) * se_scalar)
    
    data_temp = pd.DataFrame(bs_rescaled)
    data_temp['t_value'] = t_values.dropna()
    
    #Calculate the t-interval bounds
    low_trans = se_table['point_est'].values[0] - se_table['se_est'].values[0] * np.quantile(data_temp['t_value'], 1 - alpha / 2, interpolation='linear')
    high_trans = se_table['point_est'].values[0] - se_table['se_est'].values[0] * np.quantile(data_temp['t_value'], alpha / 2, interpolation='linear')
    #Create a DataFrame to hold the results
    df = pd.DataFrame({
        'point_est': se_table['point_est'].values,
        'se_est': se_scalar,
        'low_trans': low_trans,
        'high_trans': high_trans
    })

    return df


# Truncate confidence interval.
#
# @param ci_result Confidence interval dataframe (result of ci_norm or ci_tint).
# @param type Type of confidence interval ('norm' or 'tint')
#
# @returns Confidence interval dataframe with truncated interval.
def ci_trunc(ci_result, type2):
    if type2 == 'norm':
        ci_result['ci_low'] = ci_result['ci_low'].apply(lambda x: max(x, 0))
        ci_result['ci_high'] = ci_result['ci_high'].apply(lambda x: min(x, 1))
    elif type2 == 'tint':
        ci_result['low_trans'] = ci_result['low_trans'].apply(lambda x: max(x, 0))
        ci_result['high_trans'] = ci_result['high_trans'].apply(lambda x: min(x, 1))
    
    return ci_result