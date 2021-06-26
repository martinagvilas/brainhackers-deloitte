import numpy as np
import pandas as pd

from  scipy import stats
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_string_dtype

# df_file = "/Users/martina.gonzales/Projects/side_projects/brainhackers-deloitte/datasets/loan.csv"
# df = pd.read_csv(df_file)
# sensitive_feature = ["minority",]

# grouped_dataset = df.set_index(sensitive_feature)
# numerical_dataset = grouped_dataset.select_dtypes(include=["number"])

# def highlight_cols(x):
#     r = "background-color: '#d65f5f'"
#     return r

def compute_t_test(df, sensitive_feature):
    
    significant_features = []
    for feature in df.columns:
        t_val, p_val = stats.ttest_ind(
            df.loc[0][feature].values, 
            df.loc[1][feature].values
        )
        
        print('done')
        if p_val < 0.05:
            significant_features.append(feature)
    
    return significant_features

