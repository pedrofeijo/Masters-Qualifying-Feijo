import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

'''
The transformations are applied in the dataframe in the following order:
1 - Data cleaning: remove NaN and split into categorical and numerical features
2 - Remove unwanted features in numerical features
3 - Remove unwanted features in categorical features
4 - Feature scaling in numerical features (This one is created by using StandardScaler from sklearn)
5 - Concat two features set
6 - Feature selction (Optional)
7 - Drop NaN that might appear from the scaling tranformation
'''

SELECTED_FEATURES = {'FOURIER': ['fx0d5_R', 'fx1d5_R', 'fx2d5_R', 'fx3_R', 'fx5_R', 'fx7_R', 'Freq_Gen', 'CC_bus'],
                    'HOS': ['Skewness_R', 'Kurtosis_R', 'Variance_R', 'RMS_R', 'Freq_Gen', 'CC_bus'],
                    'SCM': ['scm_COR_R', 'scm_IDM_R', 'scm_ENT_R', 'scm_CSD_R', 'scm_CSR_R', 'Freq_Gen', 'CC_bus']}

PAYMENT_TYPE = ['BOLETO WEB', 'DEBITO AUTOMATICO', 'IN APP PURCHASE', 'CARTAO DE CREDITO']


class DataCleaning(BaseEstimator, TransformerMixin):
    ''' 
    Clean data according the procedures studied in the notebook analyses-02. In short: 
    (i) Drops Nan; (ii) split data in categorical and numerical features; 
    (iii) 1-hot-enconding of categorical features; (iv) Get a unique categorical features
    of an user in a period of 16 weeks; (v)  Get a unique numerical features of an user 
    in a period of 16 weeks; (vi) Average the numerical features in a period of 16-week;
    (vii) contat both feature set;
    -----
    Methods
    ------------------------
    > fit(df)
    Parameters:
    df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - df: dataframe of the dataset, in which the user name must be set as index; 
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''
    
    def fit(self, df):
        return self

    def transform(self, df):
        # Remove NaN:
        df_clean = df.dropna(how='any', inplace=False)
        return df_clean

class RemoveFeatures(BaseEstimator, TransformerMixin):
    ''' 
    Remove unwanted features from the dataframes;
    -----
    Initialized parameters:
    - features: str or list cointaining the field that ought to be removed. Default: 'week'.

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def __init__(self, features='week'):
        self.features = features

    def fit(self, df):
        return self

    def transform(self, df):
        
        return {'numerical': df['numerical'].drop(columns=self.features),
                'categorical': df['categorical'].drop(columns=self.features)}

class FeatureScaling(BaseEstimator, TransformerMixin):
    ''' 
    Scale features by standardization;
    -----
    Initialized parameters:
    - type: str cointaining the scaling method. Default: 'std'.
        - 'std': StandardScaler()

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    self

    Atrributes:
    self._scaler: saved object that sould be used along with the trained model.
    
    > transform(df)

    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def __init__(self, type='std'):
        self.type = type

    def fit(self, df):
        self._scaler = StandardScaler().fit(df)
        return self

    def transform(self, df):
        if self.type == 'std':
            df_std = self._scaler.transform(df)
            df_std = pd.DataFrame(data=df_std, 
                                  columns=df.columns, 
                                  index=df.index)
        
        return df_std

class MergeFeatures(TransformerMixin):
    ''' 
    Concat the numerical and categorical dataframes into a single one.
    -----

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - dict: a dictonary variable of dataframes: {'numerical': DataFrame, 'categorical': DataFrame}
    -----
    Returns:
    - dataframe: a daframe with both feature set.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def fit(self, df):
        return self

    def transform(self, df):
        return pd.concat([df['numerical'], df['categorical']], axis=1)

class DropNaN(TransformerMixin):
    ''' 
    Drop any row from the dataframe that contains a NaN.
    -----

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - df: a dataframe
    -----
    Returns:
    self
    
    > transform(df)

    Parameters:
    - df: a dataframe
    -----
    Returns:
    - dataframe: a daframe withou NaN.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def fit(self, df):
        return self

    def transform(self, df):
        return df.dropna()

class FeatureSelection(TransformerMixin):
    ''' 
    Select the relevant features.
    -----
    Initialized parameters:
    - features: str or list of str containing the fields the should be kept

    Atrributes:
    self.features: feature names.

    Methods
    ------------------------
    > fit(df)
    Parameters:
    - df: a dataframe.
    -----
    Returns:
    self

    > transform(df)

    Parameters:
    - df: a dataframe.
    -----
    Returns:
    - df: a dataframe.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''
    def __init__(self, extractor='FOURIER', features=None):
        if not features:
            self.features = SELECTED_FEATURES[extractor]
        self.extractor = extractor

    def fit(self, df):
        return self

    def transform(self, df):
        return df[self.features]

class GetLables(TransformerMixin):
    ''' 
    Get the labels following the user index in the feature dataframe.
    -----
    Methods
    ------------------------
    > fit(df_user, df_features)
    Parameters:
    - df_user: dataframe containing the user's data
    - df_features: dataframe the outta be used as the feature set. It MUST contain
    the user's name as index.
    -----
    Returns:
    self

    > transform(df_user, df_features)
    
    Parameters:
    - df_user: dataframe containing the user's data
    - df_features: dataframe the outta be used as the feature set. It MUST contain
    the user's name as index.
    -----
    Returns:
    - df: a dataframe.
    -----------------
    OBS.: fit_transform method is available, inherited from TransformerMixin class.
    '''

    def fit(self, df):
        return self

    def transform(self, df):
        return df[df_features['Class']]