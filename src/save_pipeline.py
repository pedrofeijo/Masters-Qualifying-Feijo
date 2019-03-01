from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)

from transforms import *

DATA_FOLDER = '../data/csv'
MODELS_FOLDER = '../models'
PIPELINE_BASE_NAME = 'transformer_pipeline'

DATA = {'FOURIER': 'v000_SCIG_SC_SENSORA_FOURIER_chunk_90', 
        'HOS': 'v000_SCIG_SC_SENSORA_HOS_chunk_90', 
        'SCM': 'v000_SCIG_SC_SENSORA_SCM_chunk_90'}

LABELS = {'FOURIER': 'v000_SCIG_SC_SENSORA_FOURIER_labels_chunk_90', 
          'HOS': 'v000_SCIG_SC_SENSORA_HOS_labels_chunk_90', 
          'SCM': 'v000_SCIG_SC_SENSORA_SCM_labels_chunk_90'}

SEED = 6969


for feature_set in DATA.keys():
    logging.info('Feature set: {}'.format(feature_set))
    
    data = pd.read_csv(os.path.join(DATA_FOLDER, DATA[feature_set]) + '.csv')
    labels = pd.read_csv(os.path.join(DATA_FOLDER, LABELS[feature_set]) + '.csv')

    pipeline = Pipeline([('cleaner', DropNaN()),
                     ('selector', FeatureSelection(extractor=feature_set)),
                     ('scaler', FeatureScaling()),
                    ])
    _ = pipeline.fit_transform(data)

    pipe_name = PIPELINE_BASE_NAME + '_{}.pkl'.format(feature_set)
    joblib.dump(pipeline, os.path.join(MODELS_FOLDER, pipe_name))
    logging.info('Pipeline saved: {}'.format(pipe_name))






































