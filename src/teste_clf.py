from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils import *

DATA_FOLDER = '../data/csv'
MODELS_FOLDER = '../models'
PIPELINE_BASE_NAME = 'transformer_pipeline'
CLASSIFIER_BASE_NAME = 'clf'

DATA = {'FOURIER': 'v000_SCIG_SC_SENSORA_FOURIER_chunk_10', 
        'HOS': 'v000_SCIG_SC_SENSORA_HOS_chunk_10', 
        'SCM': 'v000_SCIG_SC_SENSORA_SCM_chunk_10'}

LABELS = {'FOURIER': 'v000_SCIG_SC_SENSORA_FOURIER_labels_chunk_10', 
          'HOS': 'v000_SCIG_SC_SENSORA_HOS_labels_chunk_10', 
          'SCM': 'v000_SCIG_SC_SENSORA_SCM_labels_chunk_10'}

SEED = 6969

CLASSIFIERS = {'mlp', 'svm', 'knn','naive_bayes'}



def print_metrics(y_true, y_hat):
   conf_mat = confusion_matrix(y_true, y_hat)
   acc = accuracy_score(y_true, y_hat)

   print('Confusion Matrix:')
   print(percentage_confusion_matrix(conf_mat))
   print('Accuracy: {}'.format(acc))

   pass   

for feature_set in DATA.keys():
   # logging.info('Feature set: {}'.format(feature_set))
   data = pd.read_csv(os.path.join(DATA_FOLDER, DATA[feature_set]) + '.csv')
   labels = pd.read_csv(os.path.join(DATA_FOLDER, LABELS[feature_set]) + '.csv')
   pipeline = joblib.load(os.path.join(MODELS_FOLDER, PIPELINE_BASE_NAME + '_{}.pkl'.format(feature_set)))

   features = pipeline.transform(data)
   logging.info(features.head())

   X_test = features.values
   y_test = labels.values.reshape(-1)

   for classifier in CLASSIFIERS:
      logging.info('classifier: {}'.format(classifier))
      clf_name = CLASSIFIER_BASE_NAME + '_{}_{}.pkl'.format(feature_set, classifier)
      logging.info('Loading classifier: {}'.format(clf_name))
      clf = joblib.load(os.path.join(MODELS_FOLDER, clf_name))
      logging.info('Classifiers loaded')

      logging.info('Predicting with classifier: {}'.format(classifier))
      y_hat = clf.predict(X_test)

      logging.info('Calculating metrics... {}'.format(classifier))
      print_metrics(y_test, y_hat)
               