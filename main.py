
import pandas as pd
import numpy as np

from tqdm.auto import tqdm as tqdm
from sklearn import metrics

import logging
import warnings
import os
from timelibs import timedata as td
from timelibs import timeutils as tu
from timelibs import timemodels as tm

import ast
import pickle

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    warnings.filterwarnings('ignore')
    warnings.simplefilter(action='ignore', category=FutureWarning)

    df = pd.read_csv(r'c:\Users\rodina\Documents\Work\Data science\Расчёты\sales.csv',
                     ';', parse_dates=['Date'], dayfirst=True)
    df = df.fillna(0)




    tmp = td.create_all_predictions(df, ['ForecasterHolt', 'ForecasterSARIMA', 'ForecasterProphet',
                                         'ForecastersCroston', 'PredictorRidge', 'PredictorCatBoost'],
                                    slow=False, serialize=True, extended=True,
                                    horizon=3)
    print('Finish')
    for sd in tmp:
        print(sd.get_predictions())
