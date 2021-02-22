import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn import metrics
from statsmodels import api as sm
from scipy.stats import boxcox
import random
import pickle
import pywt
import copy
import os
import timelibs.timemodels as tm
from abc import ABCMeta, abstractmethod

class Smoother:
    """
    Abstract class for decreasing dispersion methods.
    Contains methods for transformation and inversed transformation.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class WaveletSmoother(Smoother):
    """
    Wavelet

    :param wavelet: name of wavelet's family.
    :type wavelet: str

    :param thresh: noise's threshold
    :type thresh: float

    :param coeff: coefficients to inverse transformation
    :type coeff: list
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._wavelet = kwargs.get("wavelet", 'db4')
        self._thresh = kwargs.get("thresh", 0.63)
        self._coeff = None

    def forward(self, data):
        thresh = self._thresh * np.nanmax(data)
        coeff = pywt.wavedec(data, self._wavelet, mode="per")
        self._coeff = copy.deepcopy(coeff)
        coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
        return pywt.waverec(coeff, self._wavelet, mode="per")

    def backward(self, data):
        return pywt.waverec(self._coeff, self._wavelet, mode="per")


class BoxCoxSmoother(Smoother):
    """
    BoxCox

    :param bias: bias to avoid zero data
    :type bias: float

    :param lmbda: transformation's coefficient
    :type lmbda: float
    """
    def __init__(self,  **kwargs):
        super().__init__()
        self._bias = kwargs.get("bias", 1)

    def forward(self, data):
        new_data, self._lmbda = boxcox(np.array(data) + self._bias)
        return new_data

    def backward(self, data):
        if self._lmbda == 0:
            y = np.exp(data)
        else:
            y = np.exp(np.log(self._lmbda * data + 1) / self._lmbda)
        return np.array(y) - self._bias


class NormalizeSmoother(Smoother):

    """Simple normalisation

    :param mean: mean of dataset
    :type mean: float

    :param std: standard deviation of dataset
    :type std: float

    """
    def __init__(self, mean, std):
        super().__init__()
        self._mean = mean
        self._std = std

    def forward(self, data):
        return list(map(lambda x: (x-self._mean)/self._std, data))

    def backward(self, data):
        return list(map(lambda x: x*self._std + self._mean, data))


def plot_resid(result, title):
    """
    Plotting resids.
    
    :param result: forecaster contained resids in it.

    :param title: Plot's title.
    :type title: str

    """
    plt.plot(result.resid.values, alpha=0.7,
             label='variance={:.3f}'.format(np.std(result.resid.values)));
    plt.hlines(0, xmin=0, xmax=350, color='r')
    plt.title(title)
    plt.legend()
    plt.show()


def visualisation_Dickey_Fuller_Test(df):
    """
    Visualisation of Dickey-Fuller test

    :param df: dataframe for checking stationarity.
    :type df: pd.DataFrame

    """
    print(" > Is the data stationary ?")
    dftest = sm.tsa.adfuller(df.iloc[:, 0], autolag='AIC')
    print("Test statistic = {:.3f}".format(dftest[0]))
    print("P-value = {:.3f}".format(dftest[1]))
    print("Critical values :")
    for k, v in dftest[4].items():
        print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v < dftest[0] else "",
                                                                                100 - int(k[:-1])))


def visualisation_KPSS(df):
    """
    Visualisation of KPSS test
    
    :param df: Dataframe for checking stationarity.
    :type df: pd.DataFrame

    """
    statistic, p_value, n_lags, critical_values = sm.tsa.stattools.kpss(df.iloc[:, 0])
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
        print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')


def visualisation_Jarque_Bera(df):
    """
    Visualisation of Jarque-Bera test
    
    :param df: Dataframe for checking stationarity.
    :type df: pd.DataFrame

    """
    value, p_value = scs.jarque_bera(df)
    print("value:{}".format(value))
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}normal')
    df.hist()


def visualisation_q_test(model):
    """Visualisation for q-q test"""
    q_test = sm.tsa.stattools.acf(model.resid, qstat=True)
    print(pd.DataFrame({'Q-stat': q_test[1], 'p-value': q_test[2]}))


def visualisation_corr(df, lags=20):
    """
    Plots ACF and PACF.
    
    :param df: Dataframe used for calculating correlation and autocorrelation.
    :type df: pd.DataFrame

    :param lags: Number of lags.
    :type lags: int

    """
    plot_acf(df, lags=lags)
    plot_pacf(df, lags=lags)


def visualisation_rmse(all_rmse):
    """
    Print specified metrics for list of values. Using RMSE.
    
    :param all_rmse: Array of tuples with two values to count rmse. First tuple value contains array of test labels,
        second contains forecasted values.
    :type all_rmse: list of tuples

    """
    for name, tw in all_rmse.items():
        print(name, ': {}'.format(np.sqrt(metrics.mean_squared_error(tw[0], tw[1]))))


def visualisation_plots(x, all_y):
    """
    Plot some forecasts with real Count-values(labels).

    :param x: DataFrame with real values (labels)/
    :type x: pd.Dataframe.

    :param all_y: Array of tuples. Second element of tuple contains array with forecasting.Array of tuples with two values to count rmse.
    First tuple value contains array of test labels, second contains forecasted values.
    :type all_y: list of tuple

    .. Note: Using random colors and line-types. Real data always black and solid.

    """

    color_chart = ['aqua', 'aquamarine', 'blue', 'blueviolet', 'brown', 'burlywood',
                   'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan',
                   'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
                   'darkmagenta', 'darkolivegreen', 'darkorange', 'darkred', 'darksalmon',
                   'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet',
                   'deeppink', 'deepskyblue', 'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia',
                   'gainsboro', 'gold', 'goldenrod', 'green', 'greenyellow', 'grey', 'hotpink', 'indigo',
                   'khaki', 'lawngreen', 'lightgray', 'lightgreen', 'lightgrey', 'lightsalmon', 'lightseagreen',
                   'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lime', 'limegreen', 'magenta',
                   'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
                   'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
                   'moccasin',
                   'navajowhite', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod',
                   'paleturquoise', 'palevioletred', 'peachpuff', 'peru', 'plum', 'purple', 'rebeccapurple', 'red',
                   'rosybrown',
                   'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'slateblue',
                   'slategray', 'slategrey', 'springgreen', 'steelblue', 'tan', 'teal', 'tomato', 'turquoise', 'violet',
                   'wheat', 'yellow', 'yellowgreen']
    lines_chart = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']

    fig, ax = plt.subplots(figsize=(20, 10))
    for name, y in all_y.items():
        one_color = random.choice(color_chart)
        print(one_color)
        ax.plot(x.index, y[1], color=one_color,
                linestyle=random.choice(lines_chart), label=name)

    ax.plot(x, color='black', label='fact')
    ax.legend()
    fig.show()


def smooth_negative(data):
    """
    Data preprocessing. Transfer negative data to nearby entry.

    :param data: preprocessing data with target column named Count.
    :type data: pd.DataFrame

    .. Note:: It happends when saled product returns next month. So previous month surely equal or greater then
    this negative value.

    """
    df_neg = data[data.Count < 0]
    for i in range(0, len(df_neg)):
        tmp_df = data[(data.Nomenclature_size == df_neg.iloc[i].Nomenclature_size) & (data.index < df_neg.index[i])]
        if len(tmp_df) > 0:
            data.at[tmp_df.index[-1], 'Count'] = tmp_df.Count.iloc[-1] + df_neg.Count.iloc[i]
            data.at[df_neg.index[i], 'Count'] = 0


def fill_series(data, end_date=None, value=0):
    """
    Data preprocessing. Fill blank month with given values.

    :param data: preprocessing data.
    :type data: pd.DataFrame

    :param end_date: right bound for data range.
    :type end_date: str|date

    :param value: values to fill.
    :type value: float

    """
    if end_date == None:
        end_date = data.index[-1]
    data = data.sort_values('Date')
    time_range = pd.date_range(data.index[0], end_date, freq='M')
    clean_data = pd.DataFrame({'Date': time_range})
    data = clean_data.merge(data, on='Date', how='outer').fillna(value)


def create_date_index(data):
    """
    Data preprocessing. Set DatetimeIndex.

    :param data: preprocessing data.
    :type data: pd.DataFrame

    """
    data.Date = pd.to_datetime(data.Date.dt.date)
    data.set_index('Date', inplace=True)
    data = data.sort_index()


def get_mean(data, Nomenclature_size):
    """
    Data preprocessing. Slice data by special sizes, preprocess it and union measurements by month.

    :param data: preprocessing data.
    :type data: pd.DataFrame

    :param Nomenclature_size: Specific information for slicing data.
    :type Nomenclature_size: str

    :return: preprocessed data ready for forecasting
    :rtype: pd.DataFrame
    """
    smooth_negative(data)
    UL_index = data['Nomenclature_size'] == Nomenclature_size
    UL = data[UL_index]
    UL = UL.drop('Nomenclature_size', axis=1)
    UL = UL[UL['Count'] > 0]
    create_date_index(UL)

    UL = UL.resample('M').sum().dropna()
    fill_series(UL, 0)
    return UL


def dump_data(data, name: str, folder='Estimators'):
    """
    Save model to specified folder.

    :param data: saving to file model.
    :type data: any

    :param name: file's name.
    :type name: str

    :param folder: folder's name
    :type folder: str

    """
    with open(os.path.join(os.path.dirname(__file__), 'models/', folder, name + '.pickle'), 'wb') as f:
        pickle.dump(data, f)


def load_data(name: str, folder='Estimators'):
    """Loads model or other data from .pickle file"""
    try:
        with open(os.path.join(os.path.dirname(__file__), 'models/', folder, name + '.pickle'), 'rb') as f:
            return pickle.load(f)
    except:
        return None
