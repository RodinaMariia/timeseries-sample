import pandas as pd
import xgboost as xgb
import numpy as np

from abc import ABCMeta, abstractmethod
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tbats import TBATS
from fbprophet import Prophet
from croston import croston
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from catboost import Pool, CatBoostRegressor

from itertools import product

class SalesModel:
    """
    Abstract class for all predicting models.

    :param _data_train: dataset for model's trainings
    :type _data_train: pandas.Dataframe

    :param _model: Object with specific model

    :param make_positive: making negative values equals zero
    :type make_positive: bool

    :param model_params: "kwargs" dictionary with kwargs used in specific model, optional/
    :type model_params: dict
    
    .. note:: '_model' creates by specific child implementations.
    Flag _one_step uses to prevent overfitting in some models.
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_train: pd.DataFrame, model_params=None):
        if model_params is None:
            model_params = {}
        self._data_train = data_train
        self._model = None
        self._model_params = model_params
        self._one_step = True

    @abstractmethod
    def predict(self):
        """
        Create a prediction by _model.
        """
        pass

    @abstractmethod
    def fit(self):
        """Fits model the prediction."""
        pass

    @abstractmethod
    def fit_predict(self):
        """
        Fits model and creates the prediction.

        :return: list of predicted values
        :rtype: list
        """
        pass

    @abstractmethod
    def set_data(self):
        """
        Set child attributes.
        """
        pass

    @abstractmethod
    def set_parameters(self, model_params=None):
        """Set model's parameters"""
        pass

    @abstractmethod
    def get_params_grid(self):
        """
        Function creates dict with kwargs for specific predicting model.
        """
        pass

    @abstractmethod
    def uses_steps(self):
        pass

    @abstractmethod
    def create_model(self):
        pass


class SalesForecaster(SalesModel):
    """
    Upper class for all predicting models using horizon for forecasting.

    :param horizon: prediction length
    :type horizon: int

    :param preds: list of predictions
    :type preds: list
    
    .. notes:: Class defines abstract function "set_data" and creating attribute _model for parent class.
    
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_train, horizon: int = 6, model_params=None):
        super().__init__(data_train, model_params)
        self.horizon = horizon
        self._fit = None
        self.preds = None

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def get_params_grid(self):
        """
        Create parameters grid for fine-tuning.

       :return: dictionary with hyperparameters.
       :rtype: dict
        """
        pass

    def fit_predict(self):
        self.fit()
        return self.predict()

    def set_data(self, data_train, horizon=0):
        """Set basic attributes"""
        self._data_train = data_train
        if horizon > 0:
            self.horizon = horizon
        self.create_model()

    def set_parameters(self, model_params=None):
        if model_params is not None:
            self._model_params = model_params
            self.create_model()

    def _createifchecked_model(self, create_model=False):
        """
        Check special flag and create model if necessary.
        
        Parameters
        ----------
        :param create_model: model creation flag, optional.
        :type create_model: bool


        """
        if create_model:
            self.create_model()
        else:
            self._model = None

    def uses_steps(self):
        return self._one_step


class SalesPredictor(SalesModel):
    """
    Upper class for all predicting models which requires features and labels.
    
    Attributes
    ----------
    :param _label_train: list of right answers for train dataset
    :type _label_train: pd.DataFrame

    :param _data_test: data for creating prediction.
    :type _data_test: pd.DataFrame

    :param _fit: fitted model

    :param preds: list of predictions
    :type preds: list
    
    .. note:: Class defines abstract function "set_data" and creating attribute _model for parent class.
    Attribute _fit contains fitted model.
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_train, label_train: pd.DataFrame, data_test: pd.DataFrame, model_params=None):
        super().__init__(data_train, model_params)
        self._label_train = label_train
        self._data_test = data_test
        self._fit = None
        self.preds = None

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def get_params_grid(self):
        pass

    def fit_predict(self):
        self.fit()
        return self.predict()

    def set_data(self, data_train, label_train, data_test):
        self._data_train = data_train
        self._label_train = label_train
        self._data_test = data_test

    def set_parameters(self, model_params=None):
        if model_params is not None:
            self._model_params = model_params
            self.create_model()

    def _createifchecked_model(self, create_model=False):
        """
        Check spesial flag and create model if necessary.
        
        Parameters
        ----------
        create_model : bool, optional
            Model creation flag.
        """
        if create_model:
            self.create_model()
        else:
            self._model = None

    def uses_steps(self):
        return self._one_step


class ForecasterHolt(SalesForecaster):
    """
    :param _model: Exponential smoothing model
    :type _model: statsmodels.ExponentialSmoothing
    """

    def __init__(self, data_train, horizon, create_model=False, model_params=None):
        super().__init__(data_train, horizon, model_params)
        self._createifchecked_model(create_model)

    def fit(self):
        self._fit = self._model.fit()

    def predict(self):
        self.preds = self._fit.forecast(self.horizon).values
        return self.preds

    def create_model(self):
        self._model = ExponentialSmoothing(self._data_train, **self._model_params)

    def get_params_grid(self):
        trend = ["add", "mul", None]
        seasonal = ["add", "mul", None]
        seasonal_periods = [3, 4, 6, 12]
        parameters_list = list(product(trend, seasonal, seasonal_periods))
        return [dict(zip(["trend", "seasonal", "seasonal_periods"], l)) for l in parameters_list]


class ForecastersCroston(SalesForecaster):
    """
    :param _model: Croston model for sparse values
    :type _model: croston.croston
    """

    def __init__(self, data_train, horizon, create_model=False, model_params=None):
        super().__init__(data_train, horizon, model_params)
        self._createifchecked_model(create_model)

    def set_data(self, data_train, horizon=0):
        self._data_train = data_train
        if horizon > 0:
            self.horizon = horizon
        self.fit()

    def fit(self):
        self._model = croston.fit_croston(self._data_train, self.horizon)
        self._fit = self._model

    def predict(self):
        self.preds = self._fit['croston_forecast'].reshape(self.horizon,)
        return self.preds

    def create_model(self):
        self._model = croston.fit_croston(self._data_train, self.horizon, **self._model_params)

    def get_params_grid(self):
        return [{"croston_variant": ["sba", "sbj", "original"]}]


class ForecasterProphet(SalesForecaster):
    """
    :param _model: Facebook Prophet model
    :type _model: fbprophet.Prophet
    """

    def __init__(self, data_train, horizon, create_model=False, model_params=None):
        super().__init__(data_train, horizon, model_params)
        self._createifchecked_model(create_model=True)

    def fit(self):
        self.create_model()
        new_df = pd.DataFrame()
        new_df['ds'] = self._data_train.index
        new_df['y'] = self._data_train.values
        self._model.fit(new_df)
        self._fit = self._model.make_future_dataframe(periods=self.horizon, freq='M')

    def predict(self):
        self.preds = self._model.predict(self._fit).iloc[-self.horizon:, :].yhat.values
        return self.preds

    def create_model(self=None):
        self._model = Prophet(**self._model_params)

    def get_params_grid(self):
        growth = ['linear', 'logistic']
        seasonality_mode = ['additive', 'multiplicative']
        parameters_list = list(product(growth, seasonality_mode))
        return [dict(zip(["growth", "seasonality_mode"], l)) for l in parameters_list]


class PredictorRidge(SalesPredictor):
    """
    :param _model: Ridge regression.
    :type _model: sklearn.Ridge
    """

    def __init__(self, data_train, label_train, data_test,
                 create_model=False, model_params=None):
        super().__init__(data_train, label_train, data_test, model_params)
        self._one_step = False
        self._createifchecked_model(create_model)

    def fit(self):
        self._fit = self._model.fit(self._data_train, self._label_train)

    def predict(self):
        self.preds = self._model.predict(self._data_test)
        return self.preds

    def create_model(self):
        self._model = Ridge(**self._model_params)


    def get_params_grid(self):
        solver = ["svd", "cholesky", "lsqr", "sparse_cg"]
        fit_intercept = [True, False]
        normalize = [True, False]
        alpha = [30]
        parameters_list = list(product(solver, fit_intercept, normalize, alpha))
        return [dict(zip(["solver", "fit_intercept", "normalize", "alpha"], l)) for l in parameters_list]


class PredictorCatBoost(SalesPredictor):
    """
    :param _model: Yandex CatBoost model.
    :type _model: catboost.CatBoostRegressor
    """

    def __init__(self, data_train, label_train, data_test,
                 create_model=False, model_params=None):
        super().__init__(data_train, label_train, data_test, model_params)
        self._createifchecked_model(create_model=True)

    def fit(self):
        self._fit = self._model.fit(Pool(self._data_train, self._label_train))

    def predict(self):
        self.preds = self._model.predict(Pool(self._data_test))
        return self.preds

    def create_model(self):
        self._model_params["logging_level"] = "Silent"
        self._model = CatBoostRegressor(**self._model_params)

    def get_params_grid(self):
        bootstrap_type = ["Bayesian", "Bernoulli", "MVS"]
        depth = np.arange(4, 16, 4)
        iterations = np.arange(100, 500, 100)
        grow_policy = ["SymmetricTree", "Depthwise", "Lossguide"]
        parameters_list = list(product(bootstrap_type, depth, iterations,
                                       grow_policy
                                       ))
        return [dict(zip(["bootstrap_type", "depth", "iterations",
                          "grow_policy"
                          ], l)) for l in parameters_list]


def get_forecasters():
    """Function returning array of SalesForecaster's children."""
    return ['ForecasterHolt', 'ForecasterProphet', 'ForecastersCroston']


def get_predictors():
    """Function returning array of SalesPredictor's children."""
    return ['PredictorRidge', 'PredictorCatBoost']


def model_fabric(model_name):
    """Simple fabric which creates model by it's classname."""
    return globals()[model_name]


