import pandas as pd
import numpy as np
import logging
import copy

from timelibs import timemodels as tm
from timelibs import timeutils as tu
from sklearn.model_selection import TimeSeriesSplit
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities import dataframe_functions
import tqdm.auto as tqdm
from sklearn import metrics
from multiprocessing import Pool
from datetime import datetime


def preprocess_sd(func):
    def wrapper(self, *args, **kwargs):
        pass
    return wrapper


def postprocess_sd(func):
    """
    Data postprocessing. Reverses smoothing (if used) and makes predictions positive.

    :param func: wrapped function/
    :return: func's result.
    """
    def wrapper(self, *args, **kwargs):
        if self.make_positive:
            func1 = lambda x: 0 if x < 0 else x
        else:
            func1 = lambda x: x

        if self.smoother is not None:
            if (func.__name__ == "get_predictions") and (isinstance(self.smoother, tu.WaveletSmoother)):
                func2 = lambda x: x
            else:
                func2 = lambda x: self.smoother.backward(x)
        else:
            func2 = lambda x: x

        all_preds = [(preds[0], list(map(func1, func2(preds[1])))) for preds in func(self, *args, **kwargs)]
        return all_preds

    return wrapper


class SalesData:
    """
    Sales prediction manager.

    Attributes
    ----------
    :param df: basic data for creating a forecast.
    :type df: pandas.DataFrame

    :param train_featured: train data with synthetic features for supervised methods, optional.
    :type train_featured:  pandas.DataFrame

    :param test_featured: test data with synthetic features for supervised methods, optional.
    :type test_featured: pandas.DataFrame

    :param horizon: prediction length, optional
    :type horizon: int

    :param season: seasonality, define manually, optional
    :type season: int

    :param smoother: special class for smoothing data
    :type smoother: tu.Smoother

    :param make_positive: flag for only-positive predictions
    :type make_positive: bool

    :param models: array of model witch using to create predictions.
    :param models: array of SalesModel
    """

    def __init__(self,
                 df: pd.DataFrame,
                 train_featured: pd.DataFrame = None,
                 test_featured: pd.DataFrame = None,
                 additional_features: pd.DataFrame = None,
                 horizon: int = 6,
                 season: int = 6,
                 models=None,
                 smoother: tu.Smoother = None,
                 make_positive=True):
        if models is None:
            models = []
        self.df = df
        self.horizon = horizon
        self.season = season
        self.make_positive = make_positive
        self._smoother = None
        self.smoother = smoother
        self._set_empty_additional_features()
        self._run_feature_manager(train_featured, test_featured,
                                  additional_features)
        self.models = models

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models):
        models_list = self._create_models(models)
        self._models = models_list

    @property
    def smoother(self):
        return self._smoother

    @smoother.setter
    def smoother(self, smoother):
        """
        Smoother's manager. If data was transform before method does the inverse conversion first

        :param smoother: new smoother to set
        :type smoother: tu.Smoother
        """
        if self.smoother is not None:
            logging.info("The data was transform before. Some information can be lost.")
            logging.warning("The data was transform before. Some information can be lost.")
            orig_df = self.smoother.backward(self.df.iloc[:, 0])
            bias = len(orig_df) - len(self.df.iloc[:, 0])
            orig_df = orig_df[bias:]
        else:
            orig_df = self.df.iloc[:, 0]

        if smoother is not None:
            new_df = smoother.forward(orig_df)
            bias = len(new_df) - len(orig_df)
            self.df.iloc[:, 0] = new_df[bias:]
        else:
            self.df.iloc[:, 0] = orig_df

        self._create_features()
        self._smoother = smoother

    def predict(self,
                models=None,
                optimize=False,
                fast_features=True,
                models_params=None,
                models_params_grid=None):

        """
        Function create and return list of predictions.

        :param models: array of models which replaces existing attribute "_models" and uses for creating forecasts,
        optional.
        :type models: array of str or SalesModel

        :param optimize: flag for using optimal models. Using grid search to find estimator with min MAE if checked,
        optional.
        :type optimize: bool

        :param fast_features: flag for using features created by pack which length equals horizon
        instead of one-by-one creation, optional
        :type fast_features: bool

        :param model_params: array of model's parameters. Parameters uses in the corresponding model. Such models won't
        participate in optimizing grid search, optional
        :type model_params: dict

        :param models_params_grid: parameters grid using for optimizing grid search.
        Function would use standard SalesModel's grid if not set, optional.
        :type models_params_grid: dict

        :return: list of predictions given by all models.
        :rtype: list

        """

        if models_params_grid is None:
            models_params_grid = {}
        if models_params is None:
            models_params = {}
        if models is None:
            models = []
        if (len(models) > 0):
            self.models = models

        if (len(models_params) > 0):
            self._set_model_params(models_params)

        if optimize:
            self._create_optimized_models(models_params, models_params_grid)

        if fast_features:
            [model.fit_predict() for model in self._models]
        else:
            self._create_slow_predictions()

        return self.get_predictions()

    @postprocess_sd
    def get_predictions(self):
        """
        Function returns predictions from all models.

        :return: list of all predictions.
        :rtype: list
        """
        return [(type(model).__name__, model.preds) for model in self._models]

    def set_default_data(self, model):
        """
        Procedure drops model's parameters for default.

        :param model: One model for dropping parameters.
        :type model: SalesModel
        """
        if isinstance(model, tm.SalesForecaster):
            model.set_data(self.df, self.horizon)
        elif isinstance(model, tm.SalesPredictor):
            model.set_data(self.train_featured,
                           self.df.iloc[2:, 0],
                           self.test_featured)

    def add_additional_features(self, data, columns=None):
        """
        Adds new valued columns to train and test features.

        :param data: new data to add.
        :type data: pd.DataFrame|np.ndarray

        :param columns: new column's names, optional
        :type columns: array of str

        .. note:: All added before columns erased from train and test dataset.
        Then new, wided pack of columns adds again.

        """
        if columns is None:
            columns = []
        bias = len(data) - len(self.train_featured) - len(self.test_featured)
        if bias < 0:
            raise ValueError("Input data is too short.")
        self._remove_added_features()
        if isinstance(data, pd.DataFrame):
            self.additional_features = self.additional_features.join(data.iloc[bias:, :])
        else:
            if len(columns) == 0:
                size_c = 1 if data.ndim == 1 else data.shape[1]
                now_time = datetime.now()
                columns = ["c" + str(l) + str(now_time.hour) + str(now_time.minute) + str(now_time.microsecond) for l in
                           np.arange(0, size_c)]
            data = data[bias:]
            self.additional_features = self.additional_features.join(pd.DataFrame(data,
                                                                                  index=self.additional_features.index,
                                                                                  columns=columns))
        slice_train = self.additional_features.iloc[bias:len(self.train_featured) + bias, :]
        slice_train.set_index(self.train_featured.index, inplace=True)
        slice_test = self.additional_features.iloc[- len(self.test_featured):, :]
        slice_test.set_index(self.test_featured.index, inplace=True)

        self.train_featured = pd.concat([self.train_featured, slice_train], axis=1)
        self.test_featured = pd.concat([self.test_featured, slice_test], axis=1)

    def delete_additional_features(self, columns=None):
        """Removes columns from train, test and additional_features datasets.
        Using additional_features columns if param columns is not identified."""
        cols = columns if columns is not None else self.additional_features.columns
        self.train_featured.drop(columns=cols, inplace=True)
        self.test_featured.drop(columns=cols, inplace=True)
        self.additional_features.drop(columns=cols, inplace=True)

    def _run_feature_manager(self, train_featured, test_featured,
                             additional_features):
        """Manager providing joint update of all feature-based attributes"""
        if (train_featured is not None) and (test_featured is not None):
            self.train_featured = train_featured
            self.test_featured = test_featured
        else:
            self._create_features()

        if additional_features is not None:
            self.additional_features = self.additional_features.join(additional_features)

    def _create_features(self, frec='M'):
        """Creates train_featured and test_featured datasets"""
        new_data = pd.DataFrame(None, index=pd.date_range(str(self.df.iloc[-1:].index.date[0]),
                                                          periods=self.horizon + 1,
                                                          freq=frec), columns=[self.df.columns[0]])
        new_data = self.df.copy().append(new_data.iloc[1:, :])
        X, y = get_new_features(new_data, self.horizon + 1, False)
        self.train_featured = X.iloc[:-self.horizon, :]
        self.test_featured = X.iloc[-self.horizon:, :]

    def _set_empty_additional_features(self):
        """Creates empty dataset for additional features"""
        self.additional_features = pd.DataFrame(index=np.append(self.train_featured.index.values,
                                                                self.test_featured.index.values, ))

    def _remove_added_features(self):
        """Removes all added by additional_features columns from train and test data."""
        self.train_featured.drop(columns=self.additional_features.columns, inplace=True)
        self.test_featured.drop(columns=self.additional_features.columns, inplace=True)

    def _create_models(self, models_list):
        """
        Create list of models by it's names or directly models.

        :param models_list:
        :type models_list: list of string|list of SalesModel

        :return: models
        :rtype: list of SalesModel
        """
        forecasters = tm.get_forecasters()
        predictors = tm.get_predictors()
        models = []

        for model in models_list:
            if isinstance(model, str):
                if model in forecasters:
                    models.append(tm.model_fabric(model)(self.df,
                                                         self.horizon,
                                                         create_model=True))
                elif model in predictors:
                    models.append(tm.model_fabric(model)(self.train_featured,
                                                         self.df.iloc[2:, 0],
                                                         self.test_featured,
                                                         create_model=True))
            elif isinstance(model, tm.SalesModel):
                models.append(model)
                self.set_default_data(self, model)

        return models

    def _create_slow_predictions(self):
        """For all SalesPredictors builds step-by-step predictions with step=1"""
        prd_indices = [isinstance(x, tm.SalesPredictor) and x.uses_steps() for x in self._models]
        prd_trains = [self.df.copy() for i in range(sum(prd_indices))]
        array_models = np.array(self._models)
        for i in range(0, self.horizon):
            for idx, model in enumerate(array_models[prd_indices]):
                add_time = pd.date_range(str(prd_trains[idx].iloc[-1:].index.date[0]),
                                         periods=2, freq='M')

                prd_trains[idx].loc[add_time[1]] = None
                X, y = get_new_features(prd_trains[idx], 2, False)
                model.set_data(X.iloc[:-1, :], y.iloc[:-1],
                               X.iloc[-1:, :])
                model.fit_predict()
                prd_trains[idx].iloc[-1:, :] = model.preds[-1:]

        for idx, model in enumerate(array_models[prd_indices]):
            model.preds = prd_trains[idx].iloc[-self.horizon:, 0]
            model.set_data(self.train_featured, self.df.iloc[2:, 0],
                           self.test_featured)
            model.fit()

        [model.fit_predict() for model in array_models[list(map(lambda x: not x, prd_indices))]]
        return self.get_predictions()

    def _create_optimized_models(self, models_params=None, models_params_grid=None):
        """
        For every model search optimal hyperperameters and applies them.

        :param model_params: array of model's parameters. Parameters uses in the corresponding model. Such models won't
        participate in optimizing grid search, optional.
        :type model_params: dict

        :param models_params_grid: parameters grid using for optimizing grid search.
        Function would use standard SalesModel's grid if not set, optional.
        :type models_params_grid: dict
        :return:

        ..notes:: Skip model if it in the models_params. If model have identified params_grid use it.
        """

        if models_params_grid is None:
            models_params_grid = {}
        if models_params is None:
            models_params = {}

        for model in self._models:
            if type(model).__name__ in models_params:
                continue

            if type(model).__name__ in models_params_grid:
                parameters_list = models_params_grid[type(model).__name__]
            else:
                parameters_list = model.get_params_grid()

            if not parameters_list is None:
                gs = timeseries_grid_search(self, model,
                                            parameters_list=parameters_list)
                model.set_parameters(gs[0][0])

    def _set_model_params(self, models_params):
        """Set parameters for Sales model. Lower-lewel model will be re-created."""
        for model in self._models:
            if type(model).__name__ in models_params:
                model.set_parameters(models_params[type(model).__name__])


# Свободные функции

def get_new_features(data, max_timeshift, use_y=True):
    """Creates new, synthetic features with tsfresh lib."""
    df_shift, y = dataframe_functions.make_forecasting_frame(data.Count,
                                                             kind="123", max_timeshift=max_timeshift,
                                                             rolling_direction=1)
    X = extract_features(df_shift, column_id="id", column_sort="time",
                         column_value="value", show_warnings=False,
                         impute_function=dataframe_functions.impute,
                         disable_progressbar=True)
    X = X.loc[:, X.apply(pd.Series.nunique) != 1]
    if use_y:
        X["feature_last_value"] = y.shift(1)

    X = X.iloc[1:, ]
    y = y.iloc[1:]
    return X, y


def serialize_all_estimators(sds, sizes, folder="Estimators"):
    """
    Serialize all models from SalesData unit.

    :param sds: SalesData's array with pretrained models
    :type sds: list of SalesData

    :param sizes: sizes used for slicing data
    :type sizes: list of str

    :param folder: folder to write data in
    :type folder: str

    .. note:: There's no check for model's number. Supposed that the first  record of model's list the best one.
     Also all estimators should be hold in the root folder.

    """
	[tu.dump_data(sd, n_size, folder) for sd, n_size in zip(sds, sizes)]
   

def timeseries_validation(sd,
                          model,
                          n_splits=3):
    """
    Cross-validation for timeseries. Uses mean absolute error.

    :param sd: SalesData object contained tested model.
    :type sd: SalesData

    :param model: tested model.
    :type model: SalesModel

    :param n_splits: number of splits.
    :type n_splits: int

    :return: mean error of all splits
    :rtype: int
    """
    if isinstance(model, tm.SalesForecaster):
        func = lambda m: m.set_data(sd.df.iloc[train_index, :], len(test_index))
        df_train = sd.df
    elif isinstance(model, tm.SalesPredictor):
        func = lambda m: m.set_data(sd.train_featured.iloc[train_index, :],
                                    sd.df.iloc[2:, 0][train_index],
                                    sd.train_featured.iloc[test_index, :])
        df_train = sd.train_featured
    else:
        print('Wrong class')
        return None

    if sd.make_positive:
        func2 = lambda x: 0 if x < 0 else x
    else:
        func2 = lambda x: x

    maes = list()
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=n_splits)
    for train_index, test_index in tscv.split(df_train):
        try:
            func(model)
            preds = model.fit_predict()
            maes.append(metrics.mean_absolute_error(sd.df.iloc[test_index, 0], [func2(pred) for pred in preds]))
        except Exception as e:
            e = None

    sd.set_default_data(model)
    try:
        model.fit_predict()
    except:
        return 10000

    if len(maes)>0:
        return np.mean(maes)
    else:
        return 100000


def timeseries_grid_search(sd,
                           model,
                           parameters_list,
                           n_splits=3, n_items=10,
                           show_progressbar=False,
                           show_errors=False):
    """
    Grid search for time series.

    :param sd: SalesData object contained tested model.
    :type sd: SalesData

    :param model: tested model.
    :type model: SalesModel

    :param parameters_list: list with sets of parameters for testing
    :type parameters_list: list of dict

    :param n_splits: number of splits.
    :type n_splits: int

    :param n_items: number of best parameters to return
    :type n_items: int

    :param show_progressbar: flag for showing progressbar
    :type show_progressbar: bool

    :param show_errors: flag for showing errors
    :type show_errors: bool

    :return: list of best parameters and it's mean errors length n_items
    :rtype: 2-d list
    """
    results = np.empty((0, 2))
    if n_items > len(parameters_list):
        n_items = len(parameters_list)
    if show_progressbar:
        parameters_list = tqdm(parameters_list)
    step_i = 0
    for param in parameters_list:
        step_i += 1
        percent_i = 100*step_i/len(parameters_list)
        if int(percent_i%10)==0:
            print('{}%'.format(int(percent_i)))
        try:
            model.set_parameters(param)
            result = timeseries_validation(sd, model, n_splits)
            results = np.append(results, [[param, result]], axis=0)
        except Exception as e:
            if show_errors:
                print(e)
            e = None

    results = results[np.argsort(results[:, 1])]
    return results[:n_items]


def pool_grid_search(arg):
    """
    Grid searcs for multiprocessing.

    :param arg: list of arguments. Contains SalesData objects with sliced and prepared data.
    List of sizes used for slicing and flag for disabling errors. 
    :type arg: list

    :return: tuple of str and list

    .. note:: All best parameters applied to it's models, so in the end SalesData object will have
    optimal models in it. Additionally function returns information of size, mean error and the best model.
    ForecasterSARIMA uses pre-counted params, no search if possible.

    """
    sd = arg[0]
    n_size = arg[1]
    print(n_size)

    if len(arg) > 2:
        show_errors = arg[2]
    else:
        show_errors = None

    res = []
    if len(sd.df) < 7:
        print("!!!!!!!!!!! less than 7", n_size)
        return (n_size, [None, None, None])

    for model in sd.models:
        if isinstance(model, tm.ForecasterSARIMA):
            try:
                all_params = [tu.load_data(n_size, "params/SARIMA")]
            except:
                all_params = model.get_params_grid()
        else:
            all_params = model.get_params_grid()
        gs_result = timeseries_grid_search(sd, model, all_params,
                                           n_splits=3, n_items=1, show_errors=show_errors)
        try:
            model.set_parameters(gs_result[0, 0])
            res.append([gs_result[0, 1], model, gs_result[0, 0]])
        except Exception as e:
            print(e)
            e = None
    return (n_size, res)


def start_searching(args, slow):
    """
    Creates list of models  with simple or multiprocessed search.

    :param args: list of arguments to the next search
    :param args: list

    :param slow: flag of using multiprocession
    :type slow: bool

    """
    if slow:
        optimized_models = []
        for arg in tqdm(args):
            optimized_models.append(pool_grid_search(arg))
    else:
        num_processors = 6
        with Pool(processes=num_processors) as pool:
            optimized_models = pool.map(pool_grid_search, args)
            optimized_models = list(optimized_models)
    return optimized_models


def create_all_predictions(df, models,
                           horizon=6, season=12,
                           serialize=False,
                           slow=True,
                           extended=False):
    """
    Creates predictions for raw data.

    :param df: raw data for building predictions.
    :type df: pd.DataFrame

    :param models: list of models using in predictions.
    :type models: list of string|list of tm.SalesModel

    :param horizon: prediction's horison
    :type horizon: int

    :param season: basic length of season
    :type season: int

    :param serialize: flag for searializing best models
    :type serialize: bool

    :param slow: flag for using multiprocessing
    :tupe slow: bool

    :param extended: flag for searching the best model with different smooters or length of forecasting step
    :type extended:

    :return: list of SalesData objects with the best model pre-trained by optimal parameters
    :rtype: list of SalesData
    """
    sizes = df.Nomenclature_size.unique()[:]
    sds = [SalesData(tu.get_mean(df, n_size),
                     horizon=horizon,
                     smoother=tu.WaveletSmoother(),
                     season=season,
                     models=models) for n_size in sizes]
    show_errors = [False] * len(sizes)
    args = list(zip(sds, sizes, show_errors))

    def set_smoother(sd, smoother):
        sd.smoother = smoother

    optimized_models = start_searching(args, slow)
    if extended:
        [set_smoother(sd, tu.WaveletSmoother()) for sd in sds]
        optimized_models_sm = start_searching(args, slow)

        for idx, sd in enumerate(sds):
            optimized_models[idx][1].sort(key=lambda x: x[0])
            optimized_models_sm[idx][1].sort(key=lambda x: x[0])
            best_model = None
            if optimized_models[idx][1][0][0] > optimized_models_sm[idx][1][0][0]:
                best_model = [optimized_models_sm[idx][1][0][1]]
            else:
                best_model = [optimized_models[idx][1][0][1]]
                sd.smoother = None
            sd.models = best_model
            sd.predict(fast_features=False)
    else:
        for idx, sd in enumerate(sds):
            sd.models = [min(optimized_models[idx][1], key=lambda x: x[0])[1]]
            sd.predict(fast_features=False)

    if serialize:
        serialize_all_estimators(sds, sizes, folder="Estimators")
    return sds
