import pandas as pd

from rpy2.robjects import pandas2ri
import rpy2.robjects as ro


class OperaRFunctions:
    """
    Class with several R functions
    """

    @staticmethod
    def define_mixture(method: str):
        """
        Defining opera bridge with R

        :param method: dynamic forecast combination method
        """
        ro.r(
            """
            define_mixture_r <-
              function(model) {
                library(opera)

                opera_model <- mixture(model = model, loss.type = 'square')

                return(opera_model)
              }
            """
        )

        define_mixture_func = ro.globalenv['define_mixture_r']

        opera_model = define_mixture_func(method)

        return opera_model

    @staticmethod
    def get_weights(opera_model: ro.vectors.ListVector):
        """
        Get the weights from an opera model

        :param opera_model rpy2 data structure ro.vectors.ListVector that contains the R object
        for forecast combination
        """
        ro.r(
            """
            get_weights_r <-
              function(opera_model) {
                library(opera)

                return(opera_model$weights)
              }
            """
        )

        get_weights_func = ro.globalenv['get_weights_r']

        pandas2ri.activate()

        weights = get_weights_func(opera_model)

        pandas2ri.deactivate()

        return weights

    @staticmethod
    def update_mixture(opera_model, predictions: pd.DataFrame, trues: pd.Series):
        """
        Update opera model with forecasts and actual values

        :param opera_model rpy2 data structure ro.vectors.ListVector that contains the R object
        for forecast combination
        :param predictions forecasts of each model as pd.DF
        :param trues actual values for computing loss
        """
        ro.r(
            """
            update_mixture_r <-
              function(opera_model, predictions,trues) {
                library(opera)

                for (i in 1:length(trues)) {
                    opera_model <- predict(opera_model, newexperts = predictions[i, ], newY = trues[i])
                }

                return(opera_model)
              }
            """
        )

        update_mixture_func = ro.globalenv['update_mixture_r']

        pandas2ri.activate()

        new_opera_model = update_mixture_func(opera_model, predictions, trues)

        pandas2ri.deactivate()

        return new_opera_model


class Opera:

    def __init__(self, method: str):
        self.method = method
        self.weights = None
        self.mixture = OperaRFunctions.define_mixture(self.method)
        self.model_names = []

    def compute_weights(self, preds: pd.DataFrame, y: pd.Series):
        self.mixture = OperaRFunctions.update_mixture(self.mixture, preds, y)

        self.weights = OperaRFunctions.get_weights(self.mixture)
        self.weights = pd.DataFrame(self.weights, columns=preds.columns)

        return self.weights
