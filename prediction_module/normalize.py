from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math


class Norm:

    def __init__(self, df,predicted):
        # nuestra clase va a recibir dos parámetros que son fijos a lo largo de toda la BBDD, el nombre de la BBDD y la contraseña con el servidor.
        self.df = df
        self.predicted = predicted

    def manual(self):
        '''

        :return: return the data frame with the respond variable (predicted) normalized by manual method based on
        formula (x-average)/(max-min)
        '''

        average = self.df[self.predicted].mean()
        maximum = self.df[self.predicted].max()
        minimum = self.df[self.predicted].min()

        self.df[self.predicted + "_NORM_MANUAL"] = (self.df[self.predicted] - average) / (maximum - minimum)

        self.df.drop(columns=self.predicted, inplace=True)

        return self.df, average, maximum, minimum

    def logarithm(self):
        '''

        :return: return the data frame with the respond variable (predicted) normalized by logarithm method
        '''

        self.df[self.predicted + "_NORM_LOG"] = self.df[self.predicted].apply(lambda x: np.log(x) if x != 0 else 0)

        self.df.drop(columns=self.predicted, inplace=True)

        return self.df

    def root_square(self):
        '''

        :return: return the data frame with the respond variable (predicted) normalized by root square method
        '''

        self.df[self.predicted + "_NORM_SQRT"] = self.df[self.predicted].apply(lambda x: math.sqrt(x))

        self.df.drop(columns=self.predicted, inplace=True)

        return self.df

    def min_max_scaler(self):
        '''

        :return: return the data frame with the respond variable (predicted) normalized by Min Max Scaler method
        '''

        # construir el modelo de escalador
        minmax = MinMaxScaler()

        # ajustamos el modelo utilizando nuestro set de datos
        minmax.fit(self.df[[self.predicted]])

        # transformamos los datos
        X_normalized = minmax.transform(self.df[[self.predicted]])

        # lo unimos a nuestro dataframe original
        self.df[self.predicted + "_NORM_MIXMAXSCALER"] = X_normalized

        self.df.drop(columns=self.predicted, inplace=True)

        return self.df, minmax


class DesNorm:

    def __init__(self, df, predicted):
        # nuestra clase va a recibir dos parámetros que son fijos a lo largo de toda la BBDD, el nombre de la BBDD y la contraseña con el servidor.
        self.df = df
        self.predicted = predicted

    def manual(self, average, maximum, minimum):
        '''

        :return: inverse method to return real variable based on manual formual (x-average)/(max-min)
        '''

        self.df[self.predicted] = self.df[self.predicted]*(maximum - minimum) + average

        return self.df

    def logarithm(self):
        '''

        :return: inverse method to return real variable based on logarithm expression (exponential to inverse)
        '''

        self.df[self.predicted] = self.df[self.predicted].apply(lambda x: np.exp(x) if x != 0 else 0)

        return self.df

    def root_square(self):
        '''

        :return: inverse method to return real variable based on root square expression
        '''

        self.df[self.predicted] = self.df[self.predicted].apply(lambda x: x**2)

        return self.df

    def min_max_scaler(self, model_fitted):
        '''

        :param model_fitted: model fitted previously to be able to recover real variable
        :return: inverse method to return real variable based on min max scaler sklearn method
        '''

        # construir el modelo de escalador
        minmax = model_fitted

        # ajustamos el modelo utilizando nuestro set de datos inversos
        X_desnormalized = minmax.inverse_transform(self.df[[self.predicted]])

        # lo unimos a nuestro dataframe original
        self.df[self.predicted] = X_desnormalized

        return self.df