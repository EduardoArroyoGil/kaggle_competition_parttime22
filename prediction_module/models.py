from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import ensemble
import xgboost as xgb

# Cluster Modeling
# ==============================================================================
from sklearn.cluster import KMeans, DBSCAN
from yellowbrick.cluster import KElbowVisualizer

# Graphs
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d

import pandas as pd
import numpy as np
import os
import math
import logging


class Supervised:

    def __init__(self, df):
        # nuestra clase va a recibir dos parámetros que son fijos a lo largo de toda la BBDD, el nombre de la BBDD y la contraseña con el servidor.
        self.df = df

    def separate_set(self):
        '''

        LAST COLUMN IS THE PREDICTED

        :return: return the data set separated by X (predictors) and Y (predictives) and train - test data sets
        '''

        df = self.df
        predicted = df.columns[-1]

        X = df.drop(predicted, axis=1)
        y = df[predicted]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=42)

        return X_train, X_test, y_train, y_test

    def metrics(self, y_test, y_train, y_test_pred, y_train_pred, model_type):

        '''
        :param y_test: respond variable known by the test data set
        :param y_train: respond variable known by the train data set
        :param y_test_pred: respond variable predicted by the test data set
        :param y_train_pred: respond variable predicted by the train data set
        :param model_type:the prediction_module model type is used (string)
        :return:this function returns the dataframe with all the errors (MAE - Mean Absolut Error
        , MSE - Mean Square Error and RMSE - root mean square error) and the R2 for test and
        train data sets
        '''

        results = {'MAE': [metrics.mean_absolute_error(y_test, y_test_pred),
                              metrics.mean_absolute_error(y_train, y_train_pred)],
                      'MSE': [metrics.mean_squared_error(y_test, y_test_pred),
                              metrics.mean_squared_error(y_train, y_train_pred)],
                      'RMSE': [np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)),
                               np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))],
                      'R2': [metrics.r2_score(y_test, y_test_pred), metrics.r2_score(y_train, y_train_pred)],
                      "set": ["test", "train"]}
        df = pd.DataFrame(results)
        df["model"] = model_type
        return df

    def linear(self):
        '''

        :return: this function returns the linear regression model fitted for the data frame given as input, as well as,
         the importances and data frame with the errors and R2
        '''

        X_train, X_test, y_train, y_test = self.separate_set()

        lr = LinearRegression()

        lr.fit(X_train, y_train)

        # hacemos las predicciones sobre los dos set de datos el X_test y el X_train
        y_pred_test = lr.predict(X_test)
        y_pred_train = lr.predict(X_train)

        lr_results = self.metrics(y_test, y_train, y_pred_test, y_pred_train, 'lr')

        return lr, lr_results

    def decission_tree(self):
        '''

        :return: this function returns the decision tree model fitted for the data frame given as input
        fitted with the grindsearch sklearn method, as well as, the importances and data frame with the errors and
        R2
        '''

        X_train, X_test, y_train, y_test = self.separate_set()

        # fitting first decision tree
        # create a regressor object
        regressor = DecisionTreeRegressor(random_state=0)

        # fit the regressor with X and Y data
        regressor.fit(X_train, y_train)

        #  just to take the max depths to gridsearch
        max_depth_first_model = regressor.tree_.max_depth
        max_depth = np.sqrt(max_depth_first_model)
        depth = 0
        max_depth_list = list()
        for i in range(0, math.ceil(max_depth)):
            depth += 1
            max_depth_list.append(depth)

        #  just to take the max features to gridsearch
        max_features = np.sqrt(len(X_train.columns))
        feature = 0
        max_features_list = list()
        for i in range(0, math.ceil(max_features)):
            feature += 1
            max_features_list.append(feature)



        # GridSearch generation

        # definimos un diccionario con los hiperparámetros que queremos testear.
        param = {"max_depth": max_depth_list,
                 "min_samples_split": [10, 50, 100],
                 "max_features": max_features_list}

        max_cpu = os.cpu_count()

        gs = GridSearchCV(
            estimator=DecisionTreeRegressor(),
            param_grid=param,
            cv=10,
            verbose=max_cpu,
            return_train_score=True,
            scoring="neg_mean_squared_error")

        gs.fit(X_train, y_train)

        best_tree = gs.best_estimator_

        y_pred_test = best_tree.predict(X_test)
        y_pred_train = best_tree.predict(X_train)

        # Results of the accuracy of Decision tree adjust by greedsearch
        dt_results = self.metrics(y_test, y_train, y_pred_test, y_pred_train, 'dt')

        predictors_importance = pd.DataFrame(
            {'predictor': X_train.columns,
             'importance': best_tree.feature_importances_}
        ).sort_values(ascending=False, by="importance", inplace=True)

        return best_tree, dt_results, predictors_importance

    def random_forest(self):
        '''

        :return: this function returns the random forest model fitted for the data frame given as input
        fitted with the grindsearch sklearn method, as well as, the importances and data frame with the errors and
        R2
        '''

        X_train, X_test, y_train, y_test = self.separate_set()

        # fitting first decision tree
        # create a regressor object
        regressor = DecisionTreeRegressor(random_state=0)

        # fit the regressor with X and Y data
        regressor.fit(X_train, y_train)

        #  just to take the max depths to gridsearch
        max_depth_first_model = regressor.tree_.max_depth
        max_depth = np.sqrt(max_depth_first_model)
        depth = 0
        max_depth_list = list()
        for i in range(0, math.ceil(max_depth)):
            depth += 1
            max_depth_list.append(depth)

        #  just to take the max features to gridsearch
        max_features = np.sqrt(len(X_train.columns))
        feature = 0
        max_features_list = list()
        for i in range(0, math.ceil(max_features)):
            feature += 1
            max_features_list.append(feature)



        # GridSearch generation

        # definimos un diccionario con los hiperparámetros que queremos testear.
        param = {"max_depth": max_depth_list,
                 "min_samples_split": [10, 50, 100],
                 "max_features": max_features_list}

        max_cpu = os.cpu_count()

        gs = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=param,
            cv=10,
            verbose=max_cpu,
            return_train_score=True,
            scoring="neg_mean_squared_error")

        gs.fit(X_train, y_train)

        bos = gs.best_estimator_

        y_pred_test = bos.predict(X_test)
        y_pred_train = bos.predict(X_train)

        # Results of the accuracy of Decision tree adjust by greedsearch
        rf_results = self.metrics(y_test, y_train, y_pred_test, y_pred_train, "rf")

        predictors_importance = pd.DataFrame(
            {'predictor': X_train.columns,
             'importance': bos.feature_importances_}
        ).sort_values(ascending=False, by="importance", inplace=True)

        return bos, rf_results, predictors_importance

    def gboostreg(self):

        X_train, X_test, y_train, y_test = self.separate_set()

        gbr = ensemble.GradientBoostingRegressor()
        gbr.fit(X_train, y_train)

        # hacemos las predicciones sobre los dos set de datos el X_test y el X_train
        y_pred_test = gbr.predict(X_test)
        y_pred_train = gbr.predict(X_train)

        gbr_results = self.metrics(y_test, y_train, y_pred_test, y_pred_train, 'gbr')

        return gbr, gbr_results

    def xgboostreg(self):

        X_train, X_test, y_train, y_test = self.separate_set()

        xgbr = xgb.XGBRegressor(objective="reg:linear", random_state=42)

        xgbr.fit(X_train, y_train)

        # hacemos las predicciones sobre los dos set de datos el X_test y el X_train
        y_pred_test = xgbr.predict(X_test)
        y_pred_train = xgbr.predict(X_train)

        xgbr_results = self.metrics(y_test, y_train, y_pred_test, y_pred_train, 'xgbr')

        return xgbr, xgbr_results

    def all_models(self):
        '''

        :return: this function returns all the models under the supervised class fitted for the df give as input
        as well as the dataframe with the errors and R2 results for each model contained in a dictionary dict_models
        '''

        dict_models = {}
        results = pd.DataFrame()

        models_disp = {
            # 'lr': self.linear(),
            # 'dt': self.decission_tree(),
            # 'rf': self.random_forest(),
            # 'gbr': self.gboostreg(),
            'xgbr': self.xgboostreg(),
        }

        model_names = {
            # 'lr': 'Linear Regression',
            # 'dt': 'Decision Tree',
            # 'rf': 'Random Forest',
            # 'gbr': 'Gradient Boost Regression',
            'xgbr': 'Extreme Gradient Boost Regression',
        }

        for key, value in models_disp.items():
            model_alias = key
            model_fitted = value
            model_name = model_names[model_alias]
            try:
                model, model_results = model_fitted
                logging.info(f'Prediction {model_name} algorithm finished')
                dict_models[model_alias] = model
                results = results.append(model_results, ignore_index=True)
            except Exception as e:
                logging.info(f"WARNING: {model_name} algorithm failed with error: {e}")
                dict_models[model_alias] = f"{model_name} algorithm failed with error: {e}"

        return dict_models, results


class Unsupervised:

    def __init__(self, df):
        # nuestra clase va a recibir dos parámetros que son fijos a lo largo de toda la BBDD, el nombre de la BBDD y la contraseña con el servidor.
        self.df = df

    def inertia_silhouette_analysis(self):
        '''

        :return: return the results of the analysis by clusters of the intertia and silhouette with a dataframe for
        intertia, and a graph for intertia and another one for silhouette
        '''
        df = self.df

        # INERTIA ANALYSIS
        # nos creamos una diccionario vacía para ir almacenando los valores de los scores del modelo
        inertia = {}

        # nos creamos una variable, que nos servirá para simular distintos números de clusters
        clusters = range(1, 11)

        # iniciamos el for para crear distintos modelos de cluster para sacar los scores de cada uno
        for i in clusters:
            kmeans = KMeans(n_clusters=i, random_state=0)
            kmeans.fit(df)

            inertia[i] = kmeans.inertia_

        inertia_results = pd.DataFrame(inertia, index=[0]).T.reset_index()
        inertia_results.columns = ["num_clus", "inertia"]

        # plotemoas los resultados
        sns.lineplot(data=inertia_results, x="num_clus", y="inertia")

        # cambiamos las etiquetas de los ejes
        plt.xlabel('Clusters')
        plt.ylabel('Inertia')

        graph_inertia = plt

        # SILHOUETTE ANALYSIS
        # iniciamos un modelo de Kmean
        model = KMeans()

        # llamamos al método KElbowVisualizer para que nos calcula el score de silhouette
        visualizer = KElbowVisualizer(model, k=(2, 15), metric='silhouette')

        # fiteamos el modelo
        visualizer.fit(df)

        graph_silhouette = visualizer

        return inertia_results, graph_inertia, graph_silhouette

    def clustering_knn(self, num_clusters):
        '''

        :param num_clusters:number of clusters data frame want to be splitted
        :return: return the data frame flagged by cluster and the knn clustering model fitted
        '''

        df = self.df

        # iniciamos el modelo
        kmeans = KMeans(n_clusters=num_clusters)

        # fitemos el modelo
        km_fit = kmeans.fit(df)

        # los clusters que se han generado
        labels = km_fit.labels_

        df["Cluster"] = labels

        return df, km_fit

    def separate_df(self, df_clustered):
        '''

        :return: it returns a dictionary separated by each cluster with the structure {cluster_name : dataframe_cluster}
        '''

        total_cluster = len(df_clustered["Cluster"].unique())

        cluster = 0
        df_clust_dict = {}
        for i in range(0, total_cluster):
            df_clust = df_clustered.copy()
            df_clust = df_clust[df_clust["Cluster"] == cluster].drop(columns="Cluster")
            df_clust_dict["cluster_" + str(i)] = df_clust
            cluster += 1

        return df_clust_dict

