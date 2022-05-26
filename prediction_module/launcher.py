import prediction_module.standarize as standarize
import prediction_module.normalize as normalize
import prediction_module.models as models
import preparation_module.encoding as encoding
import tools.utils as utils
import pandas as pd
import logging


class Process:

    def __init__(self, standarization='None', normalizacion='None'):
        # nuestra clase va a recibir dos parámetros que son fijos a lo largo de toda la BBDD, el nombre de la BBDD y la contraseña con el servidor.

        self.standarization = standarization
        self.normalization = normalizacion

    def raw(self, df, data='total'):
        '''

        :param df_raw: data frame used to calculate the predict model
        :param data: the cluster of data you want to use to fit the prediction_module, total by default
        :return: this funcion returns the results of the models wihtin a dataframe called 'results', the data frame used
         in the preocess and the models of predictions.
        '''

        # starting method utils
        util = utils.Utils()

        standarization = self.standarization
        normalization = self.normalization

        # STANDARIZE PROCESS
        logging.info(f"STANDARIZE PROCESS for {data} Data Frame:\n")
        logging.info(f"Standarize method: {standarization}\n")

        # init of standarize method
        stand = standarize.Stand(df, 'price')

        # choosing the standarize method
        if standarization == 'standard_scaler':
            df = stand.stand_stand_scaler()
        elif standarization == 'robust_scaler':
            df = stand.stand_robust_scaler()
        elif standarization == 'manual':
            df = stand.stand_manual()
        elif standarization == 'None':
            pass
        logging.info(df.columns)

        # NORMALIZE PROCESS
        logging.info(f"NORMALIZE PROCESS for {data} Data Frame:\n")
        logging.info(f"Normalize method: {normalization}\n")

        # init of normalize method
        norm = normalize.Norm(df, 'price')

        # choosing the normalize method
        if normalization == 'manual':
            df, average, maximum, minimum = norm.manual()
        elif normalization == 'log':
            df = norm.logarithm()
        elif normalization == 'rtsq':
            df = norm.root_square()
        elif normalization == 'min_max_scaler':
            df, model_fitted = norm.min_max_scaler()
        elif normalization == 'None':
            pass
        logging.info(df.columns)

        # ensuring column 'price' is in the end of dataframe
        df = util.move_price_to_end(df)

        logging.info(f"columns of {data} DF to fit models encoded: {df.columns}")

        # PREDICTING PROCESS
        logging.info(f"PREDICTING PROCESS for {data} Data Frame:\n")
        supervised = models.Supervised(df)
        all_models, results = supervised.all_models()
        results["Cluster"] = data

        # DESNORMALIZE PROCESS
        logging.info(f"DESNORMALIZE PROCESS for {data} Data Frame:\n")
        logging.info(f"Desnormalize method: {normalization}\n")

        # init of desnormalize method
        desnorm = normalize.DesNorm(df, 'price')

        # choosing the desnormalize method
        if normalization == 'manual':
            df = desnorm.manual(average, maximum, minimum)
        elif normalization == 'log':
            df = desnorm.logarithm()
        elif normalization == 'rtsq':
            df = desnorm.root_square()
        elif normalization == 'min_max_scaler':
            df = desnorm.min_max_scaler(model_fitted)
        elif normalization == 'None':
            pass

        dict_models = dict()

        dict_models[data] = all_models

        return results, df, dict_models

    def clustered_data(self, df_to_cluster, number_clusters=2):
        '''

        :param number_clusters: number of clusters you would like to divide, 2 by default
        :return: this funcion returns the results of the models wihtin a dataframe called 'results', the data frame used
         in the preocess and the models of predictions separated by clusters
        '''

        df = df_to_cluster

        # CLUSTERING DATA
        logging.info("CLUSTERING DATA :\n")
        logging.info(f"Number of clusters :{number_clusters}\n")
        unsupervised = models.Unsupervised(df)
        df, clust_knn = unsupervised.clustering_knn(number_clusters)
        df_clust = unsupervised.separate_df(df)
        logging.info('Number of clusters: ', len(df_clust))

        # cleaning the raw data
        df.drop(columns="Cluster", inplace=True)

        # DataFrame where every result for each model and each cluster will be stored
        total_results = pd.DataFrame()

        # modeling for each cluster
        dict_models = dict()
        for key, value in df_clust.items():
            results, df, dict_models_cluster = self.raw(df=value, data=key)
            total_results = total_results.append(results, ignore_index=True)
            dict_models.update(dict_models_cluster)

        return total_results, df, dict_models

    def predict(self, df_to_predict, best_model, all_models, delivery_id):
        '''

        :param df_to_predict:
        :param best_model:
        :param all_models:
        :param delivery_id:
        :return:
        '''

        standarization = self.standarization
        normalization = self.normalization

        df_prediction = pd.DataFrame()
        df_prediction_delivery = pd.DataFrame()
        for key, value in best_model.items():
            cluster = key
            model_type = value

            model_cluster = all_models[cluster][model_type]

            # ENCODING PROCESS
            logging.debug("ENCODING PROCESS FOR PREDICTION DATA SET:\n")
            df_encoded = df_to_predict.copy()
            encode = encoding.Encode(df_encoded)
            # df_test_encoded = encode.get_dummies_by_column('color')
            df_encoded, traduction_color = encode.ordinalencoding('color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
            df_encoded, traduction_cut = encode.ordinalencoding('cut',
                                                                     ['Premium', 'Ideal', 'Very Good', 'Fair', 'Good'])
            df_encoded, traduction_clarity = encode.ordinalencoding('clarity',
                                                                         ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2',
                                                                          'VVS1', 'IF'])
            # STANDARIZE PROCESS
            logging.info(f"STANDARIZE PROCESS for {cluster} Data Frame:\n")
            logging.info(f"Standarize method: {standarization}\n")

            # init of standarize method
            stand = standarize.Stand(df_encoded, 'price')

            # choosing the standarize method
            if standarization == 'standard_scaler':
                df_encoded = stand.stand_stand_scaler()
            elif standarization == 'robust_scaler':
                df_encoded = stand.stand_robust_scaler()
            elif standarization == 'manual':
                df_encoded = stand.stand_manual()
            elif standarization == 'None':
                pass
            logging.info(f"columns of DF to predict encoded: {df_encoded.columns}")

            # predicting
            prediction_cluster = model_cluster.predict(df_encoded)
            df_prediction_cluster = pd.DataFrame({'price': prediction_cluster})

            # creating dataframes of results expected
            df_data_predicted_cluster = pd.concat([df_to_predict, df_prediction_cluster], axis=1)
            df_prediction = df_prediction.append(df_data_predicted_cluster, ignore_index=True)

            df_delivery_predicted_cluster = pd.concat([delivery_id, df_prediction_cluster], axis=1)
            df_prediction_delivery = df_prediction_delivery.append(df_delivery_predicted_cluster, ignore_index=True)

        return df_prediction, df_prediction_delivery
