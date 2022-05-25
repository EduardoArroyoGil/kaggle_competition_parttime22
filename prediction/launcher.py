import prediction.standarize as standarize
import prediction.normalize as normalize
import prediction.models as models
import pandas as pd


class Process:

    def __init__(self, standarization='standard_scaler', normalizacion='min_max_scaler'):
        # nuestra clase va a recibir dos parámetros que son fijos a lo largo de toda la BBDD, el nombre de la BBDD y la contraseña con el servidor.

        self.standarization = standarization
        self.normalization = normalizacion

    def raw(self, df, data='total'):
        '''

        :param df_raw: data frame used to calculate the predict model
        :param data: the cluster of data you want to use to fit the prediction, total by default
        :return: this funcion returns the results of the models wihtin a dataframe called 'results', the data frame used in the preocess and the
        models of predictions.
        '''

        standarization = self.standarization
        normalization = self.normalization

        # STANDARIZE PROCESS
        print(f"\nSTANDARIZE PROCESS for {data} Data Frame:\n")

        # init of standarize method
        stand = standarize.Stand(df, 'price')

        # choosing the standarize method
        if standarization == 'standard_scaler':
            df = stand.stand_stand_scaler()
        elif standarization == 'robust_scaler':
            df = stand.stand_robust_scaler()
        elif standarization == 'manual':
            df = stand.stand_manual()
        print(df.columns)

        # NORMALIZE PROCESS
        print(f"\nNORMALIZE PROCESS for {data} Data Frame:\n")

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
        print(df.columns)

        # PREDICTING PROCESS
        print(f"\nPREDICTING PROCESS for {data} Data Frame:\n")
        supervised = models.Supervised(df)
        lr, dt, rf, results = supervised.all_models()
        results["Cluster"] = data

        # DESNORMALIZE PROCESS
        print(f"\nDESNORMALIZE PROCESS for {data} Data Frame:\n")

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

        return results, df, lr, dt, rf

    def clustered_data(self, df_to_cluster, number_clusters=2):
        '''

        :param number_clusters: number of clusters you would like to divide, 2 by default
        :return:
        '''

        df = df_to_cluster

        # CLUSTERING DATA
        print("\nCLUSTERING DATA :\n")
        unsupervised = models.Unsupervised(df)
        df, clust_knn = unsupervised.clustering_knn(number_clusters)
        df_clust = unsupervised.separate_df(df)
        print('Number of clusters: ', len(df_clust))

        # cleaning the raw data
        df.drop(columns="Cluster", inplace=True)

        # DataFrame where every result for each model and each cluster will be stored
        total_results = pd.DataFrame()

        # modeling for each cluster
        for key, value in df_clust.items():
            results, df, lr, dt, rf = self.raw(df=value, data=key)
            total_results = total_results.append(results, ignore_index=True)

        return total_results