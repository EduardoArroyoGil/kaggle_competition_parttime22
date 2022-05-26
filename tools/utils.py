import pandas as pd


class Utils:

    def __init__(self):
        self

    def get_best_model(self, df):
        '''

        :param df:
        :return:
        '''

        columns_order = ['RMSE', 'Cluster']

        condition_test = df['set'] == 'test'

        clusters = df['Cluster'].unique()

        best_model = dict()

        for i in clusters:

            condition_cluster = df['Cluster'] == i

            df = df[condition_test & condition_cluster]
            df = df.sort_values(by=columns_order, ascending=False, inplace=False)

            best_model[i] = str(df['model'].iloc[0])

        return best_model

    def move_price_to_end(self, df):
        '''

        :param df: df
        :return: ensuring column 'price' is in the end of dataframe
        '''

        # ensuring column 'price' is in the end of dataframe
        if 'price' in df.columns:
            df_end = df.pop('price')
            df['price'] = df_end

        return df
