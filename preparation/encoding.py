import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


class Encode:

    def __init__(self, df):
        self.df = df

    def one_hot_encoder(self, columns_to_encode):
        '''
        columns_to_encode: list the columns to encode as one hot encoder method
        :return: DataFrame encoded by columns inserted as input in the list
        '''

        oh = OneHotEncoder()

        transformed = oh.fit_transform(self.df[columns_to_encode])

        oh_df = pd.DataFrame(transformed.toarray(), columns=oh.get_feature_names(), dtype=int)

        self.df[oh_df.columns] = oh_df

        self.df.drop(columns_to_encode, axis=1, inplace=True)

        return self.df

    def get_dummies_by_column(self, column_to_encode):
        '''
        column_to_encode: string of the column to encode as get_dummies method from pandas
        :return: DataFrame encoded by column
        '''

        # lo haremos para la columna "region"

        dummies = pd.get_dummies(self.df[column_to_encode], prefix_sep="_", prefix=column_to_encode, dtype=int)

        # incluimos las columnas creadas por el método get_dummies y eliminamos la de region original
        self.df[dummies.columns] = dummies
        self.df.drop(columns=column_to_encode, axis=1, inplace=True)

        return self.df

    def labelencoding(self, columns_to_encode):
        '''
        columns_to_encode: list the columns to encode as one hot encoder method
        :return: DataFrame enconded by Label Encoder method, Traduction between encoded labales and Lables
        '''

        # iniciamos metodo
        le = LabelEncoder()

        # lista de columnas encoded
        columns_encoded = list()

        # aplicamos el metodo para cada columna
        for col in self.df[columns_to_encode].columns:
            new_name = col + "_encoded"
            self.df[new_name] = le.fit_transform(self.df[col])
            columns_encoded.append(col + "_encoded")

        label_traduction = self.df[columns_to_encode + columns_encoded].drop_duplicates(inplace=False).sort_values(
            by=columns_encoded)

        self.df.drop(columns=columns_to_encode, axis=1, inplace=True)

        return self.df, label_traduction

    def ordinal_map(self, column_to_encode, order_values):

        '''

        :param column_to_encode: the column that want to be encoded of the data frame
        :param order_values: the order that each lable want to be encoded by
        :return: Data Frame encoded based on mapping function created byu a dictionary
        , Lable traduction betwen label and codes applied
        '''

        ordinal_dict = {}

        for i, valor in enumerate(order_values):
            ordinal_dict[valor] = i

        columns_encoded = column_to_encode + "_mapped"

        self.df[columns_encoded] = self.df[column_to_encode].map(ordinal_dict)

        label_traduction = self.df[[column_to_encode, columns_encoded]].drop_duplicates(inplace=False).sort_values(
            by=columns_encoded)

        self.df.drop(columns=column_to_encode, axis=1, inplace=True)

        return self.df, label_traduction

    def ordinalencoding(self, column_to_encode, order):
        '''

        :param column_to_encode: the column that want to be encoded of the data frame
        :param order: the order that each lable want to be encoded by
        :return: Data Frame encoded based on OrdinalEncoder method
        , Lable traduction betwen label and codes applied
        '''

        columns_encoded = column_to_encode + "_encoded"

        # iniciamos el método y aplicamos la transformación a los datos.

        ordinal = OrdinalEncoder(categories=[order], dtype=int)
        transformed_oe = ordinal.fit_transform(self.df[[column_to_encode]])

        # lo convertimos a dataframe

        oe_df = pd.DataFrame(transformed_oe)

        self.df[columns_encoded] = oe_df

        label_traduction = self.df[[column_to_encode, columns_encoded]].drop_duplicates(inplace=False).sort_values(
            by=columns_encoded)

        self.df.drop(columns=column_to_encode, axis=1, inplace=True)

        return self.df, label_traduction
