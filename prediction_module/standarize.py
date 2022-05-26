from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


class Stand:

    def __init__(self, df, predicted):

        self.df = df
        self.predicted = predicted

    def stand_stand_scaler(self):
        '''

        :return:this function return the data frame standarized by all the columns except the respond (predicted)
        variable by the standard scaler from sklearn library
        '''

        # iniciamos el método para escalar
        scaler = StandardScaler()

        # ajustamos nuestros datos
        columns_df = list(self.df.columns)
        columns_standard = columns_df
        if self.predicted in self.df.columns:
            columns_standard.remove(self.predicted)

        columns_standard_renamed = columns_standard.copy()
        counter = 0
        for i in columns_standard_renamed:
            columns_standard_renamed[counter] = i + '_ES_STANDARD'
            counter += 1

        scaler.fit(self.df[columns_standard])

        X_scaled = scaler.transform(self.df[columns_standard])

        self.df[columns_standard_renamed] = X_scaled

        self.df.drop(columns=columns_standard, inplace=True)

        return self.df

    def stand_robust_scaler(self):
        '''

        :return:this function return the data frame standarized by all the columns except the respond (predicted)
        variable by the robust scaler from sklearn library
        '''
        # construir el modelo de escalador
        robust = RobustScaler()

        # ajustamos nuestros datos
        columns_df = list(self.df.columns)
        columns_standard = columns_df
        if self.predicted in self.df.columns:
            columns_standard.remove(self.predicted)

        columns_standard_renamed = columns_standard.copy()
        counter = 0
        for i in columns_standard_renamed:
            columns_standard_renamed[counter] = i + '_ES_ROBUST'
            counter += 1

        robust.fit(self.df[columns_standard])

        X_scaled = robust.transform(self.df[columns_standard])

        self.df[columns_standard_renamed] = X_scaled

        self.df.drop(columns=columns_standard, inplace=True)

        return self.df

    def stand_manual(self):
        '''

        :return: this function return the data frame standarized by all the columns except the respond (predicted)
        variable by manual method based on formula (x-average)/standar_deviation
        '''

        columns_df = list(self.df.columns)
        columns_standard = columns_df
        if self.predicted in self.df.columns:
            columns_standard.remove(self.predicted)

        columns_standard_renamed = columns_standard.copy()
        counter = 0
        for i in columns_standard_renamed:
            columns_standard_renamed[counter] = i + '_ES_MANUAL'
            counter += 1

        # vamos a crear distintas variables con los estadísticos que necesitamos, media y desviacion estándar

        for i in columns_standard:
            average = self.df[i].mean()
            deviation = self.df[i].std()

            self.df[i + '_ES_MANUAL'] = (self.df[i] - average)/deviation

        self.df.drop(columns=columns_standard, inplace=True)

        return self.df
