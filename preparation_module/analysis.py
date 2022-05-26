import logging


class Eda:

    def __init__(self, df):
        # nuestra clase va a recibir dos parámetros que son fijos a lo largo de toda la BBDD, el nombre de la BBDD y la contraseña con el servidor.
        self.df = df

    def shapeOf(self):
        logging.info('\nCuantas filas y columnas tenemos \n', self.df.shape, '\n----------------\n')

    def count_null(self):
        logging.info('\nCuántos valores nulos tenemos en el dataset ', self.df.isnull().sum(), '\n----------------\n')

    def count_nan(self):
        logging.info('\nCuántos valores nan tenemos en el dataset ', self.df.isna().sum(), '\n----------------\n')

    def count_duplicates(self):
        logging.info('\nCuántos valores duplicados tenemos en el dataset ', self.df.duplicated().sum(), '\n----------------\n')

    def typesOf(self):
        logging.info('\nExploramos los tipos de los datos que tenemos ', self.df.dtypes, '\n----------------\n')

    def describing(self):
        logging.info(self.df.describe().T, '\n----------------\n')

    def total_eda(self):
        self.shapeOf()
        self.count_null()
        self.count_nan()
        self.count_duplicates()
        self.typesOf()
        self.describing()
