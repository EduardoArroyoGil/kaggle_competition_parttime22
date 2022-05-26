import pandas as pd
import preparation_module.analysis as analysis
import preparation_module.encoding as encoding
import prediction_module.launcher as launcher
import tools.utils as utils
import time
import logging
import datetime

# ct stores current time
ct = str(datetime.datetime.now())[:19]

# setting logging method
logging.basicConfig(filename=f'logs/info_{ct}.txt', level=logging.DEBUG)

# starting time to measure timings
start_time = time.time()

# starting method utils
util = utils.Utils()

# DATA INGESTION
df = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

# df for the final delivery of the competition
df_predict_id = df_test['id']

# FEATURE ENGINEERING
# creating new features
df['feature_1'] = df['x'] - df['y']
df_test['feature_1'] = df_test['x'] - df_test['y']

df['feature_2'] = df['y'] - df['z']
df_test['feature_2'] = df_test['y'] - df_test['z']

# dropping id columns
columns_drop = ['id', 'y', 'x', 'z']
df.drop(columns=columns_drop, inplace=True)
df_test.drop(columns=columns_drop, inplace=True)

# EXPLORATORY DATA ANALYSIS
logging.debug("EXPLORATORY DATA ANALYSIS :\n")
eda = analysis.Eda(df)
eda.total_eda()

# ENCODING PROCESS
logging.debug("ENCODING PROCESS :\n")
encode = encoding.Encode(df)
# df = encode.get_dummies_by_column('color')
df, traduction_color = encode.ordinalencoding('color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
df, traduction_cut = encode.ordinalencoding('cut', ['Premium', 'Ideal', 'Very Good', 'Fair', 'Good'])
df, traduction_clarity = encode.ordinalencoding('clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

logging.info("Columns of Data Frame encoded :\n", df.columns)
logging.info("Traduction of cut labeling :\n", traduction_cut)
logging.info("Traduction of clarity labeling :\n", traduction_clarity)

# ensuring column 'price' is in the end of dataframe
df = util.move_price_to_end(df)

# LAUNCHING FITTING MODEL PROCESS
logging.debug("LAUNCHING FITTING MODEL PROCESS :\n")
process = launcher.Process(standarization='standard_scaler', normalizacion='None')
total_results, df_total, dict_models_total = process.raw(df)

logging.debug(total_results)

# GETTING BEST MODEL FOR EACH CLUSTER
logging.debug("GETTING BEST MODEL FOR EACH CLUSTER :\n")
best_model = util.get_best_model(total_results)
for key, value in best_model.items():
    logging.info(f"The best model for {key} cluster data is: {value}\n")

# PREDICTING PROCESS
df_prediction, df_prediction_delivery = process.predict(df_to_predict=df_test,
                                                        best_model=best_model,
                                                        all_models=dict_models_total,
                                                        delivery_id=df_predict_id)

# SAVING RESULTS OF PREDICTION
df_prediction.to_csv(f"result_prediction/prediction_{ct}.csv", index=False)
logging.debug(f"prediction_{ct}.csv already saved\n")

df_prediction_delivery.to_csv(f"result_prediction/delivery_prediction_{ct}.csv", index=False)
logging.debug(f"delivery_prediction_{ct}.csv already saved\n")


logging.debug("--- %s minutes ---" % round((time.time() - start_time)/60, 2))
