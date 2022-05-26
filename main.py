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

# DATA INGESTION
df = pd.read_csv("data/train.csv")

# EXPLORATORY DATA ANALYSIS
logging.debug("EXPLORATORY DATA ANALYSIS :\n")
eda = analysis.Eda(df)
eda.total_eda()

# ENCODING PROCESS
logging.debug("ENCODING PROCESS :\n")
encode = encoding.Encode(df)
df = encode.get_dummies_by_column('color')
df, traduction_cut = encode.ordinalencoding('cut', ['Premium', 'Ideal', 'Very Good', 'Fair', 'Good'])
df, traduction_clarity = encode.ordinalencoding('clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

logging.info("Columns of Data Frame encoded :\n", df.columns)
logging.info("Traduction of cut labeling :\n", traduction_cut)
logging.info("Traduction of clarity labeling :\n", traduction_clarity)

# LAUNCHING FITTING MODEL PROCESS
logging.debug("LAUNCHING FITTING MODEL PROCESS :\n")
process = launcher.Process(standarization='None', normalizacion='None')
total_results, df_total, dict_models_total = process.raw(df)

logging.debug(total_results)

# GETTING BEST MODEL FOR EACH CLUSTER
logging.debug("GETTING BEST MODEL FOR EACH CLUSTER :\n")
util = utils.Utils()
best_model = util.get_best_model(total_results)
for key, value in best_model.items():
    logging.info(f"The best model for {key} cluster data is: {value}\n")

logging.debug("--- %s minutes ---" % round((time.time() - start_time)/60, 2))
