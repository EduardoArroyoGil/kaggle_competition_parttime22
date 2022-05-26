import pandas as pd
import preparation_module.analysis as analysis
import preparation_module.encoding as encoding
import prediction_module.launcher as launcher
import time


start_time = time.time()

# DATA INGESTION
df = pd.read_csv("data/train.csv")

# EXPLORATORY DATA ANALYSIS
print("\nEXPLORATORY DATA ANALYSIS :\n")
eda = analysis.Eda(df)
eda.total_eda()

# ENCODING PROCESS
print("\nENCODING PROCESS :\n")
encode = encoding.Encode(df)
df = encode.get_dummies_by_column('color')
df, traduction_cut = encode.ordinalencoding('cut', ['Premium', 'Ideal', 'Very Good', 'Fair', 'Good'])
df, traduction_clarity = encode.ordinalencoding('clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

print("\nColumns of Data Frame encoded :\n\n", df.columns)
print("\nTraduction of cut labeling :\n\n", traduction_cut)
print("\nTraduction of clarity labeling :\n\n", traduction_clarity)


# LAUNCHING FITTING MODEL PROCESS
print("\nLAUNCHING FITTING MODEL PROCESS :\n")
process = launcher.Process(standarization='standard_scaler', normalizacion='min_max_scaler')
total_results, df_total, dict_models_total = process.raw(df)
clusters_results, df_clusters, dict_models_clusters = process.clustered_data(df)

print(total_results)
print(clusters_results)

print("--- %s minutes ---" % round((time.time() - start_time)/60, 2))
