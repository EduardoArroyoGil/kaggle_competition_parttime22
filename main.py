import pandas as pd
import numpy as np
import sidetable


import preparation.analysis as analysis
import preparation.encoding as encoding
import prediction.standarize as standarize
import prediction.normalize as normalize
import prediction.models as models
import time


start_time = time.time()

# DATA INGESTION
df = pd.read_csv("train.csv")

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


# CLUSTERING DATA
print("\nCLUSTERING DATA :\n")
unsupervised = models.Unsupervised(df)
df, clust_knn = unsupervised.clustering_knn(2)
df_clust = unsupervised.separate_df(df)
print('Number of clusters: ', len(df_clust))

# cleaning the raw data
df.drop(columns="Cluster", inplace=True)

# DataFrame where every result for each model and each cluster will be stored
total_results = pd.DataFrame()

# modeling for each cluster
for key, value in df_clust.items():

    print('\nSTANDARIZATION, NORMALIZATION and PREDICTING PROCESS for ', key)

    # geting dataframe for each cluster
    df = value

    # STANDARIZE PROCESS
    print("\nSTANDARIZE PROCESS :\n")
    stand = standarize.Stand(df, 'price')
    # df = stand.stand_stand_scaler()
    # df = stand.stand_robust_scaler()
    df = stand.stand_manual()
    print(df.columns)

    # NORMALIZE PROCESS
    print("\nNORMALIZE PROCESS :\n")
    norm = normalize.Norm(df, 'price')
    # df, average, maximum, minimum = norm.manual()
    # df = norm.logarithm()
    df = norm.root_square()
    # df, model_fitted = norm.min_max_scaler()
    print(df.columns)

    # PREDICTING PROCESS
    print("\nPREDICTING PROCESS :\n")
    supervised = models.Supervised(value)
    lr, dt, rf, results = supervised.all_models()
    results["Cluster"] = key

    total_results = total_results.append(results, ignore_index=True)

    # DESNORMALIZE PROCESS
    print("\nDESNORMALIZE PROCESS :\n")
    desnorm = normalize.DesNorm(df, 'price')
    # df = desnorm.manual(average, maximum, minimum)
    # df = desnorm.logarithm()
    df = desnorm.root_square()
    # df = desnorm.min_max_scaler(model_fitted)

print(total_results)

print("--- %s minutes ---" % round((time.time() - start_time)/60,2))