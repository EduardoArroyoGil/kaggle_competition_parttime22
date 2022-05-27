# Kaggle competition Part Time 22

## Folder Structure
The folder structure is composed by six folders:
  - data: location of the data to train and data to predict
  - log: where every log of each running is stored
  - prediction_module: there're 4 classes stored in this folder to manage the predictive model fitting and predictions, as well as, the normalization and standarization
       - launcher.py: to launch every convination of modesl, normalization and standarization to fit and predict
       - models.py: where every model is fitted (Linear, Decission Tree, Random Forest, Gradient Boost, Extrem Gradient Boost)
       - normalize.py: where every normalization method is stored (manual, logarithm, root square & min max scaler)
       - standarize.py: where every standarization method is stored (manual, standard scaler & robust scaler)
  - preparation_module: there're 2 classes stored in this folder to manage the analsysi of all the data and the different encoding methods
       - analysis.py: 
       - encoding.py: 
  - result_prediction: where every csv file with the prediction is stored, two kinds of prediction, the one for the competition (delivery) and the one with all the data and the price prediction
  - tools: Class for create usefull functions that are used along the whole code

## Best Configuration:
   The data Preparation has been based on 3 decisions:
   - Drop columns: id, x, y, z
   - Create new features:
       - feature 1 = x - y
       - feature 2 = y - z
   - Encode the category fields based on Ordinary Lable Coding for columns: color, cut and clarity
   - Standarize by standar scaler method
   - And not normlized

A process of predictive model has been created to choose automatically the best predictive model.
  
The Best model has been ***XGBoost*** by ***Decission Trees model*** with a ***squared error regression***.

Extra: As well, the code accept a compose predictive model by clustering using KNN algorithm. Nevertheless, has been tested by the most clear cluster division, two clusters, and this prediction is worst than the predictive model using the whole data set together.

