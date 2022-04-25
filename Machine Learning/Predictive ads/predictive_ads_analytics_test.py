# -*- coding: utf-8 -*-
"""predictive_ads_analytics_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14GDm9abDQhHrnjPU-AX3fHcT1JLmsB1u

The following document has information about my process of experimenting with various models to solve a classification problem, in which I tested different models of supervised and unsupervised learning. Which then save in a file to be consumed after training.

As it is a proof of concept the code is not ordered and neither optimized.
"""

import pandas as pd
import os
from os import walk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
from xgboost import plot_tree

import joblib

import tensorflow as tf

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'credentials.json'

query_url = """

WITH urls_main_table AS (
    SELECT 
    source,   
    ad_id,   
    image_asset_url as url
    FROM `omg-latam-prd-adh-datahouse-cl.clients_table.main_url_dh`
    where source = 'Facebook ads' and image_asset_url != '0'
    group by 1,2,3  

    UNION ALL 

    SELECT 
    source,   
    ad_id,   
    promoted_tweet_card_image_url as url
    FROM `omg-latam-prd-adh-datahouse-cl.clients_table.main_url_dh`
    where source = 'Twitter ads' and promoted_tweet_card_image_url != '0'
    group by 1,2,3  

),

ad_values AS (
    SELECT date,
    ad_id, spend, post_engagements as clicks
    FROM `main_views_tables.main_ad`
    where regexp_contains(account_name, '(?i).*scoti.*')
),

spend_average AS (
    select ad_id, avg(spend) as avg_spend, avg(post_engagements) as avg_clicks from `main_views_tables.main_ad`
    group by 1
),

categorical_feats AS (

    SELECT ad_id, main_topic, main_color, locale, second_topic, second_color, third_topic, third_color
    FROM `clients_table.main_categorical_values`
)



SELECT 

date, source, url, e.ad_id,
main_topic, main_color, locale, second_topic, second_color, third_topic, third_color,

(CASE 
WHEN spend >= avg_spend THEN 1
WHEN spend < avg_spend THEN 0
END) as over_avg_spend, 

(CASE 
WHEN clicks >= avg_clicks THEN 1
WHEN clicks < avg_clicks THEN 0
END) as over_avg_clicks

FROM
(SELECT
date, source, url, c.ad_id,
main_topic, main_color, locale, second_topic, second_color, third_topic, third_color, spend, clicks
FROM (SELECT date,
a.source, a.url, b.ad_id, b.spend, b.clicks
FROM urls_main_table a
RIGHT JOIN ad_values b
ON a.ad_id = b.ad_id) c

INNER JOIN categorical_feats d

ON c.ad_id = d.ad_id) e

INNER JOIN spend_average f
ON e.ad_id = f.ad_id

"""

dataframe = pd.read_gbq(query_url)

dataframe = dataframe[['main_topic', 'main_color', 'second_topic', 'second_color', 'third_topic', 'third_color', 'locale', 'over_avg_spend', 'over_avg_clicks']]

vars = ['main_topic', 'main_color', 'second_topic', 'second_color', 'third_topic', 'third_color', 'locale']

for var in vars:
  cat_list = 'var' + '_' + var
  cat_list = pd.get_dummies(dataframe[var], prefix= var)
  dataframe1 = dataframe.join(cat_list)
  dataframe = dataframe1

dataframe

data_vars = dataframe.columns.values.tolist()
to_keep = [i for i in data_vars if i not in vars]

data_final = dataframe[to_keep]
data_final.columns.values

data_final_vars = data_final.columns.values.tolist()
v_y = ['over_avg_clicks']
v_x = [i for i in data_final_vars if i not in v_y]

X = dataframe[v_x]
y = dataframe[v_y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 40, stratify = y)

model_xgb = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 105, max_depth=75, seed = 10, reg_alpha=7)
model_xgb.fit(X_train.values, y_train.values.ravel())
y_pred = model_xgb.predict(X_test.values)

print('Area debajo de la curva:', roc_auc_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('accuracy:', round(accuracy_score(y_test, y_pred), 2))

print(confusion_matrix(y_test, y_pred))

base_frame = dict.fromkeys(list(X_train.columns), [0])

main_topic = 'Clothing' #@param ['Table', 'Smile','Laptop', 'video', 'Arm', 'Hair', 'Jaw', 'Forehead','Head','Chin','Clothing','Sleeve','Plant','Tire', 'Eyelash','Hand', 'Mobilephone', 'Glasses','Shorts']
second_topic = 'Jaw' #@param ['Jaw', 'Product', 'Flashphotography', 'Sleeve', 'Beard', 'Computer', 'Glasses', 'Visioncare', 'Fashion', 'Shirt', 'Jeans', 'Wheel', 'Jersey', 'Smile', 'CommunicationDevice', 'Plant', 'Mobilephone', 'Green', 'Chin', 'Human']
third_topic = 'Font' #@param ['Beard', 'Font', 'Jersey', 'Jaw', 'Chin', 'Eyewear', 'Sleeve', 'Cap', 'Smile', 'Tableware', 'Personalcomputer', 'Eyelash', 'Skin',  'Landvehicle', 'Tabletcomputer', 'Gesture', 'Organism', 'Outerwear', 'Flashphotography', 'Sportsuniform', 'Furniture']

locale = 'en' #@param ['es', 'en']

main_color = 'skyblue' #@param ['black', 'darkslategray', 'darkolivegreen', 'cadetblue', 'dodgerblue', 'mediumpurple', 'hotpink', 'skyblue', 'dimgray', 'linen', 'yellowgreen', 'sienna']
second_color = 'darkgray' #@param ['darkslategray', 'dimgray', 'black', 'darkgray', 'seagreen','skyblue', 'maroon', 'paleturquoise', 'silver', 'crimson','darkgreen', 'slategray', 'mediumpurple', 'gray']
third_color = 'lightgray' #@param ['darkslategray', 'slateblue', 'lightgray', 'mistyrose', 'gray', 'maroon', 'black', 'tan', 'darkgray', 'crimson', 'slategray', 'dimgray', 'silver']

over_avg_spend = 'False' #@param ['True', 'False']
if over_avg_spend == 'False':
  over_avg_spend = 0
else:
  over_avg_spend = 1

dataframe = pd.DataFrame({'main_topic':[main_topic], 'main_color':[main_color], 'second_topic':[second_topic], 'second_color':[second_color], 
                          'third_topic':[third_topic], 'third_color':[third_color], 'locale':[locale], 'over_avg_spend':[over_avg_spend]})

vars = ['main_topic', 'main_color', 'second_topic', 'second_color', 'third_topic', 'third_color', 'locale']

for var in vars:
  cat_list = 'var' + '_' + var
  cat_list = pd.get_dummies(dataframe[var], prefix= var)
  dataframe1 = dataframe.join(cat_list)
  dataframe = dataframe1

data_vars = dataframe.columns.values.tolist()
to_keep = [i for i in data_vars if i not in vars]

data_final = dataframe[to_keep]
my_dict_frame = data_final.to_dict('records')

base_frame.update(my_dict_frame[0])
to_predict_frame = pd.DataFrame(base_frame)

result = model_xgb.predict(to_predict_frame.values)
result_prob = model_xgb.predict_proba(to_predict_frame.values)

print('If the ad can be above the average of the results studied it will be classified as 1, otherwise it will be classified as 0. \nThe classification of the ad model entered is: {}'.format(result))

print('Probability of not being successful according to the given parameters: {}%'.format(result_prob[0][0] * 100),
      '\nProbability of success according to the given parameters: {}%'.format(result_prob[0][1]*100))

plot_tree(model_xgb, num_trees=75)
plt.show()

"""### FIRST MDOEL"""

model_classifier = Sequential()

input_dimension = len(v_x)
layers = 6
neurons = 16
output_layer = 1

for layer in range(layers):
  model_classifier.add(Dense(neurons, activation= 'relu', kernel_initializer= 'random_normal'))
  
  if layer == layers-1:
    model_classifier.add(Dense(output_layer, activation= 'sigmoid', kernel_initializer='random_normal'))

define_optimizer = keras.optimizers.Adam(lr = 0.01)
model_classifier.compile(optimizer = define_optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

history_values = model_classifier.fit(x = X_train.values, y = y_train.values, batch_size = 15, epochs = 350, validation_split = 0.15)

model_classifier.summary()

evaluation_model = model_classifier.evaluate(X_train, y_train)
print(evaluation_model)

y_pred = model_classifier.predict(X_test).ravel()
y_pred = (y_pred > 0.5)

print('Area debajo de la curva:', roc_auc_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('accuracy:', round(accuracy_score(y_test, y_pred), 2))

print(confusion_matrix(y_test, y_pred))

"""## KERAS SKLEARN"""

def create_model(init='random_normal'):

  model = Sequential()
  model.add(Dense(16, kernel_initializer=init, activation='relu'))
  model.add(Dense(16, kernel_initializer=init, activation='relu'))
  model.add(Dense(16, kernel_initializer=init, activation='relu'))
  model.add(Dense(16, kernel_initializer=init, activation='relu'))
  model.add(Dense(16, kernel_initializer=init, activation='relu'))
  model.add(Dense(16, kernel_initializer=init, activation='relu'))
  model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
  define_optimizer = keras.optimizers.Adam(lr = 0.01)
  model.compile(loss='binary_crossentropy', optimizer=define_optimizer, metrics=['accuracy'])
  return model
 

model = KerasClassifier(build_fn=create_model, verbose=0, validation_split=0.35)
model._estimator_type = "classifier"
init = ['random_normal']
epochs = [350]
batches = [15]

param_grid = dict(epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

print(means, stds, params)

y_pred = grid_result.predict(X_test)

print('Area debajo de la curva:', roc_auc_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('accuracy:', round(accuracy_score(y_test, y_pred), 2))
print('')
print(confusion_matrix(y_test, y_pred))

"""### OTHER MODEL"""

tree_model = RandomForestClassifier()

param_grid = {'criterion':['gini', 'entropy'], 'max_depth': np.arange(5,15,5), 'min_samples_leaf': np.arange(1,15,10), 'n_estimators': np.arange(5,105,10), 'ccp_alpha':[0.0, 0.1,0.5,0.9, 1]}

grid_tree_model = GridSearchCV(tree_model, param_grid, cv = 10)
grid_tree_model.fit(X_train.values, y_train.values.ravel())

print('Tuned Decision Tree: {}'.format(grid_tree_model.best_params_))
print('Tuned accuracy: {}'.format(grid_tree_model.best_score_))

y_pred = grid_tree_model.predict(X_test.values)

print('Area debajo de la curva:', roc_auc_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('accuracy:', round(accuracy_score(y_test, y_pred), 2))

print(confusion_matrix(y_test, y_pred))

"""### OTHER XGBOOST"""

import xgboost as xgb

model_xgb = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 105, max_depth=75, seed = 10, reg_alpha=7)
model_xgb.fit(X_train.values, y_train.values.ravel())
y_pred = model_xgb.predict(X_test.values)

print('Area debajo de la curva:', roc_auc_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))

round(accuracy_score(y_test, y_pred), 2)

print(confusion_matrix(y_test, y_pred))

"""### SKLEARN NN"""

nn_class = MLPClassifier()

param_grid = {'hidden_layer_sizes':[(8,8,8,8,8,8,8), (16,16,16,16,16,16)], 
              'activation': ['relu', 'logistic', 'tanh'],
              'solver': ['lbfgs', 'adam'],
              'max_iter': [800],
              'learning_rate': ['constant', 'adaptive']
              }

grid_nn_class = GridSearchCV(nn_class, param_grid, cv = 10)
grid_nn_class.fit(X_train.values, y_train.values.ravel())

print('Tuned Decision Tree: {}'.format(grid_nn_class.best_params_))
print('Tuned accuracy: {}'.format(grid_nn_class.best_score_))

y_pred = grid_nn_class.predict(X_test.values)

print('Area debajo de la curva:', roc_auc_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('accuracy:', round(accuracy_score(y_test, y_pred), 2))

print(confusion_matrix(y_test, y_pred))

"""### VOTING"""

client = 'someclient'

model_classifier.save("model.h5") # keras

joblib.dump(model_xgb, 'xgboost_tree' + client +'.pkl') #xgboost
joblib.dump(grid_tree_model, 'sklearn_tree' + client +'.pkl') #sklearn
joblib.dump(grid_nn_class, 'sklearn_nn'+ client +'.pkl') #sklearn

_, _, filenames = next(walk('/content'))