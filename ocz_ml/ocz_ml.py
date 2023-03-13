import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
import pickle


data = pd.read_csv('Crop_recommendation.csv')
# print(data.head())


data.dropna(inplace=True)


features = data[[
    'N', 'P', 'K', 'temperature',
    'humidity', 'ph', 'rainfall']]
# print(features.head())


output = data['label']

output, uniques = pd.factorize(output)


X_train, X_test, y_train, y_test = train_test_split(
    features, output, test_size=.8)


rfc = RFC(random_state=2023)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
score = accuracy_score(y_pred, y_test)
result = round(score, 2)*100


# print('The accuracy score for this model is {}%'.format(
#     result))


rf_pickle = open('ocz.pickle', 'wb')
pickle.dump(rfc, rf_pickle)

rf_pickle.close()

output_pickle = open('output_ocz.pickle', 'wb')
pickle.dump(uniques, output_pickle)

output_pickle.close()
