import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random
from sklearn.decomposition import PCA
import itertools

def eval_acc(data, target, model, feature):
    if type(feature) == str:
        feature = [feature]

    k_fold = KFold(n_splits=2, random_state=300000007, shuffle=True)
    scores = []

    for train, test in k_fold.split(data[feature].values, data[target].values):
        x_train, x_test, y_train, y_test = data[feature].values[train], data[feature].values[test], \
                                           data[target].values[
                                               train], data[target].values[test]
        model.fit(x_train, y_train.reshape(-1, 1))

        scores.append(accuracy_score(y_test, model.predict(x_test)))

    return np.mean(scores)


# feature Selection with Genetic algorithm
def select(data, model, target, train_time=100, threshold=0.05):
    features = list(data.columns)
    features = features[:-3]
    gens = []
    for feature in features:
        gens.append(([feature], eval_acc(data, target, model, [feature])))

    for time in range(train_time * 10):
        print(time, 'Generation Is Started...')
        indices = [i for i in range(len(gens))]
        first_selected_index = indices.pop(np.random.randint(0, len(indices)))
        second_selected_index = indices.pop(np.random.randint(0, len(indices)))
        first_gen = gens[first_selected_index]
        second_gen = gens[second_selected_index]

        # Gens Combination
        fs = list(set(first_gen[0] + second_gen[0]))
        gens.append((list(fs), eval_acc(data, target, model, fs)))

    history = gens
    mutation_time = 5
    for i in range(train_time):
        history.sort(key=lambda x: x[1], reverse=True)
        print('accuracy:', history[0][1])
        if abs(history[0][1] - history[1][1]) / history[0][1] < threshold:
            for j in range(mutation_time):
                gen = gens[np.random.randint(0, len(gens))]
                if type(gen[0]) == list and len(gen[0]) > 1:
                    new_gen_features = gen[0].pop(random.randint(0, len(gen[0]) - 1))
                    score = eval_acc(data, target, model, new_gen_features)
                    gens.append((new_gen_features, score))
                    history.append((new_gen_features, score))

        gen1 = gens[np.random.randint(0, len(gens))]
        gen2 = gens[np.random.randint(0, len(gens))]

        if len(gen1) > 2 and len(gen2) > 2:
            min_upper = min(len(gen1[0]), len(gens[0]))
            start = np.random.randint(0, min_upper - 1)
            end = np.random.randint(start + 1, min_upper)

            new_gen1 = list(set(gen1[0][:start] + gen2[start:end] + gen1[0][end:]))
            new_gen2 = list(set(gen2[0][:start] + gen1[start:end] + gen2[0][end:]))
            score1 = eval_acc(data, target, model, new_gen1)
            score2 = eval_acc(data, target, model, new_gen2)
            gens.append((new_gen1, score1))
            gens.append((new_gen2, score2))
            history.append((new_gen1, score1))
            history.append((new_gen1, score2))

    history.sort(key=lambda x: x[1], reverse=True)
    return history[0]


def encoded_data(data, cols):
    new_data = data
    for col in cols:
        types = {
            type: len(data[col][data[col] == type])
            for type in data[col].unique()
            }
        for i in range(len(data)):
            new_data.set_value(i, col, types[data.iloc[i][col]])

    return new_data


data = pd.read_csv('../data/encoded_data_por.csv')
data = data.drop(['failures'], axis=1)
#
n_class = 2
for col in ['G1', 'G2', 'G3']:
    for i in range(len(data)):
        if data.iloc[i][col] >10:

            data.set_value(i, col, 1)
        else:
            data.set_value(i, col, 0)

data = encoded_data(data, data.columns[:-3])
for col in data.columns[:-3]:
    mean = np.mean(data[col].values)
    std = np.std(data[col].values)
    data[col] = (data[col] - mean) / std

for col in ['G1', 'G2', 'G3']:
    for type in data[col].unique():
        print(col, type, len(data[data[col] == type]))
data.to_csv('../data/encoded_data_por.csv', index=False)
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
C = np.linspace(0.001, 1, num=10)
errors = []
data = shuffle(data)
X, y = data[data.columns[:-3]].values, data['G1'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=50)
count = 0
for k in kernels:
    model = SVC(C=0.6, kernel=k)
    model.fit(x_train, y_train)
    errors.append((k, accuracy_score(y_test, model.predict(x_test))))
    print('step :', count)
    count += 1

print(errors)
# results -> kernel = poly and c = 0.57750753768844225
kernel = 'poly'
c = 0.57750753768844225

scores = []
data = shuffle(data)
for ca in c:
    data = shuffle(data)

    X, y = data[best_features].values, data['G1'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=50)
    model = SVC(C=ca, kernel=kernel)
    scores.append((ca, eval_acc(data, 'G1', model, best_features)))

scores.sort(key=lambda x: x[1], reverse=True)
print(scores[0])

# training model with selected feature
model = SVC(C=c, kernel=kernel)
best_features = ['goout', 'traveltime', 'famrel', 'freetime', 'romantic', 'Fedu', 'higher',
                 'schoolsup', 'reason', 'studytime', 'activities', 'Mjob', 'school', 'famsize']
data = shuffle(data)
X, y = data[best_features].values, data['G2'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=50)
model.fit(x_train, y_train)

# visualization  
predicted = model.predict(x_test)
pca = PCA(n_components=2)
X_P = pca.fit_transform(x_test, y_test)
#
for i in range(len(x_test)):
    if predicted[i] == 1:
        plt.scatter(X_P[i][0], X_P[i][1], c='c', alpha=0.5, s=200, cmap=plt.cm.Blues)
    else:
        plt.scatter(X_P[i][0], X_P[i][1], c='k', alpha=0.5, s=200, cmap=plt.cm.Blues)

for i in range(len(x_test)):
    if y_test[i] == 1:
        plt.scatter(X_P[i][0], X_P[i][1], c='c', alpha=0.5, s=60, cmap=plt.cm.Blues)
    else:
        plt.scatter(X_P[i][0], X_P[i][1], c='k', alpha=0.5, s=60, cmap=plt.cm.Blues)

miss = 0

for i in range(len(predicted)):
    if predicted[i] != y_test[i]:
        miss += 1

plt.title('miss' + ' ' + str(miss) + ' of ' + str(len(predicted)) )
plt.show()

