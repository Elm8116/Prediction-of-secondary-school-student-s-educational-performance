import numpy as np
import tflearn as tfl
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import KFold
import random
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle


def best_arch_for_nn(ds, features, target):
    layer = 50
    neuron = 20
    arch = [
        (i, j) for i in range(1, neuron) for j in range(1, layer)
        ]

    nn_models = {
        k: ('H:' + str(v[0]) + '\n' + 'N:' + str(v[1]),
            MLPRegressor(hidden_layer_sizes=(v[0], v[1]), activation='tanh', learning_rate='adaptive'))

        for k, v in enumerate(arch)

        }
    scores = []
    for k, v in nn_models.items():
        try:
            print(k + 1, 'th model')
            print('config: ')
            print(v[0])
            val = eval_acc(ds['robust'], target, v[1], features)
            scores.append((v[0], val))
            print(val)
            print('================================')
        except Warning:
            print('', end = '')

    scores.sort(key=lambda x: x[1])
    return scores[0]


# _______________________________________________________________


def eval_acc(data, target, model, feature):
        if type(feature) == str:
            feature = [feature]


        k_fold = KFold(n_splits=5, random_state=random_state, shuffle=True)
        scores = []
        for train, test in k_fold.split(data[feature].values, data[target].values):
            x_train, x_test, y_train, y_test = data[feature].values[train], data[feature].values[test], \
                                               data[target].values[
                                                   train], data[target].values[test]
            model.fit(x_train, y_train.reshape(-1, 1))

            scores.append(mean_squared_error(y_test, model.predict(x_test)))

        return np.mean(scores)


# _______________________________________________________________


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
        history.sort(key=lambda x: x[1])
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

    history.sort(key=lambda x: x[1])
    return history[0]


# _______________________________________________________________


def mean_square(y_pred, y_true):
    """ Mean Square Loss.
    Arguments:
        y_pred: `Tensor` of `float` type. Predicted values.
        y_true: `Tensor` of `float` type. Targets (labels).
    """
    with tf.name_scope("MeanSquare"):
        return tf.reduce_mean(tf.square(y_pred - y_true))


# _______________________________________________________________


def network(input_shape, arch, activation, optimizer, drop_out):
    print(activation)
    layers = [
        tfl.input_data(input_shape)
    ]
    for i in range(len(arch)):
        layers.append(
            tfl.dropout(tfl.layers.core.fully_connected(layers[-1], arch[i], activation[i], regularizer='L2'), 0.8))
    op = optimizer
    layers.append(tfl.layers.core.fully_connected(layers[-1], 1))
    nn = tfl.regression(layers[-1], optimizer=op, loss=mean_square)

    net = tfl.DNN(nn, tensorboard_verbose=3, checkpoint_path='../models_plots_files/model_save/')

    return net


# _______________________________________________________________


def data_scaler(data, scaler, features):
    new_data = data

    for feature in features:
        new_data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))

    return new_data


# _______________________________________________________________

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


# _______________________________________________________________


#Variables

random_state = 300000007

data = pd.read_csv('../data/student-por.csv')

features = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
            'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
            'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
            'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
            'Walc', 'health', 'absences']
need_to_encode = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                  'Mjob', 'Fjob', 'reason', 'guardian',
                  'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                  'higher', 'internet', 'romantic']
targets = ['G1', 'G2', 'G3']
encoded_data = encoded_data(data, need_to_encode)

scaler = {
    'robust': RobustScaler(),
    'max_abs': MaxAbsScaler(),
    'standard': StandardScaler()
}
ds = {
    k: data_scaler(encoded_data, v
                   , features)
    for k, v in scaler.items()
    }
for k in ds.keys():
    ds[k].to_csv('../data/data-' + k + '.csv', index=False)

# ############################
# N E U R A L   N E T W O R K
#        T U N I N G
# ############################

best = 10, 39
optimizer = [
    ('adam', tfl.Adam(learning_rate=0.002, beta1=0.8)),
    ('sgd', tfl.SGD(
        learning_rate=0.001, decay_step=2000
    )),
    ('adadelta', tfl.AdaDelta(
        learning_rate=0.001
    )),
    ('adagrad', tfl.AdaGrad()),
    ('mom', tfl.Momentum(decay_step=10000)),
]
activation = [
    'tanh',
    'linear',

]
best_score = []
# for op_name, op in optimizer:
#     best_score.append((op_name, 'sigmoid', eval_acc(data, 'G1', network(input_shape, [39] * 10, ['sigmoid'  for _ in range(10)], op, 0.8), features)))
#     tf.reset_default_graph()
#
# best_score.sort(key=lambda x: x[2])
#
# print(best_score[0])
#
# print(select(ds['robust'],model,'G1',200))
# print(eval_acc(encoded_data,'G1',model,features))
best_features = ['famsup', 'traveltime', 'guardian', 'freetime', 'goout', 'activities',
                 'address', 'Mjob', 'age', 'famsize', 'school', 'romantic', 'paid', 'Walc',
                 'absences', 'internet', 'studytime', 'Pstatus', 'health', 'Fedu', 'Dalc',
                 'failures', 'famrel', 'nursery', 'higher', 'schoolsup', 'sex', 'reason']
input_shape = (None, 28)


graph = tf.get_default_graph()
n_epoch = 100
shuffled_data = shuffle(encoded_data)
scaled_shuffled = shuffled_data[best_features + ['G1']]
X_train, X_test, y_train, y_test = scaled_shuffled.iloc[:345][best_features], scaled_shuffled.iloc[345:][best_features], \
                                   scaled_shuffled.iloc[:345]['G1'], scaled_shuffled.iloc[345:]['G1']

# Neuron in each Layer : 15
# Hidden layer : 10
arch = [15] * 10
activation = ['relu'] * 10

layers = [
        tfl.input_data(input_shape)
    ]
for i in range(len(arch)):
        layers.append(
            tfl.dropout(tfl.layers.core.fully_connected(layers[-1], arch[i], activation[i], regularizer='L2'), 0.8))
op = tfl.Momentum(decay_step=10000)

layers.append(tfl.layers.core.fully_connected(layers[-1], 1))

nn = tfl.regression(layers[-1], optimizer=op, loss=mean_square)

net = tfl.DNN(nn, tensorboard_verbose=3)


for col in best_features:
    mean = np.mean(X_train[col].values)
    std = np.std(X_train[col].values)
    X_train[col] = (X_train[col] - mean) / std
    X_test[col] = (X_test[col] - mean) / std
print('Training model...')
for i in range(30):
    net.fit(X_train.values, y_train.values.reshape(-1, 1))
    print(mean_squared_error(y_test.values, net.predict(X_test.values)))

graph.finalize()
print('final Error Of model:')
print(mean_squared_error(y_test.values, net.predict(X_test.values)))