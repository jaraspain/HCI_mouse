import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from Preprocessing import load
from sklearn import metrics

def model_nn():
    model = tf.keras.Sequential()
    #model.add(tf.keras.Input(shape=(16, )))
    model.add(tf.keras.layers.Dense(30, input_dim= 62, activation='relu'))
    model.add(tf.keras.layers.Dense(15, activation = 'tanh'))
    #model.add(tf.keras.layers.Dense(120, activation='relu'))
    #model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    model.add(tf.keras.layers.Softmax())
    return model

def train_model_nn():
    train_x, train_y, test_x, test_y = load()
    print('Training Neural Network in Progress...')
    model = model_nn()
    print(model.summary())
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(train_x.T.shape)
    #train_x = (train_x).reshape((16,1330))
    #train_y = np.transpose(train_y).reshape((15,1330))
    #test_x = np.transpose(test_x)
    #test_y = np.transpose(test_y)
    model.fit((train_x), (train_y), epochs = 5000, batch_size = 10)
    print(train_x.shape)
    print(train_y.shape)
    _, accuracy = model.evaluate((test_x), (test_y))
    predictions = model.predict(test_x)
    predictions = np.argmax(predictions, axis=1)
    test_y = np.argmax(test_y, axis = 1)
    print("Precision: {}%".format(100 * metrics.precision_score(test_y, predictions, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(test_y, predictions, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(test_y, predictions, average="weighted")))

    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(test_y, predictions)
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

    print("Confusion matrix (normalised to % of total test data):")
    print(normalised_confusion_matrix)
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    n_classes = 15
    LABELS = ['USER1','USER2', 'USER3', 'USER4', 'USER5', 'USER6', 'USER7', 'USER8', 'USER9', 'USER10', 'USER11', 'USER12',
              'USER13', 'USER14', 'USER15']
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('MATRIX3')

    return accuracy

