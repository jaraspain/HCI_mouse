from sklearn.ensemble import RandomForestRegressor
from Preprocessing import load_rfr, load_svm
import numpy as np
from sklearn import svm

def rfr():
    data_x, data_y = load_rfr()
    data_x = np.concatenate(data_x)
    data_y = np.concatenate(data_y)
    rf = RandomForestRegressor()
    rf.fit(data_x, data_y)
    list = []
    for i in range(data_x.shape[1]):
        if (rf.feature_importances_[i] >= 0.0045):
            list.append(i)

    train_x, train_y, test_x, test_y = load_svm(list)
    clf = svm.OneClassSVM(kernel='rbf', gamma=0.4, nu=0.23)
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)

    clf.fit(train_x)
    y_train = clf.predict(train_x)
    count = 0
    for i in range(y_train.shape[0]):
        if (y_train[i] == 1):
            count = count + 1
    print("Training accuracy:"),
    print(float(count * 100) / y_train.shape[0])

    # Accuracy on Test Data
    y_test = clf.predict(test_x)
    count = 0
    for i in range(y_test.shape[0]):
        if (y_test[i] == test_y[i]):
            count = count + 1
        # print(count)
        # print(y_test.shape[0])
    print("Test accuracy:"),
    print(float(count * 100) / y_test.shape[0])

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(y_test.shape[0]):
        if (y_test[i] == 1):
            if (test_y[i] == 1):
                tp = tp + 1
            if (test_y[i] == -1):
                fp = fp + 1
        if (y_test[i] == -1):
            if (test_y[i] == 1):
                fn = fn + 1
            if (test_y[i] == -1):
                tn = tn + 1

    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    F1_score = (2 * precision * recall) / (precision + recall)

    return precision, recall, F1_score
