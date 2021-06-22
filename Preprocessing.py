##This preprocesses the data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Data_extraction import extract
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

def one_hot(y_, n_classes = 15):
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype = np.int32)]

def load():
        print('Data Loading in Progress...')
        #User_3 = extract(10, 'Data/Part-3/')
        ##print(User_3.shape)
        ## print(User_3[0:5, -2])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        n = 15
        s1 = extract(0, 'Data/Part-1/')
        s1_y = one_hot(s1[: ,s1.shape[1]-1].reshape((s1.shape[0], 1)))
        s1_x = s1[:, :-1]
        s1_x = preprocessing.scale(s1_x, axis = 0)
        test_x.append(s1_x[0:(int(s1_x.shape[0] /5)), :])
        train_x.append(s1_x[(int(s1_x.shape[0]/5))+1: , :])
        test_y.append(s1_y[0:(int(s1_y.shape[0]/5)) ,:])
        train_y.append(s1_y[(int(s1_y.shape[0]/5)) +1: , :])


        #users.append(extract(0, 'Data/Part-2'))

        for i in range(1,n):
            #print(i)
            sf1 = extract(i, 'Data/Part-' + str(i+1) + '/')
            y  = one_hot(sf1[:, sf1.shape[1]-1].reshape((sf1.shape[0], 1)))
            x = sf1[:, :-1]
            x = preprocessing.scale(x, axis = 0)
            te_x = x[0:int(x.shape[0] / 5), :]
            tr_x = x[int(x.shape[0] / 5) + 1: , :]
            te_y = y[0:int(y.shape[0] / 5), :]
            tr_y = y[int(y.shape[0] / 5) + 1:, :]
            # print(tr_y.shape)

            train_x.append(tr_x)
            train_y.append(tr_y)
            test_x.append(te_x)
            test_y.append(te_y)

        #pca = PCA(0.90)
        #sf = np.concatenate(users)
        #scaler = StandardScaler()
        #sf = sf[:, :sf.shape[1]-1]
        #scaler.fit(sf)
        #sf = scaler.transform(sf)
        #pca.fit(sf)

        #print(sf.shape)
        return train_x, train_y, test_x, test_y


def load_rfr():
    print('Data Loading in Progress...')
    # User_3 = extract(10, 'Data/Part-3/')
    ##print(User_3.shape)
    ## print(User_3[0:5, -2])
    data_x = []
    data_y = []
    n = 15
    s1= extract(1, 'Data/Part-1/')
    s1_y = s1[:, s1.shape[1] - 1].reshape((s1.shape[0], 1))
    s1_x = s1[:, :-1]
    s1_x = preprocessing.scale(s1_x, axis=0)
    data_x.append(s1_x)
    data_y.append(s1_y)

    # users.append(extract(0, 'Data/Part-2'))

    for i in range(1, n):
        # print(i)
        sf1 = extract(-1, 'Data/Part-' + str(i + 1) + '/')
        y = sf1[:, sf1.shape[1] - 1].reshape((sf1.shape[0], 1))
        x = sf1[:, :-1]
        x = preprocessing.scale(x, axis=0)
        # print(tr_y.shape)
        data_x.append(x)
        data_y.append(y)

    # pca = PCA(0.90)
    # sf = np.concatenate(users)
    # scaler = StandardScaler()
    # sf = sf[:, :sf.shape[1]-1]
    # scaler.fit(sf)
    # sf = scaler.transform(sf)
    # pca.fit(sf)

    # print(sf.shape)
    return data_x, data_y

def load_svm(index):
    print('Data Loading in Progress...')
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    n = 15
    s1 = extract(1, 'Data/Part-1/')
    s1_y = s1[:, s1.shape[1] - 1].reshape((s1.shape[0], 1))
    s1_x = s1[:, :-1]
    s1_x = preprocessing.scale(s1_x, axis=0)
    test_x.append(s1_x[0:(int(s1_x.shape[0] / 5)), index])
    train_x.append(s1_x[(int(s1_x.shape[0] / 5)) + 1:, index])
    test_y.append(s1_y[0:(int(s1_y.shape[0] / 5)), :])
    train_y.append(s1_y[(int(s1_y.shape[0] / 5)) + 1:, :])

    # users.append(extract(0, 'Data/Part-2'))

    for i in range(1, n):
        # print(i)
        sf1 = extract(-1, 'Data/Part-' + str(i + 1) + '/')
        y = sf1[:, sf1.shape[1] - 1].reshape((sf1.shape[0], 1))
        x = sf1[:, :-1]
        x = preprocessing.scale(x, axis=0)
        te_x = x[0:int(x.shape[0] / 5), index]
        tr_x = x[int(x.shape[0] / 5) + 1:, index]
        te_y = y[0:int(y.shape[0] / 5), :]
        tr_y = y[int(y.shape[0] / 5) + 1:, :]
        # print(tr_y.shape)

        train_x.append(tr_x)
        train_y.append(tr_y)
        test_x.append(te_x)
        test_y.append(te_y)

    # pca = PCA(0.90)
    # sf = np.concatenate(users)
    # scaler = StandardScaler()
    # sf = sf[:, :sf.shape[1]-1]
    # scaler.fit(sf)
    # sf = scaler.transform(sf)
    # pca.fit(sf)

    # print(sf.shape)
    return train_x, train_y, test_x, test_y

def find_pca():
    users = []
    n = 15

    for i in range(n):
        sf = extract(i, 'Data/Part-' + str(i + 1) + '/')
        sf = sf[:, :-1]
        users.append(sf)

    pca = PCA(0.95)
    sf = np.concatenate(users)
    pca.fit(sf)
    return pca



def load_rnn():
    print('Data Loading in progress...')
    users = []
    n = 15
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    n = 15
    s1 = extract(0, 'Data/Part-1/')
    s1[:, :-1] = preprocessing.scale(s1[:, :-1], axis=0)
    users.append((s1))



    for i in range(n):
        datasets = extract(1, 'Data/Part-'+str(i+1)+'/')
        datasets[:, :-1] = preprocessing.scale(datasets[:, :-1], axis=0)
        users.append(datasets)

    x = []
    y = []

    for user in users:
        l_user = user.shape[0]
        #print(l_user)
        for i in range (l_user - 21):
            samples = user[i:i+20, :-1]

            y_ =user[i, user.shape[1] - 1].reshape((1,1))
            x.append(samples)
            y.append((y_))

    #print(len(x))
    #print(x[0].shape)
    #print(len(y))
    #print(y[0].shape)
    x = np.asarray(x)
    y = np.asarray(y)
    #print(x.shape)
    #print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.40, random_state=0)
    #y_train = y_train.reshape((y_train.shape[0], 1))
    #y_test = y_test.reshape((y_test.shape[0], 1))
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return x_train, y_train, x_test, y_test

def load_rnn_2():
    users = []
    n = 15
    users.append(extract(0, 'Data/Part-1/'))

    for i in range(2,14):
        datasets = extract(1, 'Data/Part-'+str(i+1)+'/')
        users.append(datasets)

    x = []
    y = []
    pca = find_pca()
    for user in users:
        l_user = user.shape[0]
        #print(l_user)
        us = user[:, :-1]
        pca.transform(us)
        for i in range (l_user - 21):
            samples = us[i:i+20]
            y_ =user[i, user.shape[1] - 1].reshape(1,1)
            x.append(samples)
            y.append((y_))

    #print(len(x))
    #print(x[0].shape)
    #print(len(y))
    #print(y[0].shape)
    x = np.asarray(x)
    y = np.asarray(y)
    #print(x.shape)
    #print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0)
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print('Data Loading Finished')
    return x_train, y_train, x_test, y_test