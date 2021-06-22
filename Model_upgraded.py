import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
from Preprocessing import load_rnn

##Parameter of Network
n_hidden = 32
n_classes = 2
X_train, Y_train, X_test, Y_test = load_rnn()
training_data_count = len(X_train)
test_data_count = len(X_test)
n_steps = len(X_train[0])
n_input = len(X_train[0][0])
labda1 = 0.0030
training_iters = training_data_count * 300
batch_size = 150
display_iter = 3000

def shape():
    print('Training Data Count = {%d}' % (training_data_count))
    print('Test Data Count = {%d}' % (test_data_count))
    print('n Steps %d' % (n_steps))
    print('n Inputs %d' % (n_input))


def en_batch_size(train, step, size):
    shape = list(train.shape)
    shape[0] = size
    batch_s = np.empty(shape)
    for i in range(size):
        index = ((step - 1) * size + i) % len(train)
        batch_s[i] = train[index]

    return batch_s


def one_hot(y_, n_classes=n_classes):
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]


def Recurrent_Network(x, w, b):
    x = tf.transpose(a=x, perm=[1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.nn.relu(tf.matmul(x, w['hidden']) + b['hidden'])
    x = tf.split(x, n_steps, 0)

    cell_1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    cell_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell_1, cell_2], state_is_tuple=True)
    out, states = tf.compat.v1.nn.static_rnn(cells, x, dtype=tf.float32)
    l_out = out[-1]
    return tf.matmul(l_out, w['out']) + b['out']

def train(lambda1, l_rate, train_iter, display_iter):
    test_loss = []
    test_acc = []
    train_loss = []
    train_acc = []
    x = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])
    w = {'hidden': tf.Variable(tf.compat.v1.random_normal([n_input, n_hidden])),
         'out': tf.Variable(tf.compat.v1.random_normal([n_hidden, n_classes], mean=1.0))}
    b = {'hidden': tf.Variable(tf.compat.v1.random_normal([n_hidden])), 'out': tf.Variable(tf.compat.v1.random_normal([n_classes]))}
    pred = Recurrent_Network(x, w, b)
    l2 = lambda1 * sum( tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables() )
    cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y), logits=pred)) + l2
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(input=pred, axis=1), tf.argmax(input=y, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32))

    sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    step = 1
    while step * batch_size <= train_iter:
        batch_xs = en_batch_size(X_train, step, batch_size)
        batch_ys = one_hot(en_batch_size(Y_train, step, batch_size))

        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={
                x: batch_xs,
                y: batch_ys
            }
        )
        train_loss.append(loss)
        train_acc.append(acc)

        if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > train_iter):
            print("Training iter #" + str(step * batch_size) + \
                  ":   Batch Loss = " + "{:.6f}".format(loss) + \
                  ", Accuracy = {}".format(acc))
            loss, acc = sess.run(
                [cost, accuracy],
                feed_dict={
                    x: X_test,
                    y: one_hot(Y_test)
                }
            )
            test_loss.append(loss)
            test_acc.append(acc)
            print("PERFORMANCE ON TEST SET: " + \
                  "Batch Loss = {}".format(loss) + \
                  ", Accuracy = {}".format(acc))

        step += 1

    return test_loss, test_acc, train_loss, train_acc

def training():
    test_loss, test_acc, train_loss, train_acc = train(labda1, 0.0025, training_iters, display_iter)
    return test_loss, test_acc, train_loss, train_acc

def plot(test_losses, train_losses ):
    plt.figure(figsize=(16, 16))

    indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
    plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")

    indep_test_axis = np.append(
        np.array(range(batch_size, len(test_losses) * display_iter, display_iter)[:-1]),
        [training_iters]
    )
    plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss values)')
    plt.xlabel('Training iteration')

    plt.savefig('Train.png')
    return None
