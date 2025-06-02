import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scio
import scipy.io as sio
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def random_mini_batches_GCN(X, Y, L, mini_batch_size, seed):
    
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    shuffled_L1 = L[permutation, :].reshape((L.shape[0], L.shape[1]), order = "F")
    shuffled_L = shuffled_L1[:, permutation].reshape((L.shape[0], L.shape[1]), order = "F")

    num_complete_minibatches = math.floor(m / mini_batch_size)
    
    for k in range(0, num_complete_minibatches):       
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_L = shuffled_L[k * mini_batch_size : k * mini_batch_size + mini_batch_size, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_L)
        mini_batches.append(mini_batch)
    mini_batch = (X, Y, L) 
    mini_batches.append(mini_batch)
    
    return mini_batches

def random_mini_batches_GCN1(X, X1, Y, L, mini_batch_size, seed):
    
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_X1 = X1[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    shuffled_L1 = L[permutation, :].reshape((L.shape[0], L.shape[1]), order = "F")
    shuffled_L = shuffled_L1[:, permutation].reshape((L.shape[0], L.shape[1]), order = "F")

    num_complete_minibatches = math.floor(m / mini_batch_size)
    
    for k in range(0, num_complete_minibatches):       
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_X1 = shuffled_X1[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_L = shuffled_L[k * mini_batch_size : k * mini_batch_size + mini_batch_size, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_X1, mini_batch_Y, mini_batch_L)
        mini_batches.append(mini_batch)
    mini_batch = (X, X1, Y, L) 
    mini_batches.append(mini_batch)
    
    return mini_batches
        
def random_mini_batches(X1, X2, Y, mini_batch_size, seed):
    
    m = X1.shape[0]
    m1 = X2.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X1 = X1[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    
    permutation1 = list(np.random.permutation(m1))
    shuffled_X2 = X2[permutation1, :]
    
    num_complete_minibatches = math.floor(m1/mini_batch_size)
    
    mini_batch_X1 = shuffled_X1
    mini_batch_Y = shuffled_Y
      
    for k in range(0, num_complete_minibatches):        
        mini_batch_X2 = shuffled_X2[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]        
        mini_batch = (mini_batch_X1, mini_batch_X2, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def random_mini_batches_single(X1, Y, mini_batch_size, seed):
    
    m = X1.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X1 = X1[permutation, :]
    #shuffled_X2 = X2[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
        
    for k in range(0, num_complete_minibatches):
        mini_batch_X1 = shuffled_X1[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X1, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l, dtype = bool)
    mask[idx] = True
    return mask

def create_placeholders(n_x, n_y):
   
    x_in    = tf.compat.v1.placeholder(tf.float32, [None, n_x], name="x_in")
    y_in    = tf.compat.v1.placeholder(tf.float32, [None, n_y], name="y_in")
    lap     = tf.compat.v1.placeholder(tf.float32, [None, None], name="lap")
    mask_tr = tf.compat.v1.placeholder(tf.float32, name="mask_train")
    mask_te = tf.compat.v1.placeholder(tf.float32, name="mask_test")
    return x_in, y_in, lap, mask_tr, mask_te



def initialize_parameters():
    # Use the compat.v1 API for randomness
    tf.compat.v1.set_random_seed(1)

    # Replace contrib Xavier with Keras GlorotUniform
    glorot_init = tf.keras.initializers.GlorotUniform(seed=1)

    # Note: use the compat.v1 wrappers for get_variable
    x_w1 = tf.compat.v1.get_variable(
        "x_w1", [200, 128], initializer=glorot_init)
    x_b1 = tf.compat.v1.get_variable(
        "x_b1", [128], initializer=tf.zeros_initializer())

    x_w2 = tf.compat.v1.get_variable(
        "x_w2", [128, 16], initializer=glorot_init)
    x_b2 = tf.compat.v1.get_variable(
        "x_b2", [16], initializer=tf.zeros_initializer())

    parameters = {
        "x_w1": x_w1,
        "x_b1": x_b1,
        "x_w2": x_w2,
        "x_b2": x_b2
    }
    return parameters


def GCN_layer(x_in, L_, weights):

    x_mid = tf.matmul(x_in, weights)
    x_out = tf.matmul(L_, x_mid)
    
    return x_out
    
def mynetwork(x, parameters, Lap):
    # First GCN layer + ReLU
    with tf.name_scope("x_layer_1"):
        z1 = GCN_layer(x, Lap, parameters['x_w1']) + parameters['x_b1']
        a1 = tf.nn.relu(z1)

    # Second GCN layer (logits)
    with tf.name_scope("x_layer_2"):
        z2 = GCN_layer(a1, Lap, parameters['x_w2']) + parameters['x_b2']

    l2_loss = tf.nn.l2_loss(parameters['x_w1']) + tf.nn.l2_loss(parameters['x_w2'])
    return z2, l2_loss


def mynetwork_optimaization(y_est, y_re, l2_loss, mask, reg, learning_rate, global_step):
    
    with tf.name_scope("cost"):
         cost = (tf.nn.softmax_cross_entropy_with_logits(logits = y_est, labels = y_re)) +  reg * l2_loss
         mask = tf.cast(mask, dtype = tf.float32)
         mask /= tf.reduce_mean(mask)
         cost *= mask
         cost = tf.reduce_mean(cost) +  reg * l2_loss
         
    with tf.name_scope("optimization"):
         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost,  global_step=global_step)
         optimizer = tf.group([optimizer, update_ops])
         
    return cost, optimizer

def masked_accuracy(preds, labels, mask):

      correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
      accuracy = tf.cast(correct_prediction, "float")
      mask = tf.cast(mask, dtype = tf.float32)
      mask /= tf.reduce_mean(mask)
      accuracy *= mask
      
      return tf.reduce_mean(accuracy)

def train_mynetwork(x_all, y_all, L_all, mask_in, mask_out,
                    learning_rate=0.001, beta_reg=0.001,
                    num_epochs=400, print_cost=True):

    ops.reset_default_graph()
    (m, n_x) = x_all.shape
    (_, n_y) = y_all.shape

    # Build placeholders (no isTraining)
    x_in, y_in, lap, mask_train, mask_test = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Build network
    x_out, l2_loss = mynetwork(x_in, parameters, lap)
    feature_tensor = x_out

    # Optimization & metrics
    global_step = tf.compat.v1.Variable(0, trainable=False)
    cost, optimizer = mynetwork_optimaization(
        x_out, y_in, l2_loss, mask_train, beta_reg, learning_rate, global_step)
    accuracy_train = masked_accuracy(x_out, y_in, mask_train)
    accuracy_test  = masked_accuracy(x_out, y_in, mask_test)

    # Initialize and run session
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs + 1):
            _, train_cost, train_acc = sess.run(
                [optimizer, cost, accuracy_train],
                feed_dict={
                    x_in: x_all,
                    y_in: y_all,
                    lap: L_all,
                    mask_train: mask_in,
                    mask_test: mask_out
                })
            if print_cost and epoch % 50 == 0:
                val_cost, val_accu = sess.run(
                    [cost, accuracy_test],
                    feed_dict={
                        x_in: x_all,
                        y_in: y_all,
                        lap: L_all,
                        mask_train: mask_in,
                        mask_test: mask_out
                    })
                print(f"epoch {epoch}: Train_loss={train_cost:.4f}, Val_loss={val_cost:.4f}, Train_acc={train_acc:.4f}, Val_acc={val_accu:.4f}")

        # (plotting code unchanged)
        features_all = sess.run(feature_tensor,
                                feed_dict={
                                    x_in: x_all,
                                    y_in: y_all,
                                    lap:  L_all,
                                    mask_train: mask_in,
                                    mask_test:  mask_out
                                })
        params = sess.run(parameters)
        return params, None, features_all  # return features if you compute them



ALL_X = scio.loadmat('/kaggle/input/gcn-try/GCN/ALL_X.mat')
ALL_Y = scio.loadmat('/kaggle/input/gcn-try/GCN/ALL_Y.mat')
ALL_L = scio.loadmat('/kaggle/input/gcn-try/GCN/ALL_L.mat')

ALL_L = ALL_L['ALL_L']
ALL_X = ALL_X['ALL_X']
ALL_Y = ALL_Y['ALL_Y']

GCN_mask_TR = sample_mask(np.arange(0,695), ALL_Y.shape[0])
GCN_mask_TE = sample_mask(np.arange(696,10366), ALL_Y.shape[0])

ALL_Y = convert_to_one_hot(ALL_Y - 1, 16)
ALL_Y = ALL_Y.T


parameters, val_acc, features = train_mynetwork(ALL_X, ALL_Y, ALL_L.todense(), GCN_mask_TR, GCN_mask_TE)
sio.savemat('features.mat', {'features': features})
