import os
import scipy.io as scio
import numpy as np
import tensorflow as tf
from tf_utils import random_mini_batches_GCN1, convert_to_one_hot  


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

def create_placeholders(n_x, n_x1, n_y):
    isTraining = tf.compat.v1.placeholder_with_default(False, shape=(), name="isTraining")
    x_in  = tf.compat.v1.placeholder(tf.float32, [None, n_x],  name="x_in")
    x_in1 = tf.compat.v1.placeholder(tf.float32, [None, n_x1], name="x_in1")
    y_in  = tf.compat.v1.placeholder(tf.float32, [None, n_y],  name="y_in")
    lap   = tf.compat.v1.placeholder(tf.float32, [None, None], name="lap_train")
    return x_in, x_in1, y_in, lap, isTraining


tf.compat.v1.disable_eager_execution()


sample = scio.loadmat('Test_X.mat')
X_test  = sample['Test_X']   
test_g  = sample['Test_X']    

X_train_mat = scio.loadmat('X_train.mat')
Train_X = X_train_mat['Train_X']   
_, n_x1 = sample['Test_X'].shape   
_, n_x  = Train_X.shape

TeLabel = scio.loadmat('TeLabel.mat')['TeLabel']
_, n_y = TeLabel.shape


x_in, x_in1, y_in, lap_ph, isTraining = create_placeholders(n_x, n_x1, n_y)

parameters = initialize_parameters()
logits, _ = mynetwork(x_in, x_in1, parameters, lap_ph, isTraining)

pred_ids = tf.argmax(logits, axis=1, name="predictions")

saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    
    ckpt = tf.train.latest_checkpoint('./checkpoints')
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found in ./checkpoints")
    saver.restore(sess, ckpt)
    print(f"Restored model from {ckpt}")


    X_test_mat = scio.loadmat('Test_X.mat')
    Test_X  = X_test_mat['Test_X']
    Test_L  = scio.loadmat('Test_L.mat')['Test_L']
    TeLabel  = scio.loadmat('TeLabel.mat')['TeLabel']

 
    feed = {
        x_in:  scio.loadmat('Train_X.mat')['Train_X'],
        x_in1: Test_X,
        y_in:  TeLabel,
        lap_ph: scio.loadmat('Train_L.mat')['Train_L'],
        isTraining: False
    }
    preds = sess.run(pred_ids, feed_dict=feed)

    scio.savemat('predicted_labels.mat', {'pred_ids': preds})
    print("Inference done – saved to predicted_labels.mat")
