# inference.py
import os
import scipy.io as scio
import numpy as np
import tensorflow as tf
from tf_utils import random_mini_batches_GCN1, convert_to_one_hot  # if you need any utils

# ---- 1. Re‑define all the same graph-building functions ----

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

# copy your initialize_parameters, GCN_layer, mynetwork definitions here...
# (For brevity I’m assuming you’ve imported them or pasted them exactly.)

# ---- 2. Build graph and Saver ----

tf.compat.v1.disable_eager_execution()

# Load one sample to get dimensions
sample = scio.loadmat('Test_X.mat')
X_test  = sample['Test_X']    # conv‑branch input
test_g  = sample['Test_X']    # just as a placeholder; we'll reorder below

# Actually load everything you need to predict:
X_train_mat = scio.loadmat('X_train.mat')
Train_X = X_train_mat['Train_X']    # GCN input dimension
_, n_x1 = sample['Test_X'].shape    # conv feature dim
_, n_x  = Train_X.shape
# If you have labels for test, load them (or use zeros if not):
TeLabel = scio.loadmat('TeLabel.mat')['TeLabel']
_, n_y = TeLabel.shape

# Placeholders
x_in, x_in1, y_in, lap_ph, isTraining = create_placeholders(n_x, n_x1, n_y)

# Parameters & forward‑prop
parameters = initialize_parameters()
logits, _ = mynetwork(x_in, x_in1, parameters, lap_ph, isTraining)

# If you want class‑ids:
pred_ids = tf.argmax(logits, axis=1, name="predictions")

# Saver (saves/restores all vars in graph)
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    # 3. Restore your trained weights
    ckpt = tf.train.latest_checkpoint('./checkpoints')
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found in ./checkpoints")
    saver.restore(sess, ckpt)
    print(f"Restored model from {ckpt}")

    # 4. Load test data
    X_test_mat = scio.loadmat('Test_X.mat')
    Test_X  = X_test_mat['Test_X']
    Test_L  = scio.loadmat('Test_L.mat')['Test_L']
    TeLabel  = scio.loadmat('TeLabel.mat')['TeLabel']

    # 5. Run inference
    feed = {
        x_in:  scio.loadmat('Train_X.mat')['Train_X'],
        x_in1: Test_X,
        y_in:  TeLabel,
        lap_ph: scio.loadmat('Train_L.mat')['Train_L'],
        isTraining: False
    }
    preds = sess.run(pred_ids, feed_dict=feed)

    # 6. Save or print
    scio.savemat('predicted_labels.mat', {'pred_ids': preds})
    print("Inference done – saved to predicted_labels.mat")
