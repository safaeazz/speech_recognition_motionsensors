import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# kl divergence  for sparsity regularization

def kl_divergence(p,q):
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q)) 

# weight init

def get_weight_variable(shape, name=None,type='xavier_uniform'):

    if type == 'xavier_uniform':
        initial = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
    elif type == 'xavier_normal':
        initial = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
    elif type == 'he_normal':
        initial = tf.contrib.layers.variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'he_uniform':
        initial = tf.contrib.layers.variance_scaling_initializer(uniform=True, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'caffe_uniform':
        initial = tf.contrib.layers.variance_scaling_initializer(uniform=True, factor=1.0, mode='FAN_IN', dtype=tf.float32)

    weight = tf.get_variable(name,shape=shape, initializer=initial)

    return weight 

def create_variables1(n_features,dim,weightinit,nameen,namedec):

    w_en = get_weight_variable([n_features, dim], nameen,weightinit)
    w_dec = get_weight_variable([dim, n_features], namedec,weightinit)
    #W21 = tf.transpose(W12,name='w1-dec')# tied weights
    return w_en,w_dec

def create_variables2(n_features,dim,nb1,nb2):

    b1 = tf.Variable(tf.random_normal([dim]),name=nb1)
    b2 = tf.Variable(tf.random_normal([n_features]),name=nb2)

    return b1,b2
     
# Encoding-decoding layers 
#add depth, rbm initialization

def encoder(input_x,enc_act_func,w_en,b_en):

    if enc_act_func == 'Relu':
        encode = tf.nn.elu(tf.add(tf.matmul(input_x, w_en),b_en))  
    elif enc_act_func == 'sigmoid':
        encode = tf.nn.sigmoid(tf.add(tf.matmul(input_x, w_en),b_en)) 
    elif enc_act_func == 'tanh':
        encode = tf.nn.tanh(tf.add(tf.matmul(input_x, w_en),b_en))     

    else:
        encode = None

    return encode

def encode_SSAE(input_x,w1,w2,b1,b2):

    layer1 = tf.nn.elu(tf.add(tf.matmul(input_x, w1),b1))
    layer2 = tf.nn.elu(tf.add(tf.matmul(layer1, w2),b2))
    #encode = tf.nn.elu(tf.add(tf.matmul(layer2, w3),b3))

    return layer2

def decoder(encode,dec_act_func,w_dec,b_dec):

    if dec_act_func == 'sigmoid':
        decode = tf.nn.sigmoid(tf.add(tf.matmul(encode, w_dec),b_dec))
    elif dec_act_func == 'tanh':
        decode = tf.nn.tanh(tf.add(tf.matmul(encode, w_dec),b_dec))    
    elif dec_act_func == 'Relu':
        decode = tf.nn.elu(tf.add(tf.matmul(encode, w_dec),b_dec))
    elif dec_act_func == 'none':
        decode = tf.nn.linear(tf.add(tf.matmul(encode, w_dec),b_dec))
    else:
        decode = None
    return decode


# for two layers:


def create_variables11(n_features,dim1,dim2,weightinit):

    W11 = get_weight_variable([n_features, dim1], 'w1-en', weightinit)
    W12 = get_weight_variable([dim1, dim2], 'w2-en', weightinit)
    W21 = get_weight_variable([dim2, dim1], 'w1-dec', weightinit)
    W22 = get_weight_variable([dim1, n_features], 'w2-dec', weightinit)
    #W21 = tf.transpose(W12,name='w1-dec')# tied weights
    #W22 = tf.transpose(W11,name='w2-dec')

    return W11,W12,W21,W22

def create_variables22(n_features,dim1,dim2):

  bh1 = tf.Variable(tf.random_normal([dim1]),name='b1-en')
  bh2 = tf.Variable(tf.random_normal([dim2]),name='b2-en')
  bv1 = tf.Variable(tf.random_normal([dim1]),name='b1-dec')
  bv2 = tf.Variable(tf.random_normal([n_features]),name='b2-dec')

  return bh1,bh2,bv1,bv2

     
# Encoding-decoding layers 
#add depth, rbm initialization

def encoder1(input,enc_act_func,W1,W2,bh1,bh2):

    if enc_act_func == 'Relu':
        layer1 = tf.nn.relu(tf.add(tf.matmul(input, W1),bh1))  
        encode = tf.nn.relu(tf.add(tf.matmul(layer1, W2),bh2))

    elif enc_act_func == 'sigmoid':
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(input, W1),bh1))
        encode = tf.nn.sigmoid(tf.add(tf.matmul(layer1, W2),bh2))

    elif enc_act_func == 'tanh':
        layer1 = tf.nn.tanh(tf.add(tf.matmul(input, W1),bh1))   
        encode = tf.nn.tanh(tf.add(tf.matmul(layer1, W2),bh2))

    else:
        layer1 = tf.nn.linear(tf.add(tf.matmul(input, W1),bh1))   
        encode = tf.nn.linear(tf.add(tf.matmul(layer1, W2),bh2))        
    return encode

def decoder1(encode,dec_act_func,W1,W2,bv1,bv2):

    if dec_act_func == 'sigmoid':
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(encode, W1),bv1))
        decode = tf.nn.sigmoid(tf.add(tf.matmul(layer1, W2),bv2))

    elif dec_act_func == 'tanh':
        layer1 = tf.nn.tanh(tf.add(tf.matmul(encode, W1),bv1))
        decode = tf.nn.tanh(tf.add(tf.matmul(layer1, W2),bv2))  
    elif dec_act_func == 'Relu':
        layer1 = tf.nn.relu(tf.add(tf.matmul(encode, W1),bv1))
        decode = tf.nn.relu(tf.add(tf.matmul(layer1, W2),bv2))
    elif dec_act_func == 'linear':
        layer1 = tf.nn.linear(tf.add(tf.matmul(encode, W1),bv1))
        decode = tf.nn.linear(tf.add(tf.matmul(layer1, W2),bv2))

    return decode


def cost_function(loss_func,input_x,decode):
 
    if loss_func == 'entropy':
        cost = - tf.reduce_sum(input_x * tf.log(decode))
        #_ = tf.summary.scalar("cross_entropy", cost)
    elif loss_func == 'mse':
        cost = tf.sqrt(tf.reduce_mean(tf.square(input_x - decode)))
        #tf.summary.scalar("mean_squared", cost)

    return cost

def optimize_alg(opt,learning_rate,loss,momentum):
    
    if opt == 'SGD':
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt == 'adagrad':
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt == 'moment':
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
    elif opt == 'Rmsprop':
        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    elif opt == 'adam':
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step


def gen_batches(data, batch_size):

    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]


# def save the model+ restore the model+ get the model parameters

def dense_to_one_hot(labels_dense, num_classes,num_labels):

    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def one_hot(labels1):

    labels= np.array(labels1)

    onehot_encoder = OneHotEncoder(sparse=False)
    labels = labels.reshape(len(labels), 1)
    onehot_encoded = onehot_encoder.fit_transform(labels)
    return onehot_encoded


def masking_noise(X, v):

    X_noise = X.copy()

    n_samples = X.shape[0]
    n_features = X.shape[1]

    for i in range(n_samples):
        mask = np.random.randint(0, n_features, v)

        for m in mask:
            X_noise[i][m] = 0.

    return X_noise
def salt_and_pepper_noise(X, v):

    X_noise = X.copy()
    n_features = X.shape[1]

    mn = X.min()
    mx = X.max()

    for i, sample in enumerate(X):
        mask = np.random.randint(0, n_features, v)

        for m in mask:

            if np.random.random() < 0.5:
                X_noise[i][m] = mn
            else:
                X_noise[i][m] = mx

    return X_noise

def corrupt_input(data, v):

    #x_corrupted = salt_and_pepper_noise(data, v)
    x_corrupted = masking_noise(data, v)
    return x_corrupted 