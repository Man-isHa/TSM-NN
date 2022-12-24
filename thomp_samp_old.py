                            
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf



gh_dim = 500
gh_dim1 = 100
dh_dim = 100
num_samps = 100

# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

# G(z)
def generator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, gh_dim, kernel_initializer=w_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(dense1, training=isTrain), 0.2)
        #relu1 = tf.nn.relu(dense1)
        dense2 = tf.layers.dense(lrelu1, gh_dim1, kernel_initializer=w_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(dense2, training=isTrain), 0.2)

        dense3 = tf.layers.dense(lrelu2, num_samps, kernel_initializer=w_init)
        theta = tf.nn.sigmoid(dense3)

        return theta

# D(x)
def discriminator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, gh_dim, kernel_initializer=w_init)
        lrelu1 = lrelu(dense1, 0.2)

        dense2 = tf.layers.dense(lrelu1, dh_dim, kernel_initializer=w_init)
        lrelu2 = lrelu(dense2, 0.2)

        dense3 = tf.layers.dense(lrelu2, 1, kernel_initializer=w_init)
        o = tf.nn.sigmoid(dense3)

        return o, dense3



# training parameters
batch_size = 1000
lr = 0.0002
train_epoch = 1000
num_cats = 100

# load DATA
alpha = np.random.randint(1, 100, num_cats)
beta = np.random.randint(1, 100, num_cats)
#alpha = np.ones(num_cats)*2
#beta = np.ones(num_cats)*2

# variables : input
x = tf.placeholder(tf.float32, shape=(None, num_samps))
y = tf.placeholder(tf.float32, shape=(None, 2))
z = tf.placeholder(tf.float32, shape=(None, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, y, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, y, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()



train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(1000):
        # update discriminator
        ix = np.random.randint(num_cats, size=batch_size) 
        y_ = np.transpose(np.vstack((alpha[ix], beta[ix])))
        x_ = []
        for i in range(batch_size):
            x_.append(np.random.beta(y_[i,0], y_[i,1],  num_samps))
        x_ = np.array(x_)
        z_ = np.random.normal(0, 1, (batch_size, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, y: y_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        ix = np.random.randint(num_cats, size=batch_size) 
        y_ = np.transpose(np.vstack((alpha[ix], beta[ix])))
        x_ = []
        for i in range(batch_size):
            x_.append(np.random.beta(y_[i,0], y_[i,1],  num_samps))
        x_ = np.array(x_)
        z_ = np.random.normal(0, 1, (batch_size, 100))
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, y: y_, isTrain: True})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

plt.plot(train_hist['D_losses'])
plt.plot(train_hist['G_losses'])
plt.savefig('Losses.png')
plt.close()


### Testing ###
print('testing start')
y_ = np.array([[2, 2]]*batch_size)
z_ = np.random.normal(0, 1, (batch_size, 100))
gen_samples = sess.run(G_z, {z: z_, y: y_, isTrain: False})
plt.hist(gen_samples.flatten(), 100, density=True)
plt.savefig('Generated_dist.png')
