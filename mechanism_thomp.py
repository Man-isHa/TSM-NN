import tensorflow as tf
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

	

n_arms = 2
M = 50
means = [0.87, 0.9]
trials = 100000
iterations = 1000


## Network parameters ##
inp_dim = 2*n_arms - 1
hidden_dim = 10
rho = 100
learning_rate = 0.01
batch_size = 1000



def dense(inp, inp_shape, hidden_size, act=0, name ='dense'):
	with tf.variable_scope(name):
		weights = tf.get_variable("weights", [inp_shape, hidden_size], 'float32',initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
		bias = tf.get_variable("bias", [hidden_size], 'float32', initializer=tf.constant_initializer(0.01, dtype=tf.float32))
		out = tf.matmul(inp, weights) + bias
		if act == 1:
			return tf.nn.relu(out)
		else:
			return out


## Place holders
input_data = tf.placeholder(tf.float32, shape=(batch_size, inp_dim), name="data_2")
bid_tensor = tf.placeholder(tf.float32, shape=(batch_size,), name="label_2")
M_t = tf.placeholder(tf.float32, shape=(), name='m')

## Forward Pass
fc1_act = dense(input_data, inp_dim, hidden_dim, act=1)
output = dense(fc1_act, hidden_dim, 1, name = 'dense1')

### Loss + constraints
IR = tf.maximum(-(output[:,0] - bid_tensor), 0.0)
IR_auc = tf.maximum(-(M_t - output[:,0]), 0.0)
loss = tf.reduce_mean(output[:,0] + rho * IR**2 + rho/100* (IR_auc)**2)

### Optimizer
t_vars = tf.trainable_variables()
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list = t_vars)


### New Session 
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

total_bids = np.random.uniform(0,M, size=(trials, n_arms))



##### Training Code ###########
batch_inp = []
batch_bid = []
welfare = []
regret = np.zeros((iterations, trials))

for epoch in range(iterations):
	alpha = np.ones(n_arms)
	beta = np.ones(n_arms)
	welf_  = 0
	regret_t = 0
	bids = total_bids[epoch]
	opt_arm = np.argmax(M*np.array(means) - bids)
	for t in range(1,trials+1):
			reward = []
			for i in range(n_arms):
				reward.append(np.random.beta(alpha[i], beta[i]))
			arm_selected = np.argmax(M*np.array(reward) - bids)
			x = np.random.binomial(1, means[arm_selected])
			if x == 1:
				alpha[arm_selected] += 1
			else:
				beta[arm_selected] += 1
			batch_inp.append(np.append(M*np.array(reward), bids[np.arange(n_arms)!=arm_selected]))
			batch_bid.append(bids[arm_selected])
			if t%batch_size == 0 :
				batch_inp = np.stack(batch_inp)
				batch_bid = np.stack(batch_bid)
				dict1 = {input_data:batch_inp, M_t:M, bid_tensor:batch_bid}
				_, payment_t, l = sess.run([opt, output, loss], feed_dict = dict1)
				batch_inp = []
				batch_bid = []
				################################################################################
			regret_t += M*means[opt_arm] - bids[opt_arm] - M*means[arm_selected] + bids[arm_selected]
			regret[epoch, t-1] = regret_t
	print(epoch, " over")
	#np.save('Regret', np.mean(regret, axis=0))
print(regret_t, ':  is the final regret after training')


#### Testing Code #######
utility = np.zeros((iterations, trials, n_arms))
payment = np.zeros((iterations, trials))
batch_inp = []
batch_bid = []
opt_arm = np.argmax(M*np.array(means) - bids)
bids = np.array([30.0, 35.0])
events = np.zeros((trials,n_arms))
events[:,0] = np.random.binomial(1, means[0], size=trials)
events[:,1] = np.random.binomial(1, means[1], size=trials)
for itr in range(iterations):
	alpha = np.ones(n_arms)
	beta = np.ones(n_arms)
	arms_pulled  = []
	for t in range(1, trials+1):
		reward = []
		for i in range(n_arms):
			reward.append(np.random.beta(alpha[i], beta[i]))
		arm_selected = np.argmax(M*np.array(reward) - bids)
		#x = np.random.binomial(1, means[arm_selected])
		x = events[t-1, arm_selected]
		if x == 1:
			alpha[arm_selected] += 1
		else:
			beta[arm_selected] += 1
		
		################################################################################
		batch_inp.append(np.append(M*np.array(reward), bids[np.arange(n_arms)!=arm_selected]))
		batch_bid.append(bids[arm_selected])
		arms_pulled.append(arm_selected)
		if t%batch_size == 0 :
			batch_inp = np.stack(batch_inp)
			batch_bid = np.stack(batch_bid)
			dict1 = {input_data:batch_inp, bid_tensor:batch_bid}
			payment_t = sess.run(output, feed_dict = dict1)
			utility[itr, t-batch_size:t, arms_pulled] = payment_t.reshape(batch_size) - batch_bid
			payment[itr, t-batch_size:t] = payment_t.reshape(batch_size)
			batch_inp = []
			batch_bid = []
			arms_pulled = []
	print('Iteration ' + str(itr) + ' over')	

plt.plot(np.var(utility[:,:,opt_arm], axis = 0))
plt.savefig('test_util_nn.png')
np.save('./results/unn2', utility)
plt.close()

plt.plot(np.mean(payment, axis=0))
plt.savefig('test_payment_nn.png')
np.save('payment_nn', payment)
plt.close()

# sample_bids = [np.array([30.0, 30.0]), np.array([30.0, 20.0]), np.array([25.0, 50.0]), np.array([1.0, 45.0])]
# ctr = 1
# for bids in sample_bids:
# 	utility = np.zeros((iterations, trials, n_arms))
# 	payment = np.zeros((iterations, trials))
# 	batch_inp = []
# 	batch_bid = []
# 	opt_arm = np.argmax(M*np.array(means) - bids)
# 	if ctr ==2: ctr = 3
# 	for itr in range(iterations):
# 		alpha = np.ones(n_arms)
# 		beta = np.ones(n_arms)
# 		arms_pulled  = []
# 		for t in range(1, trials+1):
# 			reward = []
# 			for i in range(n_arms):
# 				reward.append(np.random.beta(alpha[i], beta[i]))
# 			arm_selected = np.argmax(M*np.array(reward) - bids)
# 			x = np.random.binomial(1, means[arm_selected])
# 			if x == 1:
# 				alpha[arm_selected] += 1
# 			else:
# 				beta[arm_selected] += 1
			
# 			################################################################################
# 			batch_inp.append(np.append(M*np.array(reward), bids[np.arange(n_arms)!=arm_selected]))
# 			batch_bid.append(bids[arm_selected])
# 			arms_pulled.append(arm_selected)
# 			if t%1000 == 0 :
# 				batch_inp = np.stack(batch_inp)
# 				batch_bid = np.stack(batch_bid)
# 				dict1 = {input_data:batch_inp, bid_tensor:batch_bid}
# 				payment_t = sess.run(output, feed_dict = dict1)
# 				utility[itr, t-1000:t, arms_pulled] = payment_t.reshape(1000) - batch_bid
# 				payment[itr, t-1000:t] = payment_t.reshape(1000)
# 				batch_inp = []
# 				batch_bid = []
# 				arms_pulled = []
# 		print('Iteration ' + str(itr) + '_' + str(ctr) + ' over')	

# 	plt.plot(np.var(utility[:,:,opt_arm], axis = 0))
# 	plt.savefig('test_util_nn'+str(ctr)+'.png')
# 	np.save('./results/unn'+str(ctr), utility)
# 	plt.close()
# 	ctr += 1
# 	# plt.plot(np.mean(payment, axis=0))
# 	# plt.savefig('test_payment_nn.png')
# 	# np.save('payment_nn', payment)
# 	# plt.close()


