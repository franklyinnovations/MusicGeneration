import tensorflow as tf
import numpy as np

def encoder(x, weights, bias):
	layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), bias['encoder_b1']))
	layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']), bias['encoder_b2']))
	
	return layer_2

def decoder(x, weights, bias):
	layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), bias['decoder_b1']))
	layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']), bias['decoder_b2']))

	return layer_2

def preprocess(data):
	min_length = np.inf
	for sequence in data:
		if len(sequence) < min_length:
			min_length = len(sequence)

	train_X = []
	means = []
	stds = []
	for sequence in data:
		temp = np.array(sequence[0:min_length])
		mean = np.mean(temp)
		std = np.std(temp)
		temp = (temp - mean) / std
		temp = temp.reshape(1, min_length)
		train_X.append(temp)
		means.append(mean)
		stds.append(std)

	return np.array(train_X), np.array(means), np.array(stds), min_length

def postprocess(data, mean, std):
	return np.round(data * std + mean)

def generate_random_seed(dimension):
	seed = np.random.normal(0, 1, dimension)
	seed.astype(np.float64)
	return seed

def autoencoder_melody(right, epochs=5):
	train_X, train_means, train_stds, train_length = preprocess(right[0:35])
	test_X, test_means, test_stds, test_length = preprocess(right[35:40])

	learning_rate = 0.02
	num_hidden1 = 128
	num_hidden2 = 64
	num_input = train_length
	weights = {
		'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden1])),
		'encoder_h2': tf.Variable(tf.random_normal([num_hidden1, num_hidden2])),
		'decoder_h1': tf.Variable(tf.random_normal([num_hidden2, num_hidden1])),
		'decoder_h2': tf.Variable(tf.random_normal([num_hidden1, num_input])),
	}	
	bias = {
		'encoder_b1': tf.Variable(tf.random_normal([num_hidden1])),
		'encoder_b2': tf.Variable(tf.random_normal([num_hidden2])),
		'decoder_b1': tf.Variable(tf.random_normal([num_hidden1])),
		'decoder_b2': tf.Variable(tf.random_normal([num_input])),	
	}
	X = tf.placeholder("float", [None, num_input])

	encoder_op = encoder(X, weights, bias)
	decoder_op = decoder(encoder_op, weights, bias)

	y_pred = decoder_op
	y_true = X

	loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for e in range(epochs):
			curr_loss = 0
			permutations = np.random.permutation(len(train_X))
			for index in permutations:
				o, l = sess.run([optimizer, loss], feed_dict={X: train_X[index]})
				curr_loss += (l/len(train_X))
			print('Epoch ' + str(e) + ' training loss: ' + str(curr_loss))

		output = []
		for i in range(len(test_X)):
			truncate = test_X[i][0]
			truncate = truncate[0:train_length]
			truncate = truncate.reshape(1, train_length)
			encoder_decoder_sequence = sess.run(y_pred, feed_dict={X: truncate})
			output.append(postprocess(encoder_decoder_sequence, test_means[i], test_stds[i]))
		return output
		


