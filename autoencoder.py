import tensorflow as tf
import numpy as np

def encoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf,matmul(x, weights['encoder_h1']), bias['encoder_b1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), bias['encoder_b2']))
	
	return layer_2

def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf,matmul(x, weights['decoder_h1']), bias['decoder_b1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), bias['decoder_b2']))

	return layer_2

def autoencoder_melody(right, epochs=5):
	min_length = np.inf
	for sequence in right:
		if len(sequence) < min_length:
			min_length = len(sequence)

	train_X = []
	for sequence in right:
		train_X.append(sequence[0:min_length])

	learning_rate = 0.01
	num_hidden1 = 256
	num_hidden2 = 128
	num_input = min_length

	X = tf.placeholder("float", [None, num_input])
	weights = {
		'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden1])),
		'encoder_h2': tf.Variable(tf.random_normal([num_hidden1, num_hidden2])),
		'decoder_h1': tf.Variable(tf.random_normal([num_hidden2, num_hidden1])),
		'decoder_h1': tf.Variable(tf.random_normal([num_hidden1, num_input])),
	}
	bias = {
		'encoder_h1': tf.Variable(tf.random_normal([num_hidden1])),
		'encoder_h2': tf.Variable(tf.random_normal([num_hidden2])),
		'decoder_h1': tf.Variable(tf.random_normal([num_hidden2])),
		'decoder_h1': tf.Variable(tf.random_normal([num_input])),	
	}

	encoder_op = encoder(X)
	decoder_op = decoder(encoder_op)

	y_pred = decoder_op
	y_true = X

	loss = tf.reduce_mean(tf.pow(y_true - y_pred), 2)
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		curr_loss = 0
		for e in range(epochs):
			permutations = np.random.permutation(len(train_X))
			for index in permutations:
				o, l = sess.run([optimizer, loss], feed_dict={X: train_X[index]})
				curr_loss += l
			print (curr_loss)

