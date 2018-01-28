import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle

with open('model/mnist_feature.pickle', 'rb') as f:
        x_test = pickle.load(f)

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
# batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases': tf.Variable(tf.random_normal([n_classes]))}

	# (data * weights) + biases
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output


def predict_num():
	prediction = neural_network_model(x)
	print('Model init')
	#cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
	
	# learning_rate = 0.001
	#optimiser = tf.train.AdamOptimizer().minimize(cost)
	
	# cycles feed forward + backpropagation
	hm_epochs = 1
	
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		saver = tf.train.Saver()
		saver.restore(sess, "/Pro/tf/model/mnist_model.ckpt")
		


		# print(prediction.eval({x:x_test[0]})) 
		print(tf.argmax(prediction, 1).eval({x: [x_test[0][0]]}), tf.argmax(y, 1).eval({y:[x_test[1][0]]})) 
		# print(tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)).eval( {x:[epoch_x[0]], y: [epoch_y[0]]}))
		# print(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)), 'float').eval( {x:[epoch_x[0]], y: [epoch_y[0]]}))
		# print(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)), 'float')).eval( {x:[epoch_x[0]], y: [epoch_y[0]]}))
		# print(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)), 'float').eval( {x:mnist.test.images, y: mnist.test.labels}))
		
		# correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		# accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		# print('Accuracy: ', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

predict_num()


