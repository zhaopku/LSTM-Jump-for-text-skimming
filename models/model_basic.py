import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell

class ModelBasic:
	def __init__(self, args, textData, initializer=None):
		print('Creating single lstm Model')
		self.args = args
		self.textData = textData

		self.dropOutRate = None
		self.initial_state = None
		self.learning_rate = None
		self.loss = None
		self.optOp = None
		self.labels = None
		self.input = None
		self.target = None
		self.length = None
		self.embedded = None
		self.predictions = None
		self.batch_size = None
		self.corrects = None
		self.initializer = initializer


		self.v0 = None
		self.v1 = None
		self.v2 = None
		self.v3 = None
		self.v4 = None
		self.v5 = None
		self.v6 = None
		self.v7 = None

		self.buildNetwork()

	def buildNetwork(self):
		with tf.name_scope('rnn'):
			# [batchSize, hiddenSize]
			outputs, skimgates = self.buildRNN()

		with tf.name_scope('output'):
			weights = tf.get_variable(name='weights', shape=[self.args.hiddenSize, self.args.numClasses],
									  initializer=self.initializer)

			biases = tf.get_variable(name='biases', shape=[self.args.numClasses],
			                         initializer=self.initializer)
			# [batchSize, numClasses]
			logits = tf.nn.xw_plus_b(x=outputs, weights=weights, biases=biases)
		with tf.name_scope('predictions'):
			# [batchSize]
			self.predictions = tf.argmax(logits, axis=-1, name='predictions', output_type=tf.int32)
			# single number
			self.corrects = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.labels), tf.int32), name='corrects')

		with tf.name_scope('loss'):
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='loss')

			self.loss = tf.reduce_sum(loss)

		with tf.name_scope('backpropagation'):
			opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
											   epsilon=1e-08)
			self.optOp = opt.minimize(self.loss)

	def buildRNN(self):
		with tf.name_scope('placeholders'):
			# [batchSize, maxSteps]
			input_shape = [None, self.args.maxSteps]
			self.input = tf.placeholder(tf.int32, shape=input_shape, name='input')
			self.labels = tf.placeholder(tf.int32, shape=[None,], name='labels')
			self.length = tf.placeholder(tf.int32, shape=[None,], name='length')
			self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')

			self.dropOutRate = tf.placeholder(tf.float32, (), name='dropOut')

		with tf.name_scope('embedding_layer'):
			if not self.args.preEmbedding:
				print('Using randomly initialized embeddings!')
				embeddings = tf.get_variable(
					shape=[self.textData.getVocabularySize(), self.args.embeddingSize],
					initializer=tf.contrib.layers.xavier_initializer(),
					name='embeddings')
			else:
				print('Using pretrained word embeddings!')
				embeddings = tf.Variable(self.textData.preTrainedEmbedding, name='embedding', dtype=tf.float32)

			# [batchSize, maxSteps, embeddingSize]
			self.embedded = tf.nn.embedding_lookup(embeddings, self.input)
			self.embedded = tf.nn.dropout(self.embedded, self.dropOutRate, name='embedding_dropout')

		with tf.name_scope('lstm'):
			with tf.variable_scope('cell', reuse=False):

				def get_cell(hiddenSize, dropOutRate):
					print('building ordinary cell!')
					cell = BasicLSTMCell(num_units=hiddenSize, state_is_tuple=True)
					cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropOutRate,
															 output_keep_prob=dropOutRate)
					return cell

				# https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn

				cell = get_cell(self.args.hiddenSize, self.dropOutRate)

				# multiCell = []
				# for i in range(self.args.rnnLayers):
				# 	multiCell.append(get_cell(self.args.hiddenSize, self.dropOutRate))
				# multiCell = tf.contrib.rnn.MultiRNNCell(multiCell, state_is_tuple=True)

			# [batchSize, maxSteps, hiddenSize]
			state = cell.zero_state(self.batch_size, tf.float32)

			outputs = []
			# TODO: remember the length of each sentence
			with tf.variable_scope("loop", reuse=tf.AUTO_REUSE):
				for time_step in range(self.args.maxSteps):
					# [batch_size, hidden_size]
					# [batch_size, 1]
					(cell_output, state) = cell(self.embedded[:, time_step, :], state)
					outputs.append(cell_output)

			# [maxSteps, batchSize, hiddenSize]
			outputs = tf.stack(outputs)
			# [batchSize, maxSteps, hiddenSize]
			outputs = tf.transpose(outputs, [1, 0, 2], name='outputs')

			# [batchSize, maxSteps]
			last_relevant_mask = tf.one_hot(indices=self.length-1, depth=self.args.maxSteps, name='last_relevant',
											dtype=tf.int32)
			# [batchSize, hiddenSize]
			last_relevant_outputs = tf.boolean_mask(outputs, last_relevant_mask, name='last_relevant_outputs')

			# [batchSize, maxSteps]
			self.skimgates = tf.zeros([self.batch_size, self.args.maxSteps], name='skimgates')

		return last_relevant_outputs, self.skimgates

	def step(self, batch, test=False):
		feed_dict = {}

		# [batchSize, maxSteps]
		input_ = []
		length = []
		labels = []

		for sample in batch.samples:
			input_.append(sample.input_)
			labels.append(sample.label)
			length.append(sample.length)

		feed_dict[self.labels] = labels
		feed_dict[self.input] = input_
		feed_dict[self.length] = length
		feed_dict[self.batch_size] = len(length)

		if not test:
			feed_dict[self.dropOutRate] = self.args.dropOut
			ops = (self.optOp, self.loss, self.predictions, self.corrects, self.skimgates)
		else:
			# during test, do not use drop out!!!!
			feed_dict[self.dropOutRate] = 1.0
			ops = (self.loss, self.predictions, self.corrects, self.skimgates)

		return ops, feed_dict, length
