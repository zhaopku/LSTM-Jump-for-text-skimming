import tensorflow as tf
from models.utils import SkipLSTMStateTuple
from tensorflow.contrib.rnn import BasicLSTMCell
from models.utils import SkipLSTMCell

class Model:
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
		self.is_training = None
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



	def getInputs(self):
		with tf.name_scope('placeholders'):
			# [batchSize, maxSteps]
			input_shape = [None, self.args.maxSteps]
			self.input = tf.placeholder(tf.int32, shape=input_shape, name='input')
			self.labels = tf.placeholder(tf.int32, shape=[None,], name='labels')
			self.length = tf.placeholder(tf.int32, shape=[None,], name='length')
			self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')
			self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
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

	def buildNetwork(self):
		with tf.name_scope('inputs'):
			self.getInputs()

		with tf.variable_scope('sampling', reuse=tf.AUTO_REUSE):
			"""
			sample from the rnn for nSample times
			"""
			ce_loss = []
			rewards = []
			n_skips = []
			probs = []
			valid = []

			self.n_corrects = 0

			for __ in range(self.args.nSamples):
				# ce, r: [batch_size]
				# n, p, v: [batch_size, maxSteps]
				ce, r, n, p, v, n_corrects = self.get_loss_and_rewards()
				self.n_corrects += n_corrects
				ce_loss.append(ce)
				rewards.append(r)
				n_skips.append(n)
				probs.append(p)
				valid.append(v)

		with tf.name_scope('post_sampling'):

			# [batch_size, nSamples]
			ce_loss = tf.transpose(tf.stack(ce_loss), [1, 0], name='ce_loss')
			rewards = tf.transpose(tf.stack(rewards), [1, 0], name='rewards')

			# [nSamples, batch_size, maxSteps]
			probs = tf.stack(probs)
			valid = tf.stack(valid)
			n_skips = tf.stack(n_skips)
			# [batch_size, nSamples, maxSteps]
			probs = tf.add(tf.transpose(probs, [1, 0, 2]), 1e-5, name='probs')
			valid = tf.cast(tf.transpose(valid, [1, 0, 2]), tf.float32)
			n_skips = tf.transpose(n_skips, [1, 0, 2], name='n_skips')

			# mask out steps exceeding the length of each sample
			length = tf.tile(tf.expand_dims(self.length, axis=1), multiples=[1, self.args.nSamples], name='length')
			# [batch_size, nSamples, maxSteps]
			valid_mask = tf.sequence_mask(lengths=length, maxlen=self.args.maxSteps, dtype=tf.float32, name='valid_mask')
			valid = tf.multiply(valid, valid_mask, name='valid')

			# [batch_size, nSamples]
			# number of valid decisions made in each sample
			n_valids = tf.reduce_sum(valid, axis=-1, name='n_valids')

		with tf.name_scope('pg_loss'):
			# [batch_size]
			rewards_mean, rewards_var = tf.nn.moments(rewards, axes=-1, name='rewards_moments')
			rewards_std = tf.sqrt(rewards_var, name='rewards_std')

			# [batch_size, 1]
			rewards_mean = tf.expand_dims(rewards_mean, axis=-1)
			# [batch_size, nSamples]
			rewards_mean = tf.tile(rewards_mean, multiples=[1, self.args.nSamples], name='rewards_mean')

			# [batch_size, 1]
			rewards_std = tf.expand_dims(rewards_std, axis=-1)
			# [batch_size, nSamples]
			rewards_std = tf.tile(rewards_std, multiples=[1, self.args.nSamples], name='rewards_std')

			# [batch_size, nSamples]
			#rewards_norm = tf.divide(tf.subtract(rewards, rewards_mean), rewards_std, name='rewards_norm')
			rewards_norm = tf.subtract(rewards, rewards_mean, name='rewards_norm')

			# [batch_size, nSamples, maxSteps]
			rewards_norm_tiled = tf.tile(tf.expand_dims(rewards_norm, axis=-1), multiples=[1, 1, self.args.maxSteps])
			# mask out steps that are not valid
			# [batch_size, nSamples, maxSteps]
			rewards_norm_tiled = tf.multiply(rewards_norm_tiled, valid, name='rewards_norm_tiled')

			# [batch_size, nSamples, maxSteps]
			pg_loss_ori = tf.multiply(rewards_norm_tiled, tf.log(probs), name='pg_loss_ori')

			## each sampled sample average over its valid steps

			# [batch_size, nSamples]
			pg_loss_sum = tf.reduce_sum(pg_loss_ori, axis=-1, name='pg_loss_sum')
			# [batch_size, nSamples]
			pg_loss_avg = tf.divide(pg_loss_sum, n_valids, name='pg_loss_avg')

			# average over samples
			# [batch_size]
			pg_loss = tf.reduce_mean(pg_loss_avg, axis=-1, name='pg_loss')

		with tf.name_scope('gradients'):
			ce_loss = tf.reduce_mean(ce_loss, axis=-1, name='ce_loss')

			trainable_params = tf.trainable_variables()

			# ce_params = []
			# pg_params = []
			#
			# for param in trainable_params:
			# 	if param.name == 'sampling/loop/skip_lstm_cell/skip_kernel:0' \
			# 			or param.name == 'sampling/loop/skip_lstm_cell/skip_bias:0':
			# 		pg_params.append(param)
			# 	else:
			# 		ce_params.append(param)

			# TODO: should we use gradients from pg_loss for params other than skip_kernel and skip_bias?
			# Yes，lower level nets should also be optimized for prediction of skips

			# sum over examples
			self.loss = tf.reduce_sum(ce_loss + pg_loss, name='loss')

			gradients_all = tf.gradients(self.loss, trainable_params)

			# gradients_ce = tf.gradients(ce_loss, ce_params)
			# gradients_pg = tf.gradients(pg_loss, pg_params)

			opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
			                             epsilon=1e-08)

			# all_params = ce_params + pg_params
			# all_gradients = gradients_ce + gradients_pg

			self.optOp = opt.apply_gradients(zip(gradients_all, trainable_params))

			print('RL model built!')


	def get_loss_and_rewards(self):
		"""
		run the rnn for one time
		:return: cross entropy loss and rewards
		"""
		with tf.name_scope('lstm'):
			with tf.variable_scope('cell', reuse=False):

				def get_cell(hiddenSize, dropOutRate):
					# for convenience, do not build custom cells
					cell = SkipLSTMCell(num_units=hiddenSize, state_is_tuple=True, min_read=self.args.minRead,
					                    max_skip=self.args.maxSkip, is_training=self.is_training, )
					cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropOutRate,
															 output_keep_prob=dropOutRate)
					return cell

				# https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn
				cell = get_cell(self.args.hiddenSize, self.dropOutRate)

			state = self.init_state()

			outputs = []
			skips_remain = []
			n_skips = []
			probs = []
			valid = []
			with tf.variable_scope("loop", reuse=tf.AUTO_REUSE):
				for time_step in range(self.args.maxSteps):
					# state:
					# "c", "h", "r", "s", "n", "probs", "valid"
					(cell_output, state) = cell(self.embedded[:, time_step, :], state)
					# n: number of steps skipped
					# p: corresponding probs of n
					# v: if is valid for computing reward
					# all of shape [batch_size]
					(_, _, _, s, n, p, v) = state
					skips_remain.append(s)
					n_skips.append(n)
					probs.append(p)
					valid.append(v)
					outputs.append(cell_output)

			# [maxSteps, batch_size]
			skips_remain = tf.stack(skips_remain)
			n_skips = tf.stack(n_skips)
			probs = tf.stack(probs)
			valid = tf.stack(valid)

			# [batch_size, maxSteps]
			skips_remain = tf.transpose(skips_remain, [1, 0], name='skips_remain')
			n_skips = tf.transpose(n_skips, [1, 0], name='n_skips')
			probs = tf.transpose(probs, [1, 0], name='probs')
			valid = tf.transpose(valid, [1, 0], name='valid')

			# [maxSteps, batchSize, hiddenSize]
			outputs = tf.stack(outputs)
			# [batchSize, maxSteps, hiddenSize]
			outputs = tf.transpose(outputs, [1, 0, 2], name='outputs')

			# [batchSize, maxSteps]
			last_relevant_mask = tf.one_hot(indices=self.length-1, depth=self.args.maxSteps, name='last_relevant',
											dtype=tf.int32)
			# [batchSize, hiddenSize]
			last_relevant_outputs = tf.boolean_mask(outputs, last_relevant_mask, name='last_relevant_outputs')

		with tf.name_scope('output'):
			weights = tf.get_variable(name='weights', shape=[self.args.hiddenSize, self.args.numClasses],
									  initializer=self.initializer)

			biases = tf.get_variable(name='biases', shape=[self.args.numClasses],
			                         initializer=self.initializer)
			# [batchSize, numClasses]
			logits = tf.nn.xw_plus_b(x=last_relevant_outputs, weights=weights, biases=biases)

		with tf.name_scope('rewards'):
			# [batch_size]
			self.predictions = tf.argmax(logits, axis=-1, name='predictions', output_type=tf.int32)
			# [batch_size]
			self.corrects = tf.equal(self.predictions, self.labels, name='corrects')
			self.wrongs = tf.logical_not(self.corrects, name='wrongs')

			# single number
			n_corrects = tf.reduce_sum(tf.cast(self.corrects, tf.int32), name='n_corrects')

			# [batch_size], with elements 1 or -1, 1 for corrects and -1 for wrongs
			rewards = tf.subtract(tf.cast(self.corrects, tf.float32), tf.cast(self.wrongs, tf.float32), name='rewards')

		with tf.name_scope('ce_loss'):
			# [batch_size]
			ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='loss')


		return ce_loss, rewards, n_skips, probs, valid, n_corrects

	def init_state(self):
		# [batchSize, maxSteps, hiddenSize]
		c = tf.zeros(shape=[self.batch_size, self.args.hiddenSize], dtype=tf.float32)
		h = tf.zeros(shape=[self.batch_size, self.args.hiddenSize], dtype=tf.float32)
		r = tf.zeros(shape=[self.batch_size], dtype=tf.int32)
		s = tf.zeros(shape=[self.batch_size], dtype=tf.int32)
		n = tf.zeros(shape=[self.batch_size], dtype=tf.int32)
		probs = tf.zeros(shape=[self.batch_size], dtype=tf.float32)
		valid = tf.zeros(shape=[self.batch_size], dtype=tf.bool)

		state = SkipLSTMStateTuple(c, h, r, s, n, probs, valid)

		return state

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
			feed_dict[self.is_training] = True
			ops = (self.optOp, self.loss, self.n_corrects)
		else:
			# during test, do not use drop out!!!!
			feed_dict[self.dropOutRate] = 1.0
			feed_dict[self.is_training] = False
			ops = (self.loss, self.n_corrects)

		return ops, feed_dict, length
