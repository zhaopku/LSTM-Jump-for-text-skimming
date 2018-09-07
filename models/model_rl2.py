import tensorflow as tf
from models.utils import SkipLSTMStateTuple
from tensorflow.contrib.rnn import BasicLSTMCell
from models.utils import SkipLSTMCell
from models.acl_cell import ACLLSTMStateTuple, ACLSkipLSTMCell
import numpy as np

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
			# [batchSize, max_steps] during test
			# [batch_size*n_samples, max_steps] during training
			input_shape = [None, self.args.maxSteps]
			self.input = tf.placeholder(tf.int32, shape=input_shape, name='input')
			self.labels = tf.placeholder(tf.int32, shape=[None,], name='labels')
			self.length = tf.placeholder(tf.int32, shape=[None,], name='length')
			self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')
			self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
			self.random = tf.placeholder(tf.bool, shape=(), name='random')
			# number of samples for each training example, set to 1 during test
			self.n_samples = tf.placeholder(tf.int32, shape=(), name='n_samples')
			self.dropOutRate = tf.placeholder(tf.float32, (), name='dropOut')

			# pretrained values for gates, only valid for training samples
			self.induced_skips = tf.placeholder(tf.int32, shape=input_shape, name='hard_gates')
			self.is_transfering = tf.placeholder(tf.bool, shape=(), name='is_transfering')

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

			# [batchSize, maxSteps, embeddingSize] during test
			# [batch_size*n_samples, maxSteps, embeddingSize] during training
			self.embedded = tf.nn.embedding_lookup(embeddings, self.input)
			self.embedded = tf.nn.dropout(self.embedded, self.dropOutRate, name='embedding_dropout')

	def buildNetwork(self):
		with tf.name_scope('inputs'):
			self.getInputs()

		with tf.variable_scope('sampling', reuse=tf.AUTO_REUSE):
			"""
			sample from the rnn for nSample times
			"""
			# ce_loss, rewards: [batch_size*n_samples]
			# probs, valid, skip_flag: [batch_size*n_samples, maxSteps]
			# n_corrects: single number, number of correct predictions in batch_size*n_samples


			ce_loss, rewards, predicted_skips, probs, valid, self.n_corrects, skip_flag, transfering_loss = self.get_loss_and_rewards()

			# [batch_size, nSamples]
			ce_loss = tf.reshape(ce_loss, shape=[self.batch_size, self.n_samples], name='ce_loss')
			rewards = tf.reshape(rewards, shape=[self.batch_size, self.n_samples], name='rewards')
			transfering_loss = tf.reshape(transfering_loss, shape=[self.batch_size, self.n_samples], name='transfering_loss')

			# [batch_size, n_samples, maxSteps]
			probs = tf.reshape(probs, shape=[self.batch_size, self.n_samples, self.args.maxSteps], name='probs_ori')
			valid = tf.reshape(valid, shape=[self.batch_size, self.n_samples, self.args.maxSteps], name='valid_ori')
			skip_flag = tf.reshape(skip_flag, shape=[self.batch_size, self.n_samples, self.args.maxSteps], name='skip_flag_ori')
			predicted_skips = tf.reshape(predicted_skips, shape=[self.batch_size, self.n_samples, self.args.maxSteps], name='skip_flag_ori')


			probs = tf.add(probs, 1e-5, name='probs')
			valid = tf.cast(valid, tf.float32, name='valid_ori')


			# mask out steps exceeding the length of each sample
			# [batch_size, n_samples]
			length = tf.reshape(self.length, shape=[self.batch_size, self.n_samples], name='length')

			skip_flag_mask = tf.sequence_mask(lengths=length, maxlen=self.args.maxSteps, dtype=tf.float32, name='skip_flag_mask')
			# [batch_size, n_samples, maxSteps]
			skip_flag = tf.multiply(skip_flag, skip_flag_mask, name='skip_flag')
			# [batch_size, n_samples]
			n_skips = tf.reduce_sum(skip_flag, axis=-1, name='n_skips')
			# [batch_size, n_samples]
			skip_rate = tf.divide(n_skips, tf.cast(length, tf.float32), name='skip_rate')
			self.skip_rate = tf.reshape(skip_rate, shape=[-1])

			# [batch_size, n_samples]
			# number of valid decisions made in each sample
			# for sentence whose length <= min_read, n_valids would be 0
			n_valids = tf.reduce_sum(valid, axis=-1, name='n_valids')
			self.n_valids_sum = tf.reduce_sum(n_valids, name='n_valids_sum')

		with tf.name_scope('rewards'):
			# [batch_size, n_samples]

			sparse_rewards = tf.reduce_sum(predicted_skips, axis=-1, name='sparse_rewards')
			sparse_rewards = tf.multiply(self.args.sparse, tf.cast(sparse_rewards, tf.float32))
			rewards = tf.add(tf.cast(sparse_rewards, tf.float32), tf.cast(rewards, tf.float32), name='rewards')

		with tf.name_scope('pg_loss'):
			# [batch_size, ]
			rewards_mean, rewards_var = tf.nn.moments(rewards, axes=-1, name='rewards_moments')

			# [batch_size, 1]
			rewards_mean = tf.expand_dims(rewards_mean, axis=-1)
			# [batch_size, n_samples]
			rewards_mean = tf.tile(rewards_mean, multiples=[1, self.n_samples], name='rewards_mean')

			# [batch_size, n_samples]
			rewards_norm = tf.subtract(rewards, rewards_mean, name='rewards_norm')

			# [batch_size, n_samples, maxSteps]
			rewards_norm_tiled = tf.tile(tf.expand_dims(rewards_norm, axis=-1), multiples=[1, 1, self.args.maxSteps])
			# mask out steps that are not valid
			# [batch_size, n_samples, maxSteps]
			rewards_norm_tiled = tf.multiply(rewards_norm_tiled, valid, name='rewards_norm_tiled')
			rewards_norm_tiled = tf.stop_gradient(rewards_norm_tiled)
			# [batch_size, n_samples, maxSteps]
			pg_loss_ori = tf.multiply(rewards_norm_tiled, tf.log(probs), name='pg_loss_ori')

			## each sampled sample average over its valid steps

			# [batch_size, n_samples]
			pg_loss_sum = tf.reduce_sum(pg_loss_ori, axis=-1, name='pg_loss_sum')
			# [batch_size, n_samples]
			# some n_valids is 0, resulting in nan in pg_loss_avg, replace nan with 0, the average over valid steps
			n_valids = tf.where(tf.equal(tf.cast(n_valids, tf.int32), 0), tf.ones_like(n_valids)*1e10, n_valids, name='n_valids_final')
			pg_loss_avg = tf.divide(pg_loss_sum, n_valids, name='pg_loss_avg')


			# average over samples
			# [batch_size]
			pg_loss = tf.reduce_mean(pg_loss_avg, axis=-1, name='pg')
			pg_loss = tf.subtract(0.0, pg_loss, name='pg_loss')
			#pg_loss = tf.Print(pg_loss, data=[tf.reduce_sum(pg_loss)])


		with tf.name_scope('gradients'):
			# average over samples
			# [batch_size]
			ce_loss = tf.reduce_mean(ce_loss, axis=-1, name='ce_loss')
			transfering_loss = tf.reduce_mean(transfering_loss, axis=-1, name='transfering_loss')

			# mask out transfering_loss and pg_loss
			is_transfering = tf.cast(self.is_transfering, tf.float32)
			# disable transfering_loss when RL begins
			transfering_loss = tf.multiply(is_transfering, transfering_loss)
			pg_loss = tf.multiply((1.0-is_transfering), pg_loss)

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

			# add sparse_loss
			# sparse_loss = tf.Print(sparse_loss, data=[tf.reduce_sum(sparse_loss)])

			# when testing upper bound, we only cares about ce_loss
			#self.loss = tf.reduce_sum(ce_loss + pg_loss + transfering_loss, name='loss')
			self.loss = tf.reduce_sum(ce_loss, name='loss')

			gradients_all = tf.gradients(self.loss, trainable_params)

			# gradients_ce = tf.gradients(ce_loss, ce_params)
			# gradients_pg = tf.gradients(pg_loss, pg_params)
			# gradients_sparse = tf.gradients(sparse_loss, trainable_params)

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
					print('Not using ACL style jumping!')
					cell = SkipLSTMCell(num_units=hiddenSize, state_is_tuple=True, min_read=self.args.minRead,
					                    max_skip=self.args.maxSkip, is_training=self.is_training, is_transfering=self.is_transfering)
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

			predicted_logits = []

			with tf.variable_scope("loop", reuse=tf.AUTO_REUSE):
				for time_step in range(self.args.maxSteps):
					# state:
					# "c", "h", "r", "s", "n", "probs", "valid"
					(cell_output, state) = cell(self.embedded[:, time_step, :], state)
					# n: number of steps skipped
					# p: corresponding probs of n
					# v: if is valid for computing reward
					# all of shape [batch_size]
					# predicted_logits_: [batch_size*n_samples*(max_skips+1)]
					(c, h, r, s, n, p, v, _, predicted_logits_) = state
					skips_remain.append(s)
					n_skips.append(n)
					probs.append(p)
					valid.append(v)
					outputs.append(cell_output)
					induced_n = tf.slice(self.induced_skips, begin=[0, time_step+1], size=[-1, 1],
					                     name='induced_n'+str(time_step+1))
					induced_n = tf.reshape(induced_n, shape=[-1])
					state = SkipLSTMStateTuple(c, h, r, s, n, p, v, induced_n, predicted_logits_)
					#predicted_logits_ = tf.reshape(predicted_logits_, shape=[self.batch_size*self.n_samples, self.args.maxSkip+1])
					predicted_logits.append(predicted_logits_)

			# [maxSteps, batch_size]
			skips_remain.insert(0, tf.zeros(shape=[self.batch_size*self.n_samples], dtype=tf.int32))
			skips_remain = skips_remain[0:-1]
			skips_remain = tf.stack(skips_remain)
			n_skips = tf.stack(n_skips)
			probs = tf.stack(probs)
			valid = tf.stack(valid)
			# [max_steps, batch_size*n_samples, max_skips+1]
			predicted_logits = tf.stack(predicted_logits)

			# [batch_size, maxSteps]
			skip_flag = tf.cast(tf.greater(tf.transpose(skips_remain, [1, 0]), 0), tf.float32, name='skip_flag')
			#skip_flag = tf.Print(skip_flag, data=[skip_flag], summarize=100, message='skp_flag')
			n_skips = tf.transpose(n_skips, [1, 0], name='n_skips')
			probs = tf.transpose(probs, [1, 0], name='probs')
			valid = tf.transpose(valid, [1, 0], name='valid')

			# [batch_size*n_samples, max_steps, max_skips+1]
			predicted_logits = tf.transpose(predicted_logits, [1, 0, 2], name='predicted_logits')

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
			# rewards = tf.Print(rewards, data=[rewards], message='rewards')
		with tf.name_scope('ce_loss'):
			# [batch_size*n_samples]
			ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='loss')

		with tf.name_scope('transfering_loss'):
			# mask out steps exceeding the length of each sample
			# [batch_size, n_samples]
			valid = tf.cast(valid, tf.float32)
			valid = tf.reshape(valid, [self.batch_size, self.n_samples, self.args.maxSteps])
			length = tf.reshape(self.length, shape=[self.batch_size, self.n_samples], name='length')
			# [batch_size, n_samples, maxSteps]
			# note that a jump decision made at the last word is not valid
			# and, in our current mechanism, a sentence with length <= min_read does not have valid predictions
			valid_mask = tf.sequence_mask(lengths=length-1, maxlen=self.args.maxSteps, dtype=tf.float32, name='valid_mask')
			valid = tf.multiply(valid, valid_mask, name='valid')
			valid = tf.reshape(valid, shape=[-1, self.args.maxSteps])
			# predicted_logits: [batch_size*n_samples, max_steps, max_skips+1]
			# induced_skips: [batch_size*n_samples, max_steps]
			# valid: [batch_size*n_samples, max_steps]

			# transfering_loss: [batch_size*n_samples]
			transfering_loss = tf.contrib.seq2seq.sequence_loss(logits=predicted_logits, targets=self.induced_skips,
			                                                    weights=valid, average_across_timesteps=True,
			                                                    average_across_batch=False)
			# [batch_size*n_samples, max_steps]
			self.predicted_inference_skips = tf.argmax(predicted_logits, axis=-1, name='predicted_inference_skips', output_type=tf.int32)
			self.correct_predicted_inference_skips = tf.cast(tf.equal(self.predicted_inference_skips, self.induced_skips), tf.float32)
			self.correct_predicted_inference_skips = tf.multiply(self.correct_predicted_inference_skips, valid, name='correct_predicted_inference_skips')

		self.v0 = skip_flag
		return ce_loss, rewards, n_skips, probs, valid, n_corrects, skip_flag, transfering_loss

	def init_state(self):
		# [batchSize, maxSteps, hiddenSize]
		c = tf.zeros(shape=[self.batch_size*self.n_samples, self.args.hiddenSize], dtype=tf.float32)
		h = tf.zeros(shape=[self.batch_size*self.n_samples, self.args.hiddenSize], dtype=tf.float32)
		r = tf.zeros(shape=[self.batch_size*self.n_samples], dtype=tf.int32)
		s = tf.zeros(shape=[self.batch_size*self.n_samples], dtype=tf.int32)
		n = tf.zeros(shape=[self.batch_size*self.n_samples], dtype=tf.int32)
		probs = tf.zeros(shape=[self.batch_size*self.n_samples], dtype=tf.float32)
		valid = tf.zeros(shape=[self.batch_size*self.n_samples], dtype=tf.bool)

		induced_n = tf.slice(input_=self.induced_skips, begin=[0, 0], size=[-1, 1])
		induced_n = tf.reshape(induced_n, shape=[-1])
		predicted_logits = tf.zeros(shape=[self.batch_size*self.n_samples, self.args.maxSkip+1],
		                            dtype=tf.float32)
		#predicted_logits = tf.reshape(predicted_logits, shape=[-1])
		state = SkipLSTMStateTuple(c, h, r, s, n, probs, valid, induced_n, predicted_logits)

		return state


	def step(self, batch, test=False, eager = False, is_transfering = False):
		if not eager:
			return self.step_graph(batch, test, is_transfering)
		else:
			return self.step_eager(batch, test, is_transfering)

	def step_eager(self, batch, test=False, is_transfering=False):
		"""
		TODO: eager support
		:return:
		"""
		feed_dict = {}

		# [batchSize, maxSteps]
		input_ = []
		# [batch_size]
		length = []
		labels = []

		for sample in batch.samples:
			input_.append(sample.input_)
			labels.append(sample.label)
			length.append(sample.length)


		self.batch_size = len(length)

		if not test:
			input_ = np.expand_dims(input_, axis=1)
			input_ = np.tile(input_, reps=[1, self.args.nSamples, 1])
			input_ = np.reshape(input_, newshape=(-1, self.args.maxSteps))

			length = np.expand_dims(length, axis=1)
			length = np.tile(length, reps=[1, self.args.nSamples])
			length = np.reshape(length, newshape=(-1))

			labels = np.expand_dims(labels, axis=1)
			labels = np.tile(labels, reps=[1, self.args.nSamples])
			labels = np.reshape(labels, newshape=(-1))


		self.labels = labels
		self.input = input_
		self.length = length

		# feed_dict[self.labels] = labels
		# feed_dict[self.input] = input_
		# feed_dict[self.length] = length

		if not test:
			self.random = self.args.random
			self.n_samples = self.args.nSamples
			self.dropOutRate = self.args.dropOut
			self.is_training = True

		else:
			# during test, do not use drop out!!!!
			self.random = False
			self.n_samples = 1
			self.dropOutRate = 1.0
			self.is_training = False

	def step_graph(self, batch, test=False, is_transfering=False):
		"""
		during training, feed batch_size*n_samples
		:param batch:
		:param test:
		:param is_transfering:
		:return:
		"""
		feed_dict = {}

		# [batchSize, maxSteps]
		input_ = []
		# [batch_size]
		length = []
		labels = []
		induced_skips = []

		for sample in batch.samples:
			input_.append(sample.input_)
			labels.append(sample.label)
			length.append(sample.length)
			induced_skips.append(sample.induced_skips)

		feed_dict[self.batch_size] = len(length)

		if not test:
			input_ = np.expand_dims(input_, axis=1)
			input_ = np.tile(input_, reps=[1, self.args.nSamples, 1])
			input_ = np.reshape(input_, newshape=(-1, self.args.maxSteps))

			length = np.expand_dims(length, axis=1)
			length = np.tile(length, reps=[1, self.args.nSamples])
			length = np.reshape(length, newshape=(-1))

			labels = np.expand_dims(labels, axis=1)
			labels = np.tile(labels, reps=[1, self.args.nSamples])
			labels = np.reshape(labels, newshape=(-1))

			induced_skips = np.expand_dims(induced_skips, axis=1)
			induced_skips = np.tile(induced_skips, reps=[1, self.args.nSamples, 1])
			induced_skips = np.reshape(induced_skips, newshape=(-1, self.args.maxSteps))

		feed_dict[self.labels] = labels
		feed_dict[self.input] = input_
		feed_dict[self.length] = length
		feed_dict[self.induced_skips] = induced_skips
		feed_dict[self.is_transfering] = is_transfering

		if not test:
			feed_dict[self.random] = self.args.random
			feed_dict[self.n_samples] = self.args.nSamples
			feed_dict[self.dropOutRate] = self.args.dropOut
			feed_dict[self.is_training] = True
			ops = (self.optOp, self.loss, self.predictions, self.n_corrects, self.skip_rate,
			       self.correct_predicted_inference_skips, self.n_valids_sum)
		else:
			# during test, do not use drop out!!!!
			feed_dict[self.random] = False
			feed_dict[self.n_samples] = 1
			feed_dict[self.dropOutRate] = 1.0
			feed_dict[self.is_training] = False
			ops = (self.loss, self.predictions, self.n_corrects, self.skip_rate,
			       self.correct_predicted_inference_skips, self.n_valids_sum)


		return ops, feed_dict, length
