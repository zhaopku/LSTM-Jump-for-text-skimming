import os
import tensorflow as tf
import collections

"""
c, h: [batch_size, num_units], cell and hidden state computed at last step
r: [batch_size], number of steps read *before* current step
s: [batch_size], number of skips remaining *before* current step, which means if s > 0, we skip current step
n: [batch_size], predicted number of skips for each sample. note this is the actual number of skips, i.e., for
				samples that are not eligible to skip, this number is 0, but even if for samples that are eligible
				to skip, this number could also be zero, as the model may choose volunteerly not to skip
probs: [batch_size], probability of the corresponding skip (corresponds to n)
valid: [batch_size], bool type, which of the n and probs are valid at this step, note that the only criteria
					for deciding valid is: not in a skip && new_r > min_read
jump: [batch_size], int, how many jumps have happened *before* current step?
e: [batch_size], end flag for this sentence, set to True if the processing is over
"""

_ACLLSTMStateTuple = collections.namedtuple("ACLLSTMStateTuple", ("c", "h", "r", "s", "n", "probs", "valid", "jump", 'e'))


class ACLLSTMStateTuple(_ACLLSTMStateTuple):
	__slots__ = ()

	@property
	def dtype(self):
		(c, h, r, s, n, probs, valid, jump, e) = self
		if c.dtype != h.dtype:
			raise TypeError("Inconsistent internal state: %s vs %s" %
			                (str(c.dtype), str(h.dtype)))
		if r.dtype != s.dtype:
			raise TypeError("Inconsistent internal state: %s vs %s" %
			                (str(r.dtype), str(s.dtype)))

		# float, float, int, int, int, float, bool, int, bool
		return c.dtype, h.dtype, r.dtype, s.dtype, n.dtype, probs.dtype, valid.dtype, jump.dtype, e.dtype


class ACLSkipLSTMCell(tf.contrib.rnn.LayerRNNCell):
	"""
	state has three members: cell_state, output_state, and skip
	if skip == True:
		skip current step
	else:
		update cell_state and output_state
	"""

	def __init__(self, num_units, forget_bias=1.0, min_read=0, max_skip=0, max_jump=0,
	             state_is_tuple=True, activation=None, reuse=None, name=None, initializer=None, is_training=True, random=False,
	             eps=0.1):
		super(ACLSkipLSTMCell, self).__init__(_reuse=reuse, name=name)
		self._num_units = num_units
		self._forget_bias = forget_bias
		self._state_is_tuple = state_is_tuple
		self._activation = activation or tf.tanh
		self.initializer = initializer
		# minimum steps read before a step
		self.min_read = min_read
		# maximum steps skipped at each time
		self.max_skip = max_skip
		self.max_jump = max_jump
		self.random = random

		# if in the training mode
		self.is_training = is_training
		self.eps = eps

	@property
	def state_size(self):
		return (tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)
		        if self._state_is_tuple else 2 * self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def build(self, inputs_shape):
		"""
		build variables for LSTM cell
		:param inputs_shape: [batch_size, input_depth]
		:return:
		"""
		if inputs_shape[1].value is None:
			raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
			                 % inputs_shape)

		input_depth = inputs_shape[1].value
		h_depth = self._num_units

		self._bias = tf.get_variable(name='bias', shape=[4 * self._num_units],
		                             initializer=tf.zeros_initializer(dtype=self.dtype))
		self._kernel = tf.get_variable(name='kernel', shape=[input_depth + h_depth, 4 * self._num_units],
		                               initializer=self.initializer)

		# the params below must be updated by RL
		# max_skip+2: 0, 1, 2, ..., max_skip, max_skip+1
		# use max_skip+1 to denote choosing to end
		self._skip_weights = tf.get_variable(name='skip_kernel', shape=[self._num_units, self.max_skip + 2],
		                                     initializer=self.initializer)
		self._skip_bias = tf.get_variable(name='skip_bias', shape=[self.max_skip + 2],
		                                  initializer=tf.zeros_initializer(dtype=self.dtype))

		self.built = True

	def predict_skip(self, h, r, skip_flag):
		"""
		at each step we predict how many steps to skip from the next step
		note: this function should only be used after we update r and s
		:param h: old_h
		:param r: new_r computed at current step
		:param skip_flag: 1 if we skip at current step
		:return: n_skip, number of skips for each sample in the batch, this number is exact
		"""
		# [batch_size, max_skip+2]
		logits = tf.nn.xw_plus_b(h, self._skip_weights, self._skip_bias, name='logits')

		# [batch_size]
		# indifferentiable happens here

		# just choose the prediction with maximum prob during inference
		predictions_inference = tf.argmax(logits, axis=-1, output_type=tf.int32)

		# during training, use epsilon greedy
		eps = tf.random_uniform(shape=[tf.shape(logits)[0], ], minval=0, maxval=1, dtype=tf.float32)

		# if eps > self.eps, we sample from the softmax distribution, else we sample one randomly
		mask = tf.cast(tf.greater(eps, self.eps), tf.int32)

		# [batch_size, 1]
		predictions_sampled = tf.multinomial(logits=logits, num_samples=1, output_dtype=tf.int32)
		# [batch_size]
		# sampled from the softmax distribution
		predictions_sampled = tf.squeeze(predictions_sampled, axis=-1)

		# [batch_size]
		# a random uniform variable between [0, max_skip+2]
		# 0: not skip (in the ACL context, "skip" 1 step)
		# max_skip+1: out
		predictions_random = tf.random_uniform(shape=[tf.shape(logits)[0], ], minval=0, maxval=self.max_skip+2,
		                                       dtype=tf.int32)

		# number of skips predicted in training
		predictions_training = tf.multiply(predictions_sampled, mask) + tf.multiply((1 - mask), predictions_random,
		                                                                            name='predictions_training')

		is_training = tf.cast(self.is_training, tf.int32, name='is_training')
		predictions = tf.add(is_training * predictions_training, (1 - is_training) * predictions_inference,
		                     name='predictions')
		is_random = tf.cast(self.random, tf.int32, name='random')

		# if is_random == True, we randomly sample a jumping step (could lead to end of processing)
		predictions = tf.add(is_random * predictions_random, (1 - is_random) * predictions,
		                     name='predictions_')

		# [batch_size, max_skip+2]
		probs = tf.nn.softmax(logits=logits, axis=-1)

		# use the index of predictions to get corresponding probs
		# [batch_size, max_skip+2]
		predictions_mask = tf.one_hot(indices=predictions, depth=self.max_skip+2, name='predictions_mask')

		# [batch_size]
		probs = tf.boolean_mask(probs, predictions_mask, name='probs')

		# only when r > min_read can we skip
		# [batch_size]
		mask_r = tf.cast(tf.greater(r, self.min_read - 1), tf.int32)

		# only those that are not in a skip can do a new skip
		# which means the remaining skips must equal 0
		mask_s = tf.cast(1 - skip_flag, tf.int32)

		n_skip = tf.multiply(predictions, mask_r)

		# [batch_size]
		n_skip = tf.multiply(n_skip, tf.cast(mask_s, tf.int32))

		# [batch_size]
		# below is the only criteria for deciding valid
		# valid = not in a skip && new_r >= min_read
		# valid = mask_s & mask_r
		valid = tf.greater(tf.multiply(mask_r, mask_s), 0)

		return n_skip, probs, valid

	@staticmethod
	def update_s(s, skip_flag):
		skip_flag = tf.cast(skip_flag, tf.int32)
		s_subtracted = tf.subtract(s, 1)
		new_s = tf.multiply(s_subtracted, skip_flag) + tf.multiply(s, (1 - skip_flag))

		return new_s

	@staticmethod
	def update_r(r, skip_flag):
		"""
		add 1 to r if s is 0
		:param r:
		:param skip_flag:
		:return:
		"""
		skip_flag = tf.cast(skip_flag, tf.int32)
		read_flag = 1 - skip_flag
		new_r = tf.add(read_flag, r)

		return new_r

	def call(self, inputs, state):
		"""
		:param inputs: `2-D` tensor with shape `[batch_size, input_size]`.
		:param state: c, h, r, s
		c, h: [batch_size, num_units], cell and hidden state computed at last step
		r: [batch_size], number of steps read *before* current step
		s: [batch_size], number of skips remaining *before* current step, which means if s > 0, we skip current step
		n: [batch_size], predicted number of skips for each sample. note this is the actual number of skips, i.e., for
						samples that are not eligible to skip, this number is 0, but even if for samples that are eligible
						to skip, this number could also be zero, as the model may choose not to skip
		probs: [batch_size], probability of the corresponding skip (corresponds to n)
		valid: [batch_size], bool type, which of the n and probs are valid at this step, note that the only criteria
							for deciding valid is: not in a skip && new_r > min_read
		jump: [batch_size], int, how many jumps have happened *before* current step?
		e: [batch_size], end flag for this sentence, set to True if the processing is over
		:return:
		"""
		if self._state_is_tuple:
			c, h, r, s, n_skip_old, probs_old, valid_old, jump, e = state
		else:
			c, h, r, s, n_skip_old, probs_old, valid_old, jump, e = tf.split(value=state, num_or_size_splits=6, axis=1)

		# pretend to read, compute c_updated and h_updated for later use

		gate_inputs = tf.matmul(tf.concat([inputs, h], 1), self._kernel)
		gate_inputs = tf.add(gate_inputs, self._bias)

		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		i, j, f, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)
		self._forget_bias = tf.cast(self._forget_bias, f.dtype)

		# [batch_size, hidden_size]
		c_updated = tf.add(tf.multiply(c, tf.nn.sigmoid(tf.add(f, self._forget_bias))),
		                   tf.multiply(tf.nn.sigmoid(i), self._activation(j)))

		h_updated = tf.multiply(self._activation(c_updated), tf.nn.sigmoid(o))

		# compute new_s and new_r

		# to compute new s:
		#  1. for elements > 0, minus 1
		#  2. for elements = 0, add n_skip (n_skip has already take new_r into consideration)

		# to compute new r:
		#  add 1 for those are not skipping at current step (this information is in skip_flag)

		# new_r and new_s computed from old r and s
		# whether or not we skip depend on the *old* s
		skip_flag = tf.cast(tf.greater(s, 0, name='skip_flag'), tf.float32)
		new_c = tf.add(tf.expand_dims((1.0 - skip_flag), -1) * c_updated, tf.expand_dims(skip_flag, -1) * c,
		               name='new_c')
		new_h = tf.add(tf.expand_dims((1.0 - skip_flag), -1) * h_updated, tf.expand_dims(skip_flag, -1) * h,
		               name='new_h')

		# to compute new_r:
		#  add 1 for those are not skipping at current step (this information is in skip_flag)
		# new_r: number of steps read after current step
		r_updated = self.update_r(r, skip_flag)

		# to compute new s:
		#  1. for elements > 0, minus 1
		#  2. for elements = 0, add n_skip (n_skip has already take new_r into consideration)
		s_updated = self.update_s(s, skip_flag)

		# when predicting n_skip, use r after added 1

		# which h do we use: only after read the current step, we get a chance to skip
		# and n_skip is only non-zero for samples which are eligible to skip
		# n_skip could be zero even if the sample is eligible to skip
		# note: we use the last element of n_skip to decide if we are quitting the process
		n_skip, probs, valid = self.predict_skip(h_updated, r_updated, skip_flag)
		# n_skip = tf.Print(n_skip, data=[n_skip], summarize=10, message='n')

		# should not use new_s to judge which are valid probs
		new_s = tf.add(n_skip, s_updated, name='new_s')

		# previously, we reset r to 0 for those with n_skip > 0
		# but in ACL style, we reset r as long as valid is set to true
		reset_flag = tf.cast(tf.greater(tf.cast(valid, tf.int32), 0), tf.int32, name='reset_flag')
		mask = 1 - reset_flag
		new_r = tf.multiply(r_updated, mask, name='new_r')

		# if the probs is valid, this means this step is a new jump, then we can add one to the jump counter
		new_jump = tf.add(jump, tf.cast(valid, tf.int32), name='new_j')

		# to comply with ACL style, we add two new rules for ending the process:
		# 1.  The predict_skip function choose to end
		# 2.  We hit a maximum number of jumps (even predict 0 count as a jump)
		choose_to_end = tf.equal(n_skip, self.max_skip + 1, name='choose_to_end')
		hit_max_jump = tf.greater_equal(new_jump, self.max_jump, name='hit_max_jump')

		new_e = tf.logical_or(choose_to_end, hit_max_jump, name='new_e')


		# filter by end flag
		# if condition is true, i.e., the process is over, we pass the old state, else we pass the new state
		new_c = tf.where(condition=e, y=new_c, x=c)
		new_h = tf.where(condition=e, y=new_h, x=h)
		new_r = tf.where(condition=e, y=new_r, x=r)
		new_s = tf.where(condition=e, y=new_s, x=s)

		n_skip = tf.where(condition=e, y=n_skip, x=n_skip_old)

		probs = tf.where(condition=e, y=probs, x=probs_old)
		valid = tf.where(condition=e, y=valid, x=valid_old)

		new_jump = tf.where(condition=e, y=new_jump, x=jump)

		new_e = tf.where(condition=e, y=new_e, x=e)


		if self._state_is_tuple:
			new_state = ACLLSTMStateTuple(new_c, new_h, new_r, new_s, n_skip, probs, valid, new_jump, new_e)
		else:
			new_state = tf.concat([new_c, new_h, new_r, new_s, n_skip, probs, valid], 1)

		return new_h, new_state

