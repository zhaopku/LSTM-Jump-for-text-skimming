import tensorflow as tf
from models.utils import SkipLSTMCell, SkipLSTMStateTuple

batch_size = 8
max_steps = 30
hidden_size = 10


tf.enable_eager_execution()


input_ = tf.random_normal(shape=[batch_size, max_steps, hidden_size])

cell = SkipLSTMCell(num_units=hidden_size, state_is_tuple=True, min_read=2,
                    max_skip=5, is_training=True)


def init_state():
	# [batchSize, maxSteps, hiddenSize]
	c = tf.zeros(shape=[batch_size, hidden_size], dtype=tf.float32)
	h = tf.zeros(shape=[batch_size, hidden_size], dtype=tf.float32)
	r = tf.zeros(shape=[batch_size], dtype=tf.int32)
	s = tf.zeros(shape=[batch_size], dtype=tf.int32)
	n = tf.zeros(shape=[batch_size], dtype=tf.int32)
	probs = tf.zeros(shape=[batch_size], dtype=tf.float32)
	valid = tf.zeros(shape=[batch_size], dtype=tf.bool)

	state = SkipLSTMStateTuple(c, h, r, s, n, probs, valid)

	return state


state = init_state()

for i in range(max_steps):
	(cell_output, state) = cell(input_[:, i, :], state)

	(c, h, r, s, n, p, v) = state
	print('c = {}\n'.format(c))
	print('n = {}\n'.format(n))
	print('s = {}\n'.format(s))
	print('r = {}\n'.format(r))
	print('v = {}\n'.format(v))

	print()
