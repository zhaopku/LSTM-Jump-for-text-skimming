import os
import copy
import matplotlib.pyplot as plt
import numpy as np

class Result:
	def __init__(self):
		# file name
		self.file = None

		# configuration
		self.embedding_size = None
		self.hidden_size = None
		self.max_steps = None
		self.drop_out = None
		self.lr = None
		self.max_skip = None
		self.max_jump = None
		self.min_read = None
		self.eps = None
		self.sparse = None
		self.n_samples = None

		# performance
		self.train_acc = None
		self.val_acc = None
		self.test_acc = None

		self.train_skip = None
		self.val_skip = None
		self.test_skip = None

		self.cur_epoch = None

def process(f):
	with open(f, 'r') as file:
		lines = file.readlines()
		result = Result()
		result_final = Result()
		result.file = f

		for idx, line in enumerate(lines):
			if idx == 0:
				continue

			if line.startswith('embed'):
				result.embedding_size = int(line.split()[-1])
				continue
			elif line.startswith('hidden'):
				result.hidden_size = int(line.split()[-1])
				continue
			elif line.startswith('maxSteps'):
				result.max_steps = int(line.split()[-1])
				continue
			elif line.startswith('maxSk'):
				result.max_skip = int(line.split()[-1])
				continue
			elif line.startswith('maxJ'):
				result.max_jump = int(line.split()[-1])
				continue
			elif line.startswith('minR'):
				result.min_read = int(line.split()[-1])
				continue
			elif line.startswith('eps'):
				result.eps = float(line.split()[-1])
				continue
			elif line.startswith('learn'):
				result.lr = float(line.split()[-1])
				continue
			elif line.startswith('drop'):
				result.drop_out = float(line.split()[-1])
				continue
			elif line.startswith('nSamples'):
				result.n_samples = int(line.split()[-1])
				continue
			elif line.startswith('sparse'):
				result.sparse = float(line.split()[-1])
				continue

			splits = line.split()
			if line.startswith('epoch ='):
				result.train_acc = float(splits[8][:-1])
				result.train_skip = float(splits[11])
				result.cur_epoch = int(splits[2][:-1])
				continue
			elif line.startswith('\tVal'):
				result.val_acc = float(splits[6][:-1])
				result.val_skip = float(splits[9])
				continue
			elif line.startswith('\tTest'):
				result.test_acc = float(splits[6][:-1])
				result.test_skip = float(splits[9])
				continue
			elif line.startswith('New valAcc'):
				result_final = copy.deepcopy(result)

		return result_final

def my_plot(data, name='train', marker='o', color='red'):
	acc = []
	skip = []

	for it in data:
		acc.append(it[0]*100)
		skip.append(it[1]*100)
	x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	x = np.asarray(x)*100
	y = [80.3, 79.3, 77.1, 75.4, 75.3, 73.7, 68.1, 64.5, 62.7]

	plt.plot(x, y, marker='>', color='black')
	plt.xlabel('Skip rate')
	plt.ylabel('Acc')
	plt.xlim((0.0, 80.0))
	plt.ylim((50.0, 100.0))
	plt.title(name)
	plt.plot(skip, acc, marker = marker, color = color)
	plt.show()

def to_pic(all_results):
	train = []
	val = []
	test = []

	for res in all_results:
		train.append((res.train_acc, res.train_skip))
		val.append((res.val_acc, res.val_skip))
		test.append((res.test_acc, res.test_skip))

	train_sorted = sorted(train, key=lambda x:x[1])
	val_sorted = sorted(val, key=lambda x:x[1])
	test_sorted = sorted(test, key=lambda x:x[1])


	my_plot(train_sorted, 'train', 'o', 'red')
	my_plot(val_sorted, 'val', 's', 'blue')
	my_plot(test_sorted, 'test', '^', 'green')

if __name__ == '__main__':
	path = './result/rotten'
	files = os.listdir(path)
	all_results = []
	for f in files:
		if not f.startswith('em'):
			continue
		res = process(os.path.join(path, f))
		all_results.append(res)

	to_pic(all_results)
