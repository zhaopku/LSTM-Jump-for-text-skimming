"""
split original dataset to train, val, and test. Ratio: 8:1:1
"""
import random

pos_file = './rotten/rt-polarity.pos'
neg_file = './rotten/rt-polarity.neg'

train_file = './rotten/train.txt'
val_file = './rotten/val.txt'
test_file = './rotten/test.txt'

with open(pos_file, 'r', encoding='latin-1') as pos, open(neg_file, 'r', encoding='latin-1') as neg:
	pos_samples = pos.readlines()
	neg_samples = neg.readlines()

	all_samples = []

	for sample in pos_samples:
		all_samples.append(sample.strip() + '\t1')
	for sample in neg_samples:
		all_samples.append(sample.strip() + '\t0')

random.shuffle(all_samples)

train_num = 8530
val_num = 1066

with open(train_file, 'w') as train, open(val_file, 'w') as val, open(test_file, 'w') as test:
	train_samples = all_samples[:train_num]
	val_samples = all_samples[train_num:train_num+val_num]
	test_samples = all_samples[train_num+val_num:]

	for sample in train_samples:
		train.write(sample+'\n')

	for sample in val_samples:
		val.write(sample+'\n')

	for sample in test_samples:
		test.write(sample+'\n')
