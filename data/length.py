from matplotlib import pyplot as plt
import numpy as np
l = []
with open('./rotten/train.txt', 'r') as file:
	lines = file.readlines()
	for line in lines:
		l.append(len(line.split())-1)


print(np.average(l))

plt.hist(l)
plt.show()