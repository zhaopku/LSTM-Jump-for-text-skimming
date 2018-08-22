import numpy as np

class Sample:
    def __init__(self, data, words, steps, label, length):
        self.input_ = data[0:steps]
        self.sentence = words[0:steps]
        self.length = length
        self.label = label
        self.gates = None
        self.hard_gates = None
        self.induced_skips = None

    def init_gates(self, gates):
        self.gates = gates
        assert len(self.gates) == self.length

    def init_hard_gates(self, threshold=0.5, percent=False):
        # for val and test batches, we use dummy hard gates
        if self.gates is None:
            self.induced_skips = np.zeros(shape=(len(self.input_))).tolist()
            return

        # smaller gates value indicates we want to read the word
        self.hard_gates = []
        if percent:
            sorted_gates = sorted(self.gates)
            l = int(self.length*threshold)
            l_value = sorted_gates[l]

            for g in self.gates:
                if g > l_value:
                    # do not read
                    self.hard_gates.append(1)
                else:
                    # read
                    self.hard_gates.append(0)
        else:
            for g in self.gates:
                if g >= threshold:
                    # do not read
                    self.hard_gates.append(1)
                else:
                    # read
                    self.hard_gates.append(0)

        # always read the first word in a sentence
        self.hard_gates[0] = 0
        assert len(self.hard_gates) == self.length

    def convert(self, max_skip):
        if self.gates is None:
            return
        self.induced_skips = []

        for i in range(self.length):
            cnt = 0
            for j in range(i+1, self.length):
                if self.hard_gates[j] == 0 :
                    break
                elif self.hard_gates[j] == 1:
                    cnt += 1
                else:
                    print('Invalid hard gates!')
                    exit(-1)
            if cnt > max_skip:
                cnt = max_skip
            self.induced_skips.append(cnt)

        # then pad
        pad_length = len(self.sentence)

        while len(self.induced_skips) < pad_length:
            self.induced_skips.append(1)

class Batch:
    def __init__(self, samples):
        self.samples = samples
        self.batch_size = len(samples)

    def init_hard_gates(self, threshold=0.5, percent=False, max_skip=5):
        # only for training samples
        for sample in self.samples:
            sample.init_hard_gates(threshold=threshold, percent=percent)
            sample.convert(max_skip)