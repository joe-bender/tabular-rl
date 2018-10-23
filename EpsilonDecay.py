import numpy as np

class EpsilonDecay:
    def __init__(self, start, end, decay_rate):
        assert(0 <= start <= 1)
        assert(0 <= end <= 1)
        assert(0 < decay_rate < 1)
        self.start = start
        self.value = start
        self.end = end
        self.decay_rate = decay_rate
    
    def decay(self):
        self.value *= self.decay_rate
        self.value = max(self.value, self.end)
    
    def greedy(self, row):
        if np.random.rand() > self.value:
            return np.argmax(row)
        else:
            return np.random.randint(row.size)
    
    def probs(self, row):
        nA = row.size
        eps = self.value
        i_greedy = np.argmax(row)
        probs = np.full_like(row, eps/nA, dtype=np.double)
        probs[i_greedy] += 1-eps
        return probs
    
    def reset(self):
        self.value = self.start