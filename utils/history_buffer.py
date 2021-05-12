from collections import deque
import random

class HistoryBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque()

    def push(self, item):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer.popleft()
            self.buffer.append(item)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def to_list(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    history = HistoryBuffer(7)
    history.push([1, 2, 3])
    history.push([4, 5, 6])
    history.push([7, 8, 9])
    history.push([10, 11, 12])
    history.push([13, 14, 15])
    print(history.to_list())
    history.push([16, 17, 18])
    print(history.to_list())
    history.push([19, 20, 21])
    print(history.to_list())
    history.push([22, 23, 24])
    print(history.to_list())
    history.clear()
    print(history.to_list())