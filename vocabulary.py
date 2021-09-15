from typing import List


class Vocab:
    def __init__(self, data: List[str]):
        self.word2index, self.index2word = self.get_vocab(data)
        print(f"Vocabulary: {self.word2index}")
        self.size = len(self.word2index) + 1  # one for pad token

    def get_vocab(self, data: List[str]):
        word2index = {}
        index2word = {}
        index = 1
        for seq in data:
            for char in seq.split()[0]:
                if char not in word2index:
                    word2index[char] = index
                    index2word[index] = char
                    index += 1
        return word2index, index2word
