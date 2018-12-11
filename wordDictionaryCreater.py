import pandas as pd
import numpy as np


class WordDictionary:

    def __init__(self, file):
        self.word_dictionary = {}
        data = pd.read_csv(file)
        data = data.drop('sentiment', axis=1)
        self.generate_dictionary(data)

    def generate_dictionary(self, data):
        # Temp dictionary
        temp = {}
        # Get all the text in the csv
        for text in data.values:
            lines = text[0]
            words = lines.split()
            for word in words:
                try:
                    temp[word] = temp[word] + 1
                except KeyError:
                    temp[word] = 0
        self.word_dictionary["<Unk>"] = 0
        self.word_dictionary["<Pad>"] = 1
        index = 2
        for key in temp.keys():
            if temp[key] > 0:
                self.word_dictionary[key] = index
                index = index + 1
        return self.word_dictionary

    def convertData(self, data):
        new_list = []
        # Get all the text in the csv
        for text in data.values:
            lines = text[0]
            words = lines.split()
            word_list = []
            for word in words:
                try:
                    word_list.append(self.word_dictionary[word])
                except KeyError:
                    word_list.append(0)
            new_list.append(np.asarray(word_list))
        return new_list

    def getDictionary(self):
        return self.word_dictionary
