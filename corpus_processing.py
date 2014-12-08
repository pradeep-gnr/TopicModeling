import os
import sys
import csv
import string
import nltk

stopwords_dict = {}
with open("stopwords.txt") as f:
    for l in f:
        stopwords_dict[l.strip()] = True


class CorpusProcessor(object):
    # Converts raw documents into the format requireed for LDA
    # Main LDA interface requires each document to be tokinized and contain a
    # list of tokens.

    def __init__(self, remove_stopwords=False, min_term_freq=1):
        self.remove_stopwords = remove_stopwords
        self.min_term_freq = min_term_freq
        self.tokenized_docs = []

    def process(self, csv_file_path):
        # Input is a csv file. CSV file must contain a text column.
        # If ID is not present - default IDS will be used.
        has_id = False
        ip_file = csv.DictReader(open(csv_file_path))
        if "id" in ip_file.fieldnames:
            has_id = True

        print has_id
        for i, l in enumerate(ip_file):
            if has_id:
                id = l["id"]
            else:
                id = i + 1

            text = l["text"]
            tokens = nltk.word_tokenize(text)
            # Remove StopWords.
            tokens = [tok for tok in tokens if not stopwords_dict.get(tok)]
            # Remove punctuations.

            self.tokenized_docs.append({"id": id, "tokens": tokens})

if __name__ == "__main__":
    print "Testing this"
    ip_file = sys.argv[1]
    processor = CorpusProcessor()
    processor.process(ip_file)
    print "Hello..."
