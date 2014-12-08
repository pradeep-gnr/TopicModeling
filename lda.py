"""
This code is a sample prototype for infering topics using 
LDA with collpsed Gibbs Sampling for inference.
"""
from __future__ import division
import sys
import random
import numpy as np
import operator
import ipdb
from collections import defaultdict, OrderedDict


class LDA(object):

    def __init__(self, docs, alpha, beta, K, num_iterations=10):
        # Assumes documents are tokenized, stopword filtered etc. Each document
        # is a list of tokens.
        self.docs = docs
        # Dirichlet hyperparameter prior for the document-topic multinomial
        self.alpha = alpha
        # Prior hyperparameter for the topic-word multinomial
        self.beta = beta
        # Number of topics
        self.K = K
        # Number of iterations
        self.n_iter = num_iterations
        # Number of words in the corpus
        self.W = None
        # Number of words assigned to topic k in a document d
        self.n_d_k = None
        # Number of times word w is assigned to topic k
        self.n_k_w = None
        # Number of times any word is assigned to topic k
        self.n_k = None
        # Word-topic association Array
        self.z = None
        # Word-index mapping
        self.w_mappings = {}
        # Inverse word-index mapping
        self.ind_w_mappings = {}
        # Vocabulary Length
        self.W = None
        # Corpus Length
        self.D = None
        # Document to word mapping
        self.d_w_mapping = defaultdict(dict)
        # Word to document mapping
        self.w_d_mapping = defaultdict(dict)
        # Document - vocab count. Keep track of how much number of words are
        # there in each document
        self.d_w_count = defaultdict(dict)
        # Utility dict to keep track of docs for words in z array.
        self.w_d_index = defaultdict(dict)
        # Utility dict to map token indices to its corresponding word.
        self.w_z_index = defaultdict(dict)
        # Output file -- Fix me
        self.op_file = open("lda_output.txt", "wb")
        # Set up corpus and initialize all arrays.
        self._setup()

    def _setup(self):
        # Set up word-document associations and document word associations
        # prior to running inference.
        words = OrderedDict()
        self.D = len(self.docs)
        tok_ind = 0
        for i, d in enumerate(self.docs):
            self.d_w_count[i] = len(d["tokens"])
            for tok in d["tokens"]:
                words[tok] = True
                self.w_d_mapping[tok][i] = True
                if self.d_w_mapping[i].get(tok):
                    self.d_w_mapping[i][tok] += 1
                else:
                    self.d_w_mapping[i][tok] = 1
                self.w_d_index[tok_ind] = i
                self.w_z_index[tok_ind] = tok
                tok_ind += 1

        i = 0
        for w in words:
            self.w_mappings[w] = i
            self.ind_w_mappings[i] = w
            i += 1

        self.W = len(words)
        self.z = np.zeros(sum(self.d_w_count.values()))
        # Assign random topics
        self.z = map(lambda x: random.randint(0, self.K - 1), self.z)
        self.n_d_k = np.zeros((self.D, self.K))
        self.n_k_w = np.zeros((self.W, self.K))
        self.n_k = np.zeros(self.K)

        for w_ind, k in enumerate(self.z):
            w = self.w_mappings[self.w_z_index[w_ind]]
            self.n_k_w[w, k] += 1
            self.n_k[k] += 1
            d = self.w_d_index[w_ind]
            self.n_d_k[d, k] += 1

    def _sample_discrete_multinomial(self, distribution):
        # Sample from a multinomial
        d_sum = sum(distribution.values())
        dist_list = []
        for k, val in distribution.iteritems():
            distribution[k] = distribution[k] / d_sum
            dist_list.append(distribution[k])
        return np.random.multinomial(1, dist_list).argmax()

    def _run_sampler(self):
        # Main gibbs sampling function
        for it in range(self.n_iter):
            print "Iteration {}".format(it)

            # for each word.
            for i in range(len(self.z)):
                w = self.w_mappings[self.w_z_index[i]]
                d = self.w_d_index[i]
                topic = self.z[i]
                # Update document-topic counts
                self.n_d_k[d, topic] -= 1
                # Reduce word-topic counts
                self.n_k_w[w, topic] -= 1
                # Reduce topic assignment counts
                self.n_k[topic] -= 1
                # Construct discrete topic distribution for current word.
                topic_dist = {}
                for k in range(self.K):
                    # Update equation.
                    term_1 = (
                        self.n_k_w[w, k] + self.beta) / (self.n_k[k] + self.W * self.beta)
                    word = self.w_z_index[i]
                    # Number of words in the document except this word
                    n_d_i = self.d_w_count[d] - \
                        self.d_w_mapping[d].get(word, 0)
                    term_2 = (
                        self.n_d_k[d, k] + self.alpha) / (n_d_i + self.alpha * self.K)
                    topic_dist[k] = term_1 * term_2

                # Sample from the multinomial
                topic = self._sample_discrete_multinomial(topic_dist)
                # Update topic
                self.z[i] = topic
                self.n_d_k[d, topic] += 1
                # Reduce word-topic counts
                self.n_k_w[w, topic] += 1
                # Reduce topic assignment counts
                self.n_k[topic] += 1

        print "Done sampling..."

    def _get_distributions(self):
        # Extract various topic distributions
        # topic word_distribution
        topic_word_dist = defaultdict(dict)
        self.op_file.write("----LDA Output-----\n")
        self.op_file.write("\nTopic word Distributions\n")

        for k in range(self.K):
            self.op_file.write("\n\nTopic {}\n\n".format(k))
            if sum(self.n_k_w[:, k]) != 0:
                word_dist = self.n_k_w[:, k] / sum(self.n_k_w[:, k])
            else:
                word_dist = self.n_k_w[:, k]
            sort_words = np.argsort(word_dist)[::-1]
            topic_word_dist[k] = []
            for w_ind in sort_words[0:20]:
                topic_word_dist[k].append((w_ind, word_dist[w_ind]))

            # Write to file.
            for w_ind, prob in topic_word_dist[k]:
                word = self.ind_w_mappings[w_ind]

                self.op_file.write("{} : {} ".format(word, prob))
            self.op_file.write("\n")

        self.op_file.write("\n\n")
        self.op_file.write("Document topic distribution \n\n")
        # Document topic distribution.
        doc_topic_dist = defaultdict(dict)
        for d in range(self.D):
            self.op_file.write("\n\nDocument {}\nless ".format(d))
            topic_dist = self.n_d_k[d, :] / sum(self.n_d_k[d,:])
            sort_topics = np.argsort(topic_dist)[::-1]

            doc_topic_dist[d] = []
            for t_ind in sort_topics:
                doc_topic_dist[d].append((t_ind, topic_dist[t_ind]))

            for t_ind, prob in doc_topic_dist[d]:
                self.op_file.write("{} : {} ".format(t_ind, prob))

            self.op_file.write("\n")

    def run(self):
        # Main function for starting LDA.
        self._run_sampler()
        # Generate distributions.
        self._get_distributions()

if __name__ == "__main__":
    # Simple test
    docs_1 = [{"id": "1", "tokens": ["dog", "cat", "pig", "pig"]},
              {"id": "2", "tokens": ["dog", "cat", "pig", "cow"]},
              {"id": "3", "tokens": ["pig", "monkey", "donkey", "horse"]}]

    from corpus_processing import CorpusProcessor
    processor = CorpusProcessor()
    docs = processor.process(sys.argv[1])
    docs = processor.tokenized_docs
    print len(docs)
    lda = LDA(docs, 5, 0.1, 6, 30)
    lda.run()
