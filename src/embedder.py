import torch
from collections import defaultdict
import random
random.seed(0)
import os
from utils import *
import pickle as pkl
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
import fastrand
import pathlib
warnings.filterwarnings('ignore')
from os import path


class embedder:
    def __init__(self, args):
        self.embedder = args.embedder
        self.dataset = args.dataset
        self.iter_max = args.iter_max
        self.dim = args.dim
        self.window_size = args.window_size
        self.path_length = args.path_length
        self.num_neg = args.num_neg
        self.num_walks_per_node = args.num_walks_per_node
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gpu_num = args.gpu_num
        self.pooling = args.pooling
        self.isInit = args.isInit
        self.isReg = args.isReg
        self.reg_coef = args.reg_coef
        self.threshold = args.threshold
        self.remove_percent = args.remove_percent
        self.patience = args.patience
        self.num_aspects = args.num_aspects
        self.eval_freq = args.eval_freq
        self.result_dict = {}

        if self.gpu_num == -1:
            self.device = "cpu"
        else:
            self.device = torch.device("cuda:" + str(self.gpu_num) if torch.cuda.is_available() else "cpu")
        args.device = self.device

        self.isSoftmax = args.isSoftmax
        if self.isSoftmax:
            self.isGumbelSoftmax = args.isGumbelSoftmax
            self.isNormalSoftmax = args.isNormalSoftmax
            if self.isGumbelSoftmax:
                self.tau_gumbel = args.tau_gumbel
                self.isHard = args.isHard

        folder_dataset = os.path.join(os.path.dirname(os.getcwd()), 'data', self.dataset)
        fn_data = os.path.join(folder_dataset, 'data_remove_percent_{}.pkl'.format(self.remove_percent))
        self.folder_dataset = folder_dataset

        # build graph
        print("Reading {}".format(fn_data))
        data = pkl.load(open(fn_data, "rb"))
        self.train_edges = data['train_edges']
        self.num_nodes = data['num_nodes']

        # Read graph data
        self.G = load_edgelist(self.train_edges, data['isDirected'])
        self.G.num_nodes = data['num_nodes']
        self.isDirected = data['isDirected']
        print("[{}] Num Nodes: {}, Num Edges: {}".format(self.dataset, self.num_nodes, len(self.train_edges)))

        args.num_nodes = data['num_nodes']
        self.early_stop = 0
        self.data = data
        self.args = args
        self.batch_path = '{}/rm_{}_batch_bn_{}_nw_{}_pl_{}_ws_{}_neg_{}'\
            .format(self.folder_dataset, self.remove_percent, self.batch_size,
                    self.num_walks_per_node, self.path_length, self.window_size, self.num_neg)

        pathlib.Path(self.batch_path).mkdir(parents=True, exist_ok=True)

        walk_path = "{}/walks.pkl".format(self.batch_path)
        if path.exists(walk_path):
            print("[{}] Loading walks from {}...".format(currentTime(), walk_path))
            walks = pkl.load(open(walk_path, "rb"))
        else:
            print("[{}] Generating walks...".format(currentTime()))
            walks = self.generate_walks(self.G, num_walks=self.num_walks_per_node)
            print("[{}] Saving walks to {}...".format(currentTime(), walk_path))
            pkl.dump(walks, open(walk_path, "wb"))

        self.center_contexts_dict = self.generate_pairs(walks)
        self.saved_model_DW = []
        self.saved_model_asp2vec = []
        self.walks = walks

    def generate_walks(self, G, num_walks):
        walks = []
        nodes = sorted(list(G.nodes()))
        print("Total number of nodes: {}".format(G.num_nodes))

        for cnt in range(num_walks):
            for node in nodes:
                path = G.random_walk(self.path_length, start=node)
                walks.append(path)

        return walks

    def generate_pairs(self, walks):
        center_contexts_dict = dict()
        for walk in walks:
            for i in range(len(walk)):
                curr_node = walk[i]
                context_nodes = walk[max(0, i - self.window_size): i] + walk[i+1: min(i + self.window_size + 1, len(walk))]
                center_contexts_dict.setdefault(curr_node, []).append(context_nodes)

        return center_contexts_dict

    def generate_training_batch(self):
        print("[{}] Generating training batch".format(currentTime()))
        pairs = [] # (center, context)
        negs = []
        offsets = [0]
        lists = []
        for center_node in self.center_contexts_dict:
            context_lists = self.center_contexts_dict[center_node]
            for context_list in context_lists:
                for ctx in context_list:
                    pair = [center_node, ctx]
                    pairs.append(pair)
                    neg_tmp = []
                    for _ in range(self.num_neg):
                        neg = fastrand.pcg32bounded(self.num_nodes)
                        while neg in set(context_list):
                            neg = fastrand.pcg32bounded(self.num_nodes)
                        neg_tmp.append(neg)
                    negs.append(neg_tmp)

                    # contexts.append(context_list)
                    # contexts_lens.append(len(context_list))

                    context_list_removed_duplicate = list(set(context_list))
                    offsets.append(offsets[-1] + len(context_list_removed_duplicate))
                    lists.extend(context_list_removed_duplicate)

        offsets = offsets[:-1]

        print("[{}] Done generating training batch".format(currentTime()))

        # Divide batches
        pairs = np.array(pairs)
        negs = np.array(negs)
        offsets = np.array(offsets)
        mini_batch_n = int(pairs.shape[0] / self.batch_size)
        pairs_batch = []
        negs_batch = []
        offsets_batch = []
        lists_batch = []
        for i in range(mini_batch_n):
            pairs_batch.append(pairs[i * self.batch_size:(i + 1) * self.batch_size])
            negs_batch.append(negs[i * self.batch_size:(i + 1) * self.batch_size])
            off = offsets[i * self.batch_size:(i + 1) * self.batch_size]
            offsets_batch.append(off - off[0])
            lists_batch.append(lists[offsets[i * self.batch_size]: offsets[(i + 1) * self.batch_size]])

        pairs_batch.append(pairs[mini_batch_n * self.batch_size:])
        negs_batch.append(negs[mini_batch_n * self.batch_size:])
        off = offsets[mini_batch_n * self.batch_size:]
        offsets_batch.append(off - off[0])
        lists_batch.append(lists[offsets[mini_batch_n * self.batch_size]:])

        print("Num. batches: {}".format(len(pairs_batch)))
        for idx in range(len(pairs_batch)):
            f_name = "{}/batch_{}.pkl".format(self.batch_path, idx)
            batch = [pairs_batch[idx], negs_batch[idx], offsets_batch[idx], lists_batch[idx]]
            pkl.dump(batch, open(f_name, "wb"))
            if idx % 100 == 0:
                print("Saving {}".format(f_name))

        return len(pairs_batch) # return number of batches


    def print_result(self, epoch=0, warmup=False, isFinal='Final'):
        idx = int(epoch / self.eval_freq)
        result = ''
        for data_type in sorted(self.result_dict):
            for metric in self.result_dict[data_type]:
                if isFinal == 'Final':
                    # Extract the best result
                    val = max(self.result_dict[data_type][metric])
                else:
                    val = self.result_dict[data_type][metric][idx]
                result += '{}: {}, '.format(metric, val)


        if isFinal == 'Final':
            if not warmup:
                print("[{}][{}][{}][Final] {}".format(currentTime(), self.embedder, self.dataset, result))
            else:
                print("[{}][warm-up][{}][Final] {}".format(currentTime(), self.dataset, result))
        else:
            if not warmup:
                print("[{}][{}][{}][Iter {}] Loss: {} | {}".format(currentTime(), self.embedder, self.dataset, epoch, np.round(self.batch_loss, 2), result))
            else:
                print("[{}][warm-up][{}][Iter {}] Loss: {} | {}".format(currentTime(), self.dataset, epoch, np.round(self.batch_loss, 2), result))

        sys.stdout.flush()

        if isFinal == 'Final':
            best = max(self.result_dict['Test']['AUC'])
            idx = self.result_dict['Test']['AUC'].index(best)
            return idx


    def is_converged(self, epoch):
        if epoch == 'Final':
            return True
        curr_idx = int(epoch / self.eval_freq)

        criterion = self.result_dict['Test']['AUC']

        if (curr_idx > 0) and (criterion[curr_idx] < criterion[curr_idx-1]):
            self.early_stop += 1

        if self.early_stop == self.patience / self.eval_freq:
            return True
        else:
            return False

    def eval_link_prediction(self, emb):
        train_edges = self.data['train_edges']
        train_eval_negatives = self.data['train_edges_neg']
        test_edges = self.data['test_edges']
        if self.isDirected:
            test_negatives = self.data['test_edges_neg_directed']
        else:
            test_negatives = self.data['test_edges_neg']

        train_pos = emb[train_edges[:, 0]] * emb[train_edges[:, 1]]
        train_neg = emb[train_eval_negatives[:, 0]] * emb[train_eval_negatives[:, 1]]
        train_X = np.concatenate((train_pos, train_neg))
        train_y = [1] * len(train_pos) + [0] * len(train_pos)
        classifier = LogisticRegression(random_state=0)
        classifier.fit(train_X, train_y)

        test_pos = emb[test_edges[:, 0]] * emb[test_edges[:, 1]]
        test_neg = emb[test_negatives[:, 0]] * emb[test_negatives[:, 1]]
        test_X = np.concatenate((test_pos, test_neg))
        test_y = [1] * len(test_pos) + [0] * len(test_neg)

        preds = classifier.predict_proba(test_X)
        AUC_test = roc_auc_score(test_y, preds[:, 1])

        self.result_dict.setdefault('Test', {}).setdefault('AUC', []).append(np.round(AUC_test,4))


class Graph(defaultdict):
    def __init__(self):
        super().__init__(list)
        self.num_nodes = 0

    def onehot_encoder(self, nodes):
        row = np.zeros(self.num_nodes)
        for node in nodes:
            row[node] = 1

        return row    

    def nodes(self):
        return self.keys()


    def remove_self_loops(self):
        for x in self:
            if x in self[x]:
                self[x].remove(x)
        return self

    def remove_test_data(self, fn_testing):
        test_data = pkl.load(open(fn_testing, "rb"))
        for node in test_data:
            self[node].remove(test_data[node][0])

    def random_walk(self, path_length, rand = random.Random(), start = None):
        G = self
        path = [start]
        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                node_new = rand.choice(G[cur])
                path.append(node_new)
            else:
                break

        return path

def load_edgelist(edges, isDirected):
    G = Graph()
    for edge in edges:
        node1 = edge[0]
        node2 = edge[1]

        G[node1].append(node2)

        if not isDirected:
            if node1 not in G[node2]:
                G[node2].append(node1)

    return G

