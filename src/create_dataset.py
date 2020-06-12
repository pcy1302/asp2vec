# Modified from https://github.com/google/asymproj_edge_dnn/blob/master/create_dataset_arrays.py
import pickle as pkl
import copy
import random
import networkx as nx
import numpy
import os
import sys
import argparse
import pandas as pd
import pdb

random.seed(0)

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='preprocess')

    parser.add_argument('--input', nargs='?', default='filmtrust')
    parser.add_argument('--directed', action='store_true', default=False)
    parser.add_argument('--remove_percent', type=float, default=0.5)

    return parser.parse_known_args()

def LargestSubgraph(graph):
    """Returns the Largest connected-component of `graph`."""
    if graph.__class__ == nx.Graph:
        return LargestUndirectedSubgraph(graph)
    elif graph.__class__ == nx.DiGraph:
        largest_undirected_cc = LargestUndirectedSubgraph(nx.Graph(graph))
        directed_subgraph = nx.DiGraph()
        for (n1, n2) in graph.edges():
            if n2 in largest_undirected_cc and n1 in largest_undirected_cc[n2]:
                directed_subgraph.add_edge(n1, n2)

        return directed_subgraph

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def LargestUndirectedSubgraph(graph):
    """Returns the largest connected-component of undirected `graph`."""
    if nx.is_connected(graph):
        return graph

    # cc = list(nx.connected_component_subgraphs(graph))
    cc = list(connected_component_subgraphs(graph))
    sizes = [len(c) for c in cc]
    max_idx = sizes.index(max(sizes))
    return cc[max_idx]

    # return sizes_and_cc[-1][1]


def SampleTestEdgesAndPruneGraph(graph, remove_percent, check_every=5):
    """Removes and returns `remove_percent` of edges from graph.
    Removal is random but makes sure graph stays connected."""
    graph = copy.deepcopy(graph)
    undirected_graph = graph.to_undirected()

    edges = list(copy.deepcopy(graph.edges()))
    random.shuffle(edges)
    remove_edges = int(len(edges) * remove_percent)
    num_edges_removed = 0
    currently_removing_edges = []
    removed_edges = []
    last_printed_prune_percentage = -1
    for j in range(len(edges)):
        n1, n2 = edges[j]
        graph.remove_edge(n1, n2)
        if n1 not in graph[n2]:
            undirected_graph.remove_edge(*(edges[j]))
        currently_removing_edges.append(edges[j])
        if j % check_every == 0:
            if nx.is_connected(undirected_graph):
                num_edges_removed += check_every
                removed_edges += currently_removing_edges
                currently_removing_edges = []
            else:
                for i in range(check_every):
                    graph.add_edge(*(edges[j - i]))
                    undirected_graph.add_edge(*(edges[j - i]))
                currently_removing_edges = []
                if not nx.is_connected(undirected_graph):
                    print ('  DID NOT RECOVER :(')
                    return None
        prunned_percentage = int(100 * len(removed_edges) / remove_edges)
        rounded = (prunned_percentage / 10) * 10
        if rounded != last_printed_prune_percentage:
            last_printed_prune_percentage = rounded
            # print ('Partitioning into train/test. Progress=%i%%' % rounded)

        if len(removed_edges) >= remove_edges:
            break

    return graph, removed_edges

def SampleNegativeEdges(graph, num_edges):
    """Samples `num_edges` edges from compliment of `graph`."""
    random_negatives = set()
    nodes = list(graph.nodes())
    while len(random_negatives) < num_edges:
        i1 = random.randint(0, len(nodes) - 1)
        i2 = random.randint(0, len(nodes) - 1)
        if i1 == i2:
            continue
        if i1 > i2:
            i1, i2 = i2, i1
        n1 = nodes[i1]
        n2 = nodes[i2]
        if graph.has_edge(n1, n2):
            continue
        random_negatives.add((n1, n2))

    return random_negatives


def RandomNegativesPerNode(graph, test_nodes_PerNode, negatives_per_node=499):
    """For every node u in graph, samples 20 (u, v) where v is not in graph[u]."""
    node_list = list(graph.nodes())
    num_nodes = len(node_list)
    for n in test_nodes_PerNode:
        found_negatives = 0
        while found_negatives < negatives_per_node:
            n2 = node_list[random.randint(0, num_nodes - 1)]
            if n == n2 or n2 in graph[n]:
                continue
            test_nodes_PerNode[n].append(n2)
            found_negatives += 1

    return test_nodes_PerNode


def NumberNodes(graph):
    """Returns a copy of `graph` where nodes are replaced by incremental ints."""
    node_list = sorted(graph.nodes())
    index = {n: i for (i, n) in enumerate(node_list)}

    newgraph = graph.__class__()
    for (n1, n2) in graph.edges():
        newgraph.add_edge(index[n1], index[n2])

    return newgraph, index



def MakeDirectedNegatives(positive_edges):
    positive_set = set([(u, v) for (u, v) in list(positive_edges)])
    directed_negatives = []
    for (u, v) in positive_set:
        if (v, u) not in positive_set:
            directed_negatives.append((v, u))
    return numpy.array(directed_negatives, dtype='int32')

def CreateDatasetFiles(graph, output_dir, directed, remove_percent, partition=True):
    """Writes a number of dataset files to `output_dir`.
    Args:
      graph: nx.Graph or nx.DiGraph to simulate walks on and extract negatives.
      output_dir: files will be written in this directory, including:
        {train, train.neg, test, test.neg}.txt.npy, index.pkl, and
        if flag --directed is set, test.directed.neg.txt.npy.
        The files {train, train.neg}.txt.npy are used for model selection;
        {test, test.neg, test.directed.neg}.txt.npy will be used for calculating
        eval metrics; index.pkl contains information about the graph (# of nodes,
        mapping from original graph IDs to new assigned integer ones in
        [0, largest_cc_size-1].
      partition: If set largest connected component will be used and data will
        separated into train/test splits.
    Returns:
      The training graph, after node renumbering.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_size = len(graph)
    if partition:
        graph = LargestSubgraph(graph)
        size_largest_cc = len(graph)
    else:
        size_largest_cc = -1
    graph, index = NumberNodes(graph)

    if partition:
        print("Generate dataset for link prediction")
        # For link prediction (50%:50%)
        train_graph, test_edges = SampleTestEdgesAndPruneGraph(graph, remove_percent)

    else:
        train_graph, test_edges = graph, []

    assert len(graph) == len(train_graph)

    # Sample negatives, to be equal to number of `test_edges` * 2.
    random_negatives = list(SampleNegativeEdges(graph, len(test_edges) + len(train_graph.edges())))
    random.shuffle(random_negatives)
    test_negatives = random_negatives[:len(test_edges)]
    # These are only used for evaluation, never training.
    train_eval_negatives = random_negatives[len(test_edges):]

    test_negatives = numpy.array(test_negatives, dtype='int32')
    test_edges = numpy.array(test_edges, dtype='int32')
    train_edges = numpy.array(train_graph.edges(), dtype='int32')
    train_eval_negatives = numpy.array(train_eval_negatives, dtype='int32')


    if directed:
        directed_test_negatives = MakeDirectedNegatives(numpy.concatenate([train_edges, test_edges], axis=0))
        test_edges_neg_directed = numpy.concatenate([directed_test_negatives, test_negatives], axis=0)

    data = {'isDirected':directed, 'index': index, 'num_nodes': len(graph), 'remove_percent': remove_percent,
            'train_edges': train_edges, 'train_edges_neg': train_eval_negatives, 'test_edges': test_edges, 'test_edges_neg': test_negatives}

    print("Size of train_edges: {}".format(len(train_edges)))
    print("Size of train_eval_negatives: {}".format(len(train_eval_negatives)))
    print("Size of test_edges: {}".format(len(test_edges)))
    print("Size of test_edges_neg: {}".format(len(test_negatives)))


    if directed:
        data['test_edges_neg_directed'] = test_edges_neg_directed
        print("Size of test_edges_neg_directed: {}".format(len(test_edges_neg_directed)))



    pkl.dump(data, open(os.path.join(output_dir, 'data_remove_percent_{}.pkl'.format(remove_percent)), "wb"))
    print("Saved to {}".format(os.path.join(output_dir, 'data_remove_percent_{}.pkl'.format(remove_percent))))


def main():
    args, unknown = parse_args()
    print(args)
    if args.directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    folder_dataset = os.path.join(os.path.dirname(os.getcwd()), 'data', args.input)
    edge_path = '{}/edges.txt'.format(folder_dataset)


    # Read graph
    print("Reading graph from {}".format(edge_path))
    graph = nx.read_edgelist(edge_path, create_using=graph, nodetype=int)

    # Create dataset files.
    print("Create {} dataset (Remove percent: {})".format(args.input, args.remove_percent))
    CreateDatasetFiles(graph, folder_dataset, args.directed, args.remove_percent)


if __name__ == '__main__':
    main()
