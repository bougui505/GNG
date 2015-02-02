#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 02 02
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import numpy
import scipy.spatial.distance

class GNG:
    def __init__(self, inputvectors, max_nodes = 100, metric = 'sqeuclidean', learning_rate = [0.05,0.0006], lambda_value = 300, a_max = 100):
        self.inputvectors = inputvectors
        self.n_input, self.cardinal  = inputvectors.shape
        self.max_nodes = max_nodes
        self.random_graph()
        self.errors = numpy.zeros(max_nodes) #the error between the BMU and the input vector
        self.metric = metric
        self.learning_rate = learning_rate #learning rate for the winner (BMU) and the neighbors
        self.a_max = a_max # maximal age

    def random_graph(self):
        """
        create a graph with randomize weights. The initial graph contains two
        nodes connected by an edge
        """
        print "Graph initialization..."
        maxinpvalue = self.inputvectors.max(axis=0)
        mininpvalue = self.inputvectors.min(axis=0)
        weights = numpy.random.uniform(mininpvalue[0], maxinpvalue[0], (self.max_nodes,1))
        for e in zip(mininpvalue[1:],maxinpvalue[1:]):
            weights = numpy.concatenate( (weights, numpy.random.uniform(e[0],e[1], (self.max_nodes,1))), axis=1  )
        self.weights = weights
        self.graph = {}
        self.updategraph(0,1)

    def updategraph(self, n1, n2, age=0):
        """
        update graph with node n1 and n2 and age
        """
        graph = self.graph
        try:
            graph[n1].update({(n2):age})
        except KeyError:
            graph[n1] = {(n2):age}
        try:
            graph[n2].update({(n1):age})
        except KeyError:
            graph[n2] = {(n1):age}

    def get_nodes(self):
        """
        return the list of nodes in a graph
        """
        G = self.graph
        vertlist = set([])
        for n1 in G.keys():
            vertlist.add(n1)
            for n2 in G[n1].keys():
                vertlist.add(n2)
        return list(vertlist)

    def findBMU(self, k, return_distance=False):
        """
        Find the two Best Matching Unit for the input vector number k and add
        error for the BMU
        """
        graph = self.graph
        nodes = numpy.asarray(self.get_nodes())
        cdist = scipy.spatial.distance.cdist(self.inputvectors[None,k], self.weights[nodes], self.metric)[0]
        indices = cdist.argsort()[:2]
        indices = nodes[indices]
        self.errors[indices[0]] += cdist[0]
        if not return_distance:
            return indices
        else:
            return indices, cdist[indices]

    def has_edge(self, n1, n2):
        """
        test the existence of a edge n1-n2 in a graph
        """
        G = self.graph
        if G.has_key(n1):
            return G[n1].has_key(n2)
        else:
            return False

    def adapt(self, bmus, k):
        """
        - adapts the weights for bmu and input vector k
        - modifies the age of the edges
        - creates edge if necessary
        """
        bmu = bmus[0]
        neighbors = self.graph[bmu].keys()
        self.weights[bmu] += self.learning_rate[0] * (self.inputvectors[k] - self.weights[bmu])
        self.weights[neighbors] += self.learning_rate[1] * (self.inputvectors[k] - self.weights[neighbors])
        for i in neighbors:
            self.graph[bmu][i] += 1 # increment age of the edge
        if self.has_edge(bmus[0], bmus[1]):
            self.graph[bmus[0]][bmus[1]] = 0 # if the edge already exist set the age to 0
        else:
            self.updategraph(bmus[0],bmus[1]) # else create the edge

    def delete_edge(self, n1, n2):
        """
        delete an edge n1 -> n2 from a graph
        """
        G = self.graph
        del G[n1][n2]
        del G[n2][n1]
        if G[n1] == {}:
            del G[n1]
        if G[n2] == {}:
            del G[n2]

    def insert_node(self):
        graph = self.graph
        u = self.errors.argmax()
        neighbors = graph[u].keys()
        v = neighbors[self.errors[neighbors].argmax()] # neighbor of i with the largest error
        print v


#    def learn():

