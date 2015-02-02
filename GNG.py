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
    def __init__(self, inputvectors, max_node_count=2500, metric='sqeuclidean', learning_rate=[0.05,0.0006]):
        self.inputvectors = inputvectors
        self.n_input, self.cardinal  = inputvectors.shape
        self.max_node_count = max_node_count
        self.random_graph()
        self.errors = numpy.zeros(max_node_count)
        self.metric = metric
        self.learning_rate = learning_rate #learning rate for the winner (BMU) and the neighbors

    def random_graph(self):
        """
        create a graph with randomize weights. The initial graph contains two
        nodes connected by an edge
        """
        print "Graph initialization..."
        maxinpvalue = self.inputvectors.max(axis=0)
        mininpvalue = self.inputvectors.min(axis=0)
        weights = numpy.random.uniform(mininpvalue[0], maxinpvalue[0], (self.max_node_count,1))
        for e in zip(mininpvalue[1:],maxinpvalue[1:]):
            weights = numpy.concatenate( (weights, numpy.random.uniform(e[0],e[1], (self.max_node_count,1))), axis=1  )
        self.weights = weights
        self.graph = {}
        self.updategraph(0,1)

    def updategraph(self, n1, n2, graph=None):
        """
        update graph with node n1 and n2
        """
        if graph == None:
            graph = self.graph
        try:
            graph[n1].add(n2)
        except KeyError:
            graph[n1] = set([n2])
        try:
            graph[n2].add(n1)
        except KeyError:
            graph[n2] = set([n1])

    def get_vertices(self, graph=None):
        """
        return the list of vertices in a graph
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        vertlist = []
        for n1 in G.keys():
            if n1 not in vertlist:
                vertlist.append(n1)
            for n2 in G[n1]:
                if n2 not in vertlist:
                    vertlist.append(n2)
        return vertlist

    def findBMU(self, k, graph=None, return_distance=False):
        """
        Find the two Best Matching Unit for the input vector number k and add
        error for the BMU
        """
        if graph == None:
            graph = self.graph
        vertices = numpy.asarray(self.get_vertices())
        cdist = scipy.spatial.distance.cdist(self.inputvectors[None,k], self.weights[vertices], self.metric)[0]
        indices = cdist.argsort()[:2]
        indices = vertices[indices]
        self.errors[indices[0]] += cdist[0]
        if not return_distance:
            return indices
        else:
            return indices, cdist[indices]

    def adapt(self, bmu, k):
        """
        adapt the weights for bmu and input vector k
        """
        neighbors = list(self.graph[bmu])
        self.weights[bmu] += self.learning_rate[0] * (self.inputvectors[k] - self.weights[bmu])
        self.weights[neighbors] += self.learning_rate[1] * (self.inputvectors[k] - self.weights[neighbors])

#    def learn():

