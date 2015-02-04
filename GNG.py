#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 02 04
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if is_interactive():
    import progressbar_notebook as progressbar
else:
    import progressbar

import numpy
import scipy.spatial.distance
import random

class GNG:
    def __init__(self, inputvectors, max_nodes = 100, metric = 'sqeuclidean', learning_rate = [0.2,0.006], lambda_value = 100, a_max = 50, alpha_value = 0.5, beta_value = 0.0005, max_iterations=None, graph=None):
        if metric == 'RMSD':
            self.metric = lambda A,B: self.align(A,B)[1]
        else:
            self.metric = metric
        self.inputvectors = inputvectors
        self.n_input, self.cardinal  = inputvectors.shape
        self.max_nodes = max_nodes
        self.unvisited_nodes = set(range(max_nodes)) # set of unvisited nodes
        self.learning_rate = learning_rate #learning rate for the winner (BMU) and the neighbors
        self.a_max = a_max # maximal age
        self.alpha_value = alpha_value # the coefficient of error decreasing in insertion place
        self.beta_value = beta_value # the global coefficient of error decreasing
        self.lambda_value = lambda_value # the frequency of growing steps
        if max_iterations == None:
            self.max_iterations = 2*self.n_input
        else:
            self.max_iterations = max_iterations
        if graph == None:
            self.random_graph()
            self.errors = numpy.zeros(max_nodes) #the error between the BMU and the input vector
        else:
            npz = numpy.load(graph)
            self.graph = npz['graph'].item()
            self.weights = npz['weights']
            self.errors = npz['errors']
            self.unvisited_nodes = self.unvisited_nodes - set(self.graph.keys())

    def align(self,A,B):
        """
        align coordinates of A on B
        return: new coordinates of A
        """
        N = A.shape[0]
        A = A.reshape(N/3,3)
        B = B.reshape(N/3,3)
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = numpy.dot(AA.T, BB)
        U, S, Vt = numpy.linalg.svd(H)
        R = numpy.dot(U,Vt)
        #  change sign of the last vector if needed to assure similar orientation of bases
        if numpy.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = numpy.dot(U,Vt)
        trans = centroid_B - centroid_A
        A = numpy.dot(A,R) + trans
        RMSD = numpy.sqrt( ( (B - A)**2 ).sum(axis=1).mean() )
        return A.flatten(), RMSD

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
            graph[n1].update({n2:age})
        except KeyError:
            graph[n1] = {n2:age}
            self.unvisited_nodes.remove(n1)
        try:
            graph[n2].update({n1:age})
        except KeyError:
            graph[n2] = {n1:age}
            self.unvisited_nodes.remove(n2)

    def get_nodes(self, graph = None):
        """
        return the list of nodes in a graph
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        return G.keys()

    def findBMU(self, v, return_distance=False):
        """
        Find the two Best Matching Unit for the input vector number k and add
        error for the BMU
        """
        graph = self.graph
        nodes = numpy.asarray(self.get_nodes())
        cdist = scipy.spatial.distance.cdist(v[None,:], self.weights[nodes], self.metric)[0]
        indices = cdist.argsort()
        bmus = nodes[indices][:2]
        self.errors[bmus[0]] += cdist[indices[0]]
        if not return_distance:
            return bmus
        else:
            return bmus, cdist[indices][:2]

    def has_edge(self, n1, n2, G = None):
        """
        test the existence of a edge n1-n2 in a graph
        """
        if G == None:
            G = self.graph
        if G.has_key(n1):
            return G[n1].has_key(n2)
        else:
            return False

    def adapt(self, bmus, v):
        """
        - adapts the weights for bmu and input vector k
        - creates edge if necessary
        - modifies the age of the edges
        - delete edges if their ages are more than a_max
        """
        bmu = bmus[0]
        neighbors = self.graph[bmu].keys()
        self.weights[bmu] += self.learning_rate[0] * (v - self.weights[bmu])
        self.weights[neighbors] += self.learning_rate[1] * (v - self.weights[neighbors])
        if not self.has_edge(bmus[0], bmus[1]):
            self.updategraph(bmus[0],bmus[1]) # create the edge if not present
        self.graph[bmus[0]][bmus[1]] = 0 # the edge between the two nearest nodes is set to age 0
        for i in neighbors:
            self.updategraph(bmu,i, age=self.graph[bmu][i]+1) # increment age of the edge
            if self.graph[bmu][i] > self.a_max: # delete edge if age is more than a_max
                self.delete_edge(bmu,i)

    def delete_edge(self, n1, n2, G = None):
        """
        delete an edge n1 -> n2 from a graph
        """
        if G == None:
            G = self.graph
        del G[n1][n2]
        del G[n2][n1]
        if G[n1] == {}:
            del G[n1]
            self.unvisited_nodes.add(n1)
            self.errors[n1] = 0
        if G[n2] == {}:
            del G[n2]
            self.unvisited_nodes.add(n2)
            self.errors[n2] = 0

    def insert_node(self):
        graph = self.graph
        u = self.errors.argmax() # node with the largest error
        neighbors = graph[u].keys() # neighbors of u
        v = neighbors[self.errors[neighbors].argmax()] # neighbor of i with the largest error
        r = min(self.unvisited_nodes) # attribution of an unvisited index to the new node
        self.weights[r] = ( self.weights[u] + self.weights[v] ) / 2 # attribution of the weights
        self.updategraph(u,r) # create edge u-r
        self.updategraph(v,r) # create edge v-r
        self.delete_edge(u,v) # delete edge u-v
        self.errors[u] = self.alpha_value*self.errors[u] # decrease the error of u
        self.errors[v] = self.alpha_value*self.errors[v] # decrease the error of v
        self.errors[r] = self.errors[u] # compute the error of r

    def learn(self):
        kv = []
        step = 0
        widgets = ['Growing Neural Gas: ', progressbar.Percentage(), progressbar.Bar(marker='=',left='[',right=']'), progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=self.max_iterations)
        pbar.start()
        while len(self.unvisited_nodes) > 0 and step < self.max_iterations:
            if len(kv) > 0:
                k = kv.pop()
            else:
                kv = range(self.n_input)
                random.shuffle(kv)
                k = kv.pop()
            v = self.inputvectors[k]
            bmus = self.findBMU(v)
            if self.metric == 'RMSD':
                v = self.align(v, self.weight[bmus[0]])[0]
            self.adapt(bmus,v)
            if step % self.lambda_value == 0:
                self.insert_node()
            self.errors = self.errors - self.beta_value * self.errors # decrease globally the error
            step += 1
            pbar.update(step)
        pbar.finish()
        self.weights = self.weights[self.graph.keys()] # remove unattributed weights
        self.errors = self.errors[self.graph.keys()]
        print "remove age from graph"
        self.G = {}
        for n1 in self.graph.keys():
            self.G[n1] = self.graph[n1].keys()
        outfilename = 'gng.npz'
        print "saving data in %s"%outfilename
        self.save_data(outfile='%s'%outfilename)

    def save_data(self, outfile='gng.npz', **kwargs):
        data = {'graph': self.graph, 'weights':self.weights, 'errors':self.errors}
        for key, value in kwargs.iteritems():
            data[key] = value
        numpy.savez(outfile, **data)

    def ugraph(self):
        graph = self.graph
        U = {}
        for n1 in graph:
            U[n1] = {}
            for n2 in graph[n1]:
                d = scipy.spatial.distance.euclidean(self.weights[n1], self.weights[n2])
                U[n1][n2] = d
        self.U = U

    def undirected_edges(self, graph=None):
        """
        If an edge n1->n2 exists and n2->n1 exists. The function delete n2->n1
        """
        if graph == None:
            G = self.graph
        else:
            G = graph
        for n1 in G.keys():
            for n2 in G[n1].keys():
                if self.has_edge(n2, n1, G):
                    del G[n1][n2]
        return G

    def write_GML(self, graph, outfilename):
        """
        write gml file for ugraph
        """
        outfile = open(outfilename, 'w')
        outfile.write('graph [\n')
        outfile.write('directed 0\n')
        nodes = self.get_nodes()
        for n in nodes:
            mean_d = numpy.mean(graph[n].values())
            outfile.write('node [ id %d weight %.4f ]\n'%(n,mean_d))
        graph = self.undirected_edges(graph)
        for n1 in graph.keys():
            for n2 in graph[n1].keys():
                d = graph[n1][n2]
#                outfile.write('edge [\nsource %d\ntarget %d\n]\n'%(n1, n2))
                outfile.write('edge [ source %d target %d weight %.4f ]\n'%(n1, n2, d))
        outfile.write(']')
        outfile.close()


    def writeSIF(self, graph, outfilename):
        outfile = open(outfilename, 'w')
        for n1 in graph.keys():
            for n2 in graph[n1].keys():
                d = graph[n1][n2]
                outfile.write('%d %.4f %d\n'%(n1, d, n2))
        outfile.close()


    def adjacency_matrix(self):
        graph = self.graph
        verts = self.get_nodes()
        vertdict = {}
        for i, vert in enumerate(verts):
            vertdict[vert] = i
        A = numpy.zeros((len(verts), len(verts)))
        for n1 in graph.keys():
            for n2 in graph[n1].keys():
                i,j = vertdict[n1], vertdict[n2]
                A[i,j] = 1
        return A

    def degree_matrix(self):
        graph = self.graph
        verts = self.get_nodes()
        D = numpy.zeros((len(verts), len(verts)))
        for i, n1 in enumerate(graph.keys()):
            D[i,i] = len(graph[n1])
        return D

    def laplacian_matrix(self):
        L = self.degree_matrix() - self.adjacency_matrix()
        return L
