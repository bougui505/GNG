#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 02 18
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
import sklearn.manifold
import sklearn.cluster
from sklearn.neighbors import NearestNeighbors
import networkx
import community
import pickle

class GNG:
    def __init__(self, inputvectors=None, max_nodes = 100, metric = 'sqeuclidean', learning_rate = [0.2,0.006], lambda_value = 100, a_max = 50, alpha_value = 0.5, beta_value = 0.0005, max_iterations=None, data=None):
        if data == None:
            self.inputvectors = inputvectors
            self.n_input, self.cardinal  = inputvectors.shape
            self.max_nodes = max_nodes
            self.unvisited_nodes = set(range(max_nodes)) # set of unvisited nodes
            self.metric = metric
            self.learning_rate = learning_rate #learning rate for the winner (BMU) and the neighbors
            self.a_max = a_max # maximal age
            self.alpha_value = alpha_value # the coefficient of error decreasing in insertion place
            self.beta_value = beta_value # the global coefficient of error decreasing
            self.lambda_value = lambda_value # the frequency of growing steps
            if max_iterations == None:
                self.max_iterations = self.n_input
            else:
                self.max_iterations = max_iterations
            self.random_graph()
            self.errors = numpy.zeros(max_nodes) #the error between the BMU and the input vector
        else:
            self.load_data(infile=data)

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

    def findBMU(self, k, return_distance=False):
        """
        Find the two Best Matching Unit for the input vector number k and add
        error for the BMU
        """
        graph = self.graph
        nodes = numpy.asarray(self.get_nodes())
        cdist = scipy.spatial.distance.cdist(self.inputvectors[None,k], self.weights[nodes], self.metric)[0]
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

    def adapt(self, bmus, k):
        """
        - adapts the weights for bmu and input vector k
        - creates edge if necessary
        - modifies the age of the edges
        - delete edges if their ages are more than a_max
        """
        bmu = bmus[0]
        neighbors = self.graph[bmu].keys()
        self.weights[bmu] += self.learning_rate[0] * (self.inputvectors[k] - self.weights[bmu])
        self.weights[neighbors] += self.learning_rate[1] * (self.inputvectors[k] - self.weights[neighbors])
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
            bmus = self.findBMU(k)
            self.adapt(bmus,k)
            if step % self.lambda_value == 0:
                self.insert_node()
            self.errors = self.errors - self.beta_value * self.errors # decrease globally the error
            step += 1
            pbar.update(step)
        pbar.finish()
        self.weights = self.weights[self.graph.keys()] # remove unattributed weights
        self.errors = self.errors[self.graph.keys()]
        self.delete_age()
        self.save_data()

    def delete_age(self):
        print "remove age from graph"
        self.G = {}
        for n1 in self.graph.keys():
            self.G[n1] = self.graph[n1].keys()
        print "unweighted graph stored in self.G"

    def save_data(self, outfile='gng.dat', **kwargs):
        print 'saving data in %s'%outfile
        data = self.__dict__
        for key, value in kwargs.iteritems():
            data[key] = value
        f = open(outfile,'wb')
        pickle.dump(data, f, 2)
        f.close()
        print 'done'

    def load_data(self, infile='gng.dat'):
        print 'loading data from %s'%infile
        f = open(infile,'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        print 'done'

    def ugraph(self):
        graph = self.graph
        U = {}
        for n1 in graph:
            U[n1] = {}
            for n2 in graph[n1]:
                d = scipy.spatial.distance.euclidean(self.weights[n1], self.weights[n2])
                U[n1][n2] = d
        self.U = U

    def get_population(self):
        """
        return the index of input data populating each node
        """
        print 'Computing population per node...'
        population = {}
        bmus = {}
        nbrs = NearestNeighbors(n_neighbors=1).fit(self.weights)
        distances, indices = nbrs.kneighbors(self.inputvectors)
        bmu_list = indices[:,0]
        for k in range(self.n_input):
            bmu = bmu_list[k]
            bmus[k] = bmu
            try:
                population[bmu].append(k)
            except KeyError:
                population[bmu] = [k]
        self.population = population
        self.bmus = bmus
        print 'Population per node stored in self.population dictionnary'
        print 'BMUs stored in self.bmus dictionnary'

    def get_transition_network(self, lag=1):
        """
        return the transition network
        """
        try:
            bmus = self.bmus
        except AttributeError:
            self.get_population()
            bmus = self.bmus
        print "computing transition network"
        transition_network = {}
        n = max(bmus.values())
        transition_matrix = numpy.zeros((n+1,n+1))
        density = numpy.zeros(n+1)
        for k1, k2 in zip(range(self.n_input), range(lag, self.n_input)):
            bmu1, bmu2 = bmus[k1], bmus[k2]
            transition_matrix[bmu1,bmu2] += 1
            density[bmu1] += 1
            if transition_network.has_key(bmu1):
                if transition_network[bmu1].has_key(bmu2):
                    transition_network[bmu1][bmu2] += 1
                else:
                    transition_network[bmu1].update({bmu2:1})
            else:
                transition_network.update({bmu1:{bmu2:1}})
        w, v = numpy.linalg.eig(transition_matrix)
        w, v = numpy.real(w), numpy.real(v)
        v = v[:,w.argsort()[::-1]]
        w = w[w.argsort()[::-1]]
        self.transition_matrix = transition_matrix / density
        self.transition_network = transition_network
        self.w = w
        self.v = v
        print "transition network stored in self.transition_network dictionnary"
        print "transition matrix stored in self.transition_matrix array"
        print "eigenvalues and eigenvectors of the transition matrix stored in self.w and self.v respectively"

    def get_metastable_states(self, k):
        """
        define k metastable states from the transition matrix decomposition
        see: http://bloggb.fr/python/2014/10/21/transition-networks-with-python.html
        """
        try:
            T = self.transition_matrix
        except AttributeError:
            self.get_transition_network()
            T = self.transition_matrix
        print "computing metastable states from the transition matrix decomposition"
        proj = numpy.dot(numpy.transpose(T),self.v[:,1:k])
        kmeans = sklearn.cluster.KMeans(n_clusters=k)
        kmeans.fit(proj)
        n = max(self.bmus.values())
        metastable_states = {}
        for i in range(n+1):
            metastable_states[i] = kmeans.labels_[i]
        self.metastable_states = metastable_states
        print "metastable states ids stored in self.metastable_states dictionnary"

    def get_medoids(self):
        """
        return the index of the medoid for each node
        """
        try:
            population = self.population
        except AttributeError:
            self.get_population()
            population = self.population
        print 'Computing medoid per node...'
        medoids = {}
        for n in population:
            index = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.inputvectors[population[n]])).mean(axis=0).argmin()
            medoid = population[n][index]
            medoids[n] = medoid
        self.medoids = medoids
        print 'Medoids stored in self.medoids dictionnary'

    def get_metamedoid(self, kinetic = False):
        """
        return the medoid node for each community
        if kinetic is true return the metamedoids for kinetic communities
        """
        if not kinetic:
            try:
                communities = self.communities
            except AttributeError:
                self.best_partition()
                communities = self.communities
            print 'Computing metamedoid per community...'
        else:
            try:
                communities = self.kinetic_communities
            except AttributeError:
                self.kinetic_best_partition()
                communities = self.kinetic_communities
            print 'Computing metamedoid per kinetic community...'
        community_ids = list(set(communities.values()))
        metamedoids = {}
        metamedoid_distances = {}
        for i in community_ids:
            nodes = self.get_nodes_for_community(i, kinetic=kinetic)
            index = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.weights[nodes])).mean(axis=0).argmin()
            medoid = nodes[index]
            metamedoids[i] = medoid
            metamedoid_distances.update( self.dijkstra(medoid, nodes, kinetic=kinetic) )
        if not kinetic:
            self.metamedoids = metamedoids
            self.metamedoid_distances = metamedoid_distances
            print 'Metamedoids stored in self.metamedoids dictionnary'
            print 'Shortest path distance to the metamedoid stored in self.metamedoid_distances dictionnary'
        else:
            self.kinetic_metamedoids = metamedoids
            self.kinetic_metamedoid_distances = metamedoid_distances
            print 'Kinetic metamedoids stored in self.kinetic_metamedoids dictionnary'
            print 'Shortest path distance to the kinetic metamedoid stored in self.kinetic_metamedoid_distances dictionnary'

    def project(self, data):
        """
        project data onto the nodes. Return a dictionnary with the value of
        the projection (value) for each node (key)
        """
        try:
            population = self.population
        except AttributeError:
            self.get_population()
            population = self.population
        print "projecting data onto each node..."
        projection = {}
        for n in population.keys():
            projection[n] = numpy.mean(data[population[n]])
        print "projection stored in self.projection"
        self.projection = projection
        return projection

    def undirected_edges(self, graph=None):
        """
        If an edge n1->n2 exists and n2->n1 exists. The function delete n2->n1
        """
        if graph == None:
            G = self.graph.copy()
        else:
            G = graph.copy()
        for n1 in G.keys():
            for n2 in G[n1].keys():
                if self.has_edge(n2, n1, G):
                    del G[n1][n2]
        return G

    def write_GML(self, outfilename, graph = None, directed_graph = False, community_detection = True, write_density = True, write_age = True, write_medoids = True, write_metamedoid_distances = True, kinetic = False, write_metastable = False, **kwargs):
        """
        Write gml file for ugraph.
        
        **kwargs: data to write for each node.  Typically, these data are
        obtained from the self.project function. The keys of the kwargs are
        used as keys in the GML file
        """
        if graph == None:
            try:
                graph = self.U
            except AttributeError:
                self.ugraph()
                graph = self.U
        try:
            population = self.population
        except AttributeError:
            self.get_population()
            population = self.population
        if community_detection:
            try:
                communities = self.communities
            except AttributeError:
                self.best_partition()
                communities = self.communities
        if write_medoids:
            try:
                medoids = self.medoids
            except AttributeError:
                self.get_medoids()
                medoids = self.medoids
        if write_metamedoid_distances:
            try:
                metamedoid_distances = self.metamedoid_distances
            except AttributeError:
                self.get_metamedoid()
                metamedoid_distances = self.metamedoid_distances
        if kinetic:
            try:
                kinetic_communities = self.kinetic_communities
                kinetic_metamedoids_distances = self.kinetic_metamedoid_distances
            except AttributeError:
                self.get_metamedoid(kinetic=True)
                kinetic_communities = self.kinetic_communities
                kinetic_metamedoids_distances = self.kinetic_metamedoid_distances
        density = {}
        for n in population.keys():
            density[n] = len(population[n])
        outfile = open(outfilename, 'w')
        outfile.write('graph [\n')
        if directed_graph:
            outfile.write('directed 1\n')
        else:
            outfile.write('directed 0\n')
        nodes = self.get_nodes(graph)
        for n in nodes:
            outfile.write('node [ id %d\n'%n)
            if write_density:
                try:
                    outfile.write('density %d\n'%density[n])
                except KeyError:
                    outfile.write('density 0\n')
            if community_detection:
                outfile.write('community %d\n'%(communities[n]))
            if write_medoids:
                try:
                    outfile.write('medoid %d\n'%(medoids[n]))
                except KeyError:
                    print "no medoid for node %d"%n
                    pass
            if write_metamedoid_distances:
                outfile.write('metamedoid %.4f\n'%numpy.exp(-metamedoid_distances[n]))
            if kinetic:
                try:
                    outfile.write('kinetic_community %d\n'%(kinetic_communities[n]))
                    outfile.write('kinetic_metamedoid %.4f\n'%(numpy.exp(-kinetic_metamedoids_distances[n])))
                except KeyError:
                    print "no kinetic community for node %d"%n
                    pass
            if write_metastable:
                outfile.write('metastable_state %d\n'%self.metastable_states[n])
            for key in kwargs.keys():
                try:
                    outfile.write('%s %.4f\n'%(key, kwargs[key][n]))
                except KeyError:
                    outfile.write('%s 0.0000\n'%key)
            outfile.write(']\n')
        if not directed_graph:
            undirected_graph = self.undirected_edges(graph)
        else:
            undirected_graph = graph
        for n1 in undirected_graph.keys():
            for n2 in undirected_graph[n1].keys():
                d = undirected_graph[n1][n2]
#                outfile.write('edge [\nsource %d\ntarget %d\n]\n'%(n1, n2))
                outfile.write('edge [ source %d target %d weight %.4f\n'%(n1, n2, d))
                if write_age:
                    outfile.write('age %d\n'%self.graph[n1][n2])
                if community_detection:
                    if communities[n1] == communities[n2]:
                        outfile.write('community %d\n'%communities[n1])
                if kinetic:
                    try:
                        if kinetic_communities[n1] == kinetic_communities[n2]:
                            outfile.write('kinetic_community %d\n'%kinetic_communities[n1])
                    except KeyError:
                        print "no kinetic community for edge %d-%d"%(n1,n2)
                outfile.write(']\n')
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

    def isomap(self):
        """
        perform isomap manifold embedding
        """
        print 'isomap manifold embedding'
        X = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.weights))
        self.manifold = sklearn.manifold.Isomap()
        self.manifold.fit(X)
        print 'isomap manifold embedding stored in self.manifold.embedding_'

    def best_partition(self):
        print "computing communities maximizing modularity"
        try:
            G = self.G
        except AttributeError:
            self.delete_age()
            G = self.G
        gnx = networkx.Graph(G)
        self.communities = community.best_partition(gnx)
        print "communities stored in self.communities"

    def kinetic_best_partition(self):
        try:
            transition_network = self.transition_network
        except AttributeError:
            self.get_transition_network()
            transition_network = self.transition_network
        print 'computing communities maximizing modularity from transition network'
        gnx = networkx.Graph(transition_network)
        self.kinetic_communities = community.best_partition(gnx)
        print 'kinetic communities stored in self.kinetic_communities'

    def get_transition_rate(self, c1, c2):
        """
        compute the transition rate between two kinetic communities c1 and c2
        """
        try:
            kinetic_communities = self.kinetic_communities
        except AttributeError:
            self.kinetic_best_partition()
            kinetic_communities = self.kinetic_communities
        nodelist_c1 = [n for n in kinetic_communities if kinetic_communities[n] == c1]
        nodelist_c2 = [n for n in kinetic_communities if kinetic_communities[n] == c2]
        num = 0
        population_c1 = 0
        for n1 in nodelist_c1:
            population_n1 = len(self.population[n1])
            population_c1 += population_n1
            for n2 in nodelist_c2:
                num += population_n1 * self.transition_matrix[n2,n1]
        trans = num / population_c1
        return trans

    def get_markov_chain_model(self):
        """
        return the Markov chain model from kinetic communities
        """
        try:
            kinetic_communities = self.kinetic_communities
        except AttributeError:
            self.kinetic_best_partition()
            kinetic_communities = self.kinetic_communities
        print "Computing Markov chain model from kinetic communities..."
        c_list = list(set(kinetic_communities.values()))
        markov_chain = {}
        for c1 in c_list:
            for c2 in c_list:
                rate = self.get_transition_rate(c1,c2)
                if rate > 0:
                    if markov_chain.has_key(c1):
                        markov_chain[c1].update({c2:rate})
                    else:
                        markov_chain[c1] = {c2:rate}
        self.markov_chain = markov_chain
        print "Markov chain model stored in self.markov_chain"
        print "Writing Markov chain model in markov_chain.gml file"
        self.write_GML('markov_chain.gml', graph = markov_chain, directed_graph = True, community_detection = False, write_density=False, write_age = False, write_medoids = False, write_metamedoid_distances = False, kinetic = False, write_metastable = False)

    def get_nodes_for_community(self, community_id, kinetic = False):
        """
        return a list of nodes belonging to the community with id community_id
        """
        if not kinetic:
            try:
                communities = self.communities
            except AttributeError:
                self.best_partition()
                communities = self.communities
        else:
            try:
                communities = self.kinetic_communities
            except AttributeError:
                self.kinetic_best_partition()
                communities = self.kinetic_communities
        return [k for k,v in communities.iteritems() if v == community_id]

    def dijkstra(self, start, nodes = None, kinetic = False):
        """
        computing shortest path from start for each node
        """
        if not kinetic:
            graph = self.graph.copy()
        else:
            graph = self.transition_network.copy()
        if nodes == None: # if a subset of nodes is not given
            nodes = self.get_nodes() # nodes to visit
        cc = start # current cell
        distances = {}
        for n in nodes:
            distances[n] = numpy.inf
        distances[start] = 0
        unvisited_cells_distance = distances.copy()
        visited_cells = set([])
        while len(visited_cells) < len(nodes):
            neighbors = set(graph[cc].keys()) & set(nodes)
            for n in neighbors - visited_cells:
                d = distances[cc] + 1
                if d < distances[n]:
                    distances[n] = d
                    unvisited_cells_distance[n] = d
            unvisited_cells_distance[cc] = numpy.inf # set visited cell distance to inf
            visited_cells.add(cc)
            cc = min(unvisited_cells_distance, key=unvisited_cells_distance.get) # cell with the minimum distance
        return distances
