#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 02 09
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import GNG
import matplotlib.pyplot as plt

class graph_plot:
    def __init__(self, gng):
        self.gng = gng

    def plot_isomap(self):
        try:
            embedding = self.gng.manifold.embedding_
        except AttributeError:
            self.gng.isomap()
            embedding = self.gng.manifold.embedding_
        try:
            communities = self.gng.communities
        except AttributeError:
            self.gng.best_partition()
            communities = self.gng.communities
        community_list = []
        for u in self.gng.graph:
            x = []
            y = []
            x.append(self.gng.manifold.embedding_[u,0])
            y.append(self.gng.manifold.embedding_[u,1])
            community_list.append(communities[u])
            for v in self.gng.graph[u]:
                x.append(self.gng.manifold.embedding_[v,0])
                y.append(self.gng.manifold.embedding_[v,1])
            plt.plot(x,y, '-', linewidth=0.25, c='green', alpha=0.25);
        plt.scatter(self.gng.manifold.embedding_[:,0], self.gng.manifold.embedding_[:,1], c=community_list, s=25)
