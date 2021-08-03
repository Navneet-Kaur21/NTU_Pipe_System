import sys
import numpy as np
class Graph():
    def __init__(self, numvertex):
        self.graph = [[-1] * numvertex for x in range(numvertex)]
        self.mstMatrix = [[-1]*numvertex for x in range(numvertex)]
        self.numvertex = numvertex
        self.V = numvertex
        self.verticeslist =[0]*numvertex #list of obj
        self.edgelist= [[-1] * numvertex for x in range(numvertex)]
        self.edge_count = 0

    def set_vertex(self, id, obj):
        if 0<=id<=self.numvertex:
            self.verticeslist[id] = obj

    def set_edge(self,frm,to,cost=0,pos=-1):
        self.graph[frm][to] = cost
        self.graph[to][frm] = self.graph[frm][to]

        self.edgelist[frm][to] = pos
        self.edgelist[to][frm] = self.edgelist[frm][to]
        self.edge_count += 1

    def get_vertex(self):
        return self.verticeslist

    def get_edges(self):
        edges=[]
        for i in range (self.numvertex):
            for j in range (self.numvertex):
                if (self.graph[i][j]!=-1):
                    edges.append((i,j,self.edgelist[i][j]))
        return edges

    def get_edge_num(self):
        return self.edge_count


    def get_matrix(self):
        return self.graph

    # A utility function to print the constructed MST stored in parent[]
    def printMST(self, parent,mstSet):
        print("Edge \tWeight")
        # for i in range(1, self.V):
        for i in range(len(parent)):
            if parent[i] is not None and mstSet[i] and parent[i]!=i:
                print (parent[i], "-", i, "\t", self.graph[i][ parent[i] ])


    def minKey(self, key, mstSet,remaining_v):

        # Initilaize min value
        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False and remaining_v[v]:
                min = key[v]
                min_index = v
        # return min_index
        try:
            min_index
            return min_index
        except:return None

        # Function to construct and print MST for a graph
        # represented using adjacency matrix representation
    def primMST(self):

        # Key values used to pick minimum weight edge in cut
        key = [sys.maxsize] * self.V
        parent = [None] * self.V # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0

        remaining_v = [True] * self.V

        self.mstForest = []

        while (np.array(remaining_v).any()):
            cur_mstSet = [False] * self.V
            cur_root = remaining_v.index(True)
            parent[cur_root] = cur_root # First node is always the root
            key[cur_root] = 0

            for cout in range(self.V):

                # Pick the minimum distance vertex from
                # the set of vertices not yet processed.
                # u is always equal to src in first iteration
                u = self.minKey(key, cur_mstSet,remaining_v)
                if u is None:
                    break
                remaining_v[cur_root] = False
                # Put the minimum distance vertex in
                # the shortest path tree
                cur_mstSet[u] = True
                remaining_v[u] = False

                # Update dist value of the adjacent vertices
                # of the picked vertex only if the current
                # distance is greater than new distance and
                # the vertex in not in the shotest path tree
                for v in range(self.V):

                    # graph[u][v] is non zero only for adjacent vertices of m
                    # mstSet[v] is false for vertices not yet included in MST
                    # Update the key only if graph[u][v] is smaller than key[v]
                    if self.graph[u][v] > 0 and cur_mstSet[v] == False and key[v] > self.graph[u][v]:
                        key[v] = self.graph[u][v]
                        parent[v] = u

            self.printMST(parent,cur_mstSet)
            # self.printMST(parent[cur_mstSet])
            self.mstForest.append(cur_mstSet)
            print(f'Not included vertices: {[i for i in range(self.V) if remaining_v[i]==True]}')



# G =Graph(9)
# G.set_vertex(0,'a')
# G.set_vertex(1,'b')
# G.set_vertex(2,'c')
# G.set_vertex(3,'d')
# G.set_vertex(4,'e')
# G.set_vertex(5,'f')
# G.set_vertex(6,'z')
# G.set_vertex(7,'z1')
# G.set_vertex(8,'sp')
# G.set_edge(0,4,10)
# G.set_edge(0,2,20)
# G.set_edge(2,1,30)
# G.set_edge(1,4,40)
# G.set_edge(4,3,50)
# G.set_edge(5,4,60)
# G.set_edge(6,7,60)
# G.primMST()
# print("Vertices of Graph")
# print(G.get_vertex())
# print("Edges of Graph")
# print(G.get_edges())
# print("Adjacency Matrix of Graph")
# print(G.get_matrix())