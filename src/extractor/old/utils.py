import numpy as np

class Node:
    def __init__(self, id, all_pos_len):
        self.id = id
        self.all_pos_len = all_pos_len

        #list of bool that represent the connectivity
        #self.left = [False for i in range(all_pos_len)]
        self.right = [False for i in range(all_pos_len)]


class Graph:
    def __init__(self, inds, num_iter = 3):
        self.inds = inds
        self.ind_len = len(inds)

        self.num_iter = num_iter
        self.nodes = []
        for i in range(num_iter):
            nodes = []
            for j in range(self.ind_len):
                nodes.append(Node((i, j), self.ind_len))
            self.nodes.append(nodes)

        self.all_paths = []

    def calc_pair_connectivity(self, pair):
        '''
        For all the pair that satisfy the connection requirement. 
        Change the Node connectivity.
        '''
        x, y = pair
        for i in range(self.num_iter):
            self.nodes[i][x].right[y] = True
        return 

    def calc_pair_connectivity_all(self, all_pair):
        for pair in all_pair:
            self.calc_pair_connectivity(pair)
        return

    def get_paths(self):
        '''
        After calc connectivity, calc path with dynamic programming. 
        '''
        c = 0
        for r in range(self.ind_len):
            self.get_path_helper(c, r, [r])
        return

    def get_path_helper(self, c, r, temp):
        '''
        Dynamic programming. 
        '''
        #if c == self.num_iter -1:
        if len(temp) == self.num_iter:
            self.all_paths.append([t for t in temp])
            return
        #print('c ' + str(c) + ' r ' + str(r))
        rs = [i for i, x in enumerate(self.nodes[c][r].right) if x]
        if len(rs) == 0:
            return
        for _r in rs:
            #print('col ' + str(c) + ' temp ' + '_'.join([str(t) for t in temp]))
            _temp = temp.copy()
            _temp.append(_r)
            self.get_path_helper(c+1, _r, [t for t in _temp])

        return


#How to handle pair-wise combinations.
def combination_calc(inds, pair_set, num_iter = 3):
    '''
    inds: List of positions for combination calcluation.
    previously use of 'itertools.combinations(resinds, 3)', which is not efficient.
    Should use a graph based method with dynamic programming to remove useless combinations.
    '''
    graph = Graph(inds, num_iter)
    graph.calc_pair_connectivity_all(pair_set)
    graph.get_paths()

    return graph.all_paths

