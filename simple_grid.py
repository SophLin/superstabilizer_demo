#import qiskit

import numpy as np
import networkx as nx


def convert_syn_coords(i,j):
    # from coords in a syn_lattice to coords in memory experiments
    return (2*i-1,2*j-1)

def syn_coords_memory_to_lattice(i,j):
    return ((i+1)/2, (j+1)/2)

def convert_data_coords(i,j):
    # from coords in a dataq_lattice to coords in memory experiments
    return (2*i,2*j)

def data_coords_memory_to_lattice(i,j):
    return (i/2, j/2)

class chiplet:
    # a chiplet holding 1 patch of rotated surface code
    def __init__(self,code_dist,qubit_defects = None):
        self.d = code_dist
        self.link_defects = []
        #broken qubits are denoted by 1s and other qubits are denoted by 0s
        print("intializing lattices to 0")
        self.dataq_lattice = np.zeros((code_dist + 1, code_dist + 1))
        self.syn_lattice = np.zeros((code_dist + 1, code_dist + 1))
        if qubit_defects is None:
            self.qubit_defects = []
        else:
            self.qubit_defects = qubit_defects
            print("setting qubit defects according to qubit_defects")
            self.set_defects(qubit_defects)
        
    def clear_defects(self):
        self.link_defects = []
        self.qubit_defects = []
        self.dataq_lattice = np.zeros((self.d + 1, self.d + 1))
        self.syn_lattice = np.zeros((self.d + 1, self.d + 1))
    
    def set_defects(self, qubit_defects):
        for x,y in qubit_defects:
            if x%2 == 0:#data qubit
                self.dataq_lattice[int(x/2)][int(y/2)] = 1
            else: #syndrome qubit
                self.syn_lattice[int((x+1)/2)][int((y+1)/2)] = 1

    def set_link_defects(self, link_defects, disable_data = True):
        for data_coords, syn_coords in link_defects:
            #if the syn is not already broken, disable the data qubit included in each broken link
            if disable_data and self.syn_lattice[syn_coords[0]][syn_coords[1]] == 0:
                self.dataq_lattice[data_coords[0], data_coords[1]] = 1
            if not disable_data and self.dataq_lattice[data_coords[0]][data_coords[1]] == 0:
                self.syn_lattice[syn_coords[0], syn_coords[1]] = 1

    def broken_links(self,link_err):
        #create coupling map and set some of the links to be broken
        new_coupling_map = [] #(dataq coords, synq coords)

        for i in range(self.dataq_lattice.shape[0]):
            for j in range(self.dataq_lattice.shape[1]):
                #connect to upper left syn
                new_coupling_map.append(((i,j),(i,j)))
                if j != self.dataq_lattice.shape[1] - 1:
                    #connect to upper right syn
                    new_coupling_map.append(((i,j),(i,j+1)))
                if i != self.dataq_lattice.shape[0] - 1:
                    #connect to lower left syn
                    new_coupling_map.append(((i,j),(i+1,j)))
                if j != self.dataq_lattice.shape[1] - 1 and i != self.dataq_lattice.shape[0] - 1:
                    #connect to lower right syn
                    new_coupling_map.append(((i,j),(i+1,j+1)))

        coins = np.random.rand(len(new_coupling_map))
        faulty_links = []
        for i in range(len(coins)):
            if coins[i] < link_err:
                faulty_links.append(new_coupling_map[i])
        return faulty_links

    def set_err_fixed_freq(self,link_err):
        self.link_defects = self.broken_links(link_err)
        #disable the data qubit included in each broken link
        for data_coords, _ in self.link_defects:
            self.dataq_lattice[data_coords[0],data_coords[1]] = 1

    def set_err_tunable(self,link_err, qub_err):
        # broken data qubits
        for i in range(self.dataq_lattice.shape[0]):
            for j in range(self.dataq_lattice.shape[1]):
                if np.random.rand() < qub_err:
                    print("data err", i, j)
                    self.dataq_lattice[i][j] = 1
                    self.qubit_defects.append(convert_data_coords(i,j))
        # broken syndrome qubits
        for i in range(self.syn_lattice.shape[0]):
            for j in range(self.syn_lattice.shape[1]):
                if np.random.rand() < qub_err:
                    print("syndrome error", i, j)
                    self.syn_lattice[i][j] = 1
                    self.qubit_defects.append(convert_syn_coords(i,j))
        self.link_defects = self.broken_links(link_err)
        #if the syn is not already broken, disable the data qubit included in each broken link
        for data_coords, syn_coords in self.link_defects:
            if self.syn_lattice[syn_coords[0]][syn_coords[1]] == 0:
                self.dataq_lattice[data_coords[0],data_coords[1]] = 1

    def rotate_chiplet(self): # do not rotate back!
        #rotate the chiplet by 180 degrees
        self.dataq_lattice = np.zeros((self.d + 1, self.d + 1))
        self.syn_lattice = np.zeros((self.d + 1, self.d + 1))
        if self.qubit_defects:
            self.set_defects(self.qubit_defects)
        if self.link_defects:
            self.set_link_defects(self.link_defects, disable_data = False)
        new_synq_lattice = self.dataq_lattice[::-1,::-1]
        new_dataq_lattice = self.syn_lattice[::-1,::-1]
        self.dataq_lattice = new_dataq_lattice
        self.syn_lattice = new_synq_lattice

    def list_defects(self):
        #return a list of defects, in the coords system that we use for memory experiments
        #includes the qubits that are not used in the surface code patch,
        #but the memory experiment will ignore those
        defect_coords = []
        for i in range(self.dataq_lattice.shape[0]):
            for j in range(self.dataq_lattice.shape[1]):
                if self.dataq_lattice[i][j]:
                    defect_coords.append(convert_data_coords(i,j))
                if self.syn_lattice[i][j]:
                    defect_coords.append(convert_syn_coords(i,j))
        return defect_coords
    
    def syndrome_in_patch(self,i,j):
        #return True if (i,j) in the syndrome lattice is in the patch used for memory experiment
        if i == 0 and j%2 == 1:
            return False
        if j == 0 and i%2 == 0:
            return False
        if i == self.d and j%2 == 0:
            return False
        if j == self.d and i%2 == 1:
            return False
        return True

    def data_in_patch(self,i,j):
        #return True if (i,j) in the data lattice is in the patch used for memory experiment
        if i == self.d or j == self.d:
            return False
        return True

    def list_defects_in_patch(self):
        #return a list of defects, in the coords system that we use for memory experiments
        #only include qubits that are in the surface code patch
        defect_coords = []
        for i in range(self.dataq_lattice.shape[0]):
            for j in range(self.dataq_lattice.shape[1]):
                if self.dataq_lattice[i][j] and self.data_in_patch(i,j):
                    defect_coords.append(convert_data_coords(i,j))
                if self.syn_lattice[i][j] and self.syndrome_in_patch(i,j):
                    defect_coords.append(convert_syn_coords(i,j))
        return defect_coords

    def list_defects_rotated(self):
        self.rotate_chiplet()
        rotated_defects = self.list_defects()
        self.rotate_chiplet()
        return rotated_defects

    def count_defect_data(self):
        #number of faulty data qubits
        return int(np.sum(self.dataq_lattice))

    def count_defect_syn(self):
        #number of faulty syndrome qubits
        return int(np.sum(self.syn_lattice))

    def lower_edge_defect(self):
        #if one of the qubits on the lower edge of the chiplet is broken but
        #not used in the surface code
        if np.any(self.dataq_lattice[-1][:-1]) or np.any(self.syn_lattice[-1]):
            return True
        return False

    def right_edge_defect(self):
        if np.any(self.dataq_lattice[:, -1][:-1]) or np.any(self.syn_lattice[:, -1]):
            return True
        return False

    def upper_edge_defect(self):
        if np.any(self.syn_lattice[0]):
            return True
        return False

    def left_edge_defect(self):
        if np.any(self.syn_lattice[:, 0]):
            return True
        return False

class combined_chip:#a combined chip consisting of an array of chiplets
    def __init__(self,code_dist,numq_row,numq_col):
        self.d = code_dist
        self.numq_row = numq_row
        self.numq_col = numq_col
        #broken qubits are denoted by 1s and other qubits are denoted by 0s
        #self.dataq_lattice = np.zeros((code_dist * (numq_row + 1), code_dist * (numq_col + 1)))
        #self.syn_lattice = np.zeros((code_dist * (numq_row + 1), code_dist * (numq_col + 1)))
        self.chiplet_array = [[chiplet(code_dist) for i in range(numq_col)] for j in range (numq_row)]

    def set_err_fixed_freq(self,link_err):
        for i in range(self.numq_row):
            for j in range(self.numq_col):
                self.chiplet_array[i][j].set_err_fixed_freq(link_err)

    def set_err_tunable(self,link_err, qub_err):
        for i in range(self.numq_row):
            for j in range(self.numq_col):
                self.chiplet_array[i][j].set_err_tunable(link_err, qub_err)

    def check_tile_defective(self, fixed_freq=False):
        #return a boolean matrix denoting whether each tile is defective
        is_defective = np.zeros((self.numq_row,self.numq_col))
        for i in range(self.numq_row):
            for j in range(self.numq_col):
                if self.chiplet_array[i][j].count_defect_data() > 0:
                    is_defective[i,j] = 1
                if not fixed_freq and self.chiplet_array[i][j].count_defect_syn() > 0:
                    is_defective[i,j] = 1
        return is_defective

    def check_middle_data(self,ta,tb):
        #check the data qubit between 2 tiles
        #return True if any of them is defective
        if ta[0] == tb[0] and abs(ta[1]-tb[1]) == 1:#adjacent tiles in the same row
            r = ta[0]
            c = min(ta[1],tb[1])
            # coords are (r,c) and (r,c+1)
            if self.chiplet_array[r][c].right_edge_defect() or self.chiplet_array[r][c+1].left_edge_defect():
                return True
            else:
                return False
        elif ta[1] == tb[1] and abs(ta[0]-tb[0]) == 1:#adjacent tiles in the same col
            r = min(ta[0],tb[0])
            c = ta[1]
            # coords are (r,c) and (r+1,c)
            if self.chiplet_array[r][c].lower_edge_defect() or self.chiplet_array[r+1][c].upper_edge_defect():
                return True
            else:
                return False

    def connected_tiles(self,fixed_freq=False):
        #return the connected components of tiles.
        #2 tiles are linked if they are adjacent, both are defect free and the data qubits in between are non-faulty
        defective_tiles = self.check_tile_defective(fixed_freq)
        G = nx.grid_2d_graph(self.numq_row, self.numq_col)
        #remove defective tiles
        for i in range(self.numq_row):
            for j in range(self.numq_col):
                if defective_tiles[i][j]:
                    G.remove_node((i,j))
        #remove edges between tiles with broken data qubits in middle
        for u,v in G.edges:
            if self.check_middle_data(u,v):
                G.remove_edge(u,v)
        return sorted(nx.connected_components(G), key=len, reverse=True)

    def num_defect_free_tiles(self,fixed_freq=False):
        #return number of defect-free tiles
        defect_mat = self.check_tile_defective(fixed_freq)
        return np.sum(defect_mat)
