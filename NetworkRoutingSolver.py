#!/usr/bin/python3


from CS312Graph import *
import time
from abc import ABC, abstractmethod
from math import ceil


class NetworkRoutingSolver:
    def __init__(self):
        # list of previous nodes from last run of computeShortestPaths()
        # prev: [None, 0, 1, 2, 0, 6, 2, 6]
        self.prev = None

        # list of distances from the starting node to each node from the last run of computeShortestPaths()
        # dist: [0, 1, 3, 4, 4, 6, 5, 6]
        self.dist = None

    def initializeNetwork(self, network):
        assert (type(network) == CS312Graph)
        self.network = network

    def getShortestPath(self, destIndex):
        # self.dest = destIndex # TODO DELETEME

        # TODO - probably don't need
        # create dictionary for constant edge lookup { node: dictionary { edge_dest_node: edge }}

        # work backwards from last edge to first edge to get path
        path_edges = []
        total_length = 0

        prev_index = destIndex
        while True:
            curr_index = prev_index
            prev_index = self.prev[curr_index]

            if prev_index is None:
                # no more edges to traverse
                break

            # get the edge from prev -> curr
            prev = self.network.nodes[prev_index]
            curr = self.network.nodes[curr_index]
            edge_prev_to_curr = None
            for edge in prev.neighbors:
                if edge.dest == curr:
                    # found the edge from prev -> curr
                    edge_prev_to_curr = edge
            assert edge_prev_to_curr is not None    # the edge should exist

            # append the edge and increase the total length
            path_edges.append((edge_prev_to_curr.src.loc, edge_prev_to_curr.dest.loc, '{:.0f}'.format(edge_prev_to_curr.length)))
            total_length += edge_prev_to_curr.length

        # we built the path as last_edge, ..., first_edge
        path_edges.reverse()

        if len(path_edges) == 0:
            # there is no path to destIndex
            total_length = float('inf')

        return {'cost': total_length, 'path': path_edges}

    def computeShortestPaths(self, srcIndex, use_heap=False):
        # self.source = srcIndex    # TODO Deleteme
        t1 = time.time()
        self.dist, self.prev = _dijkstra(network=self.network, queue_type=ArrayQueue, src_index=srcIndex)
        t2 = time.time()

        return t2 - t1


def _dijkstra(network: CS312Graph, queue_type, src_index: int):
    """
    Runs Dijkstra's algorithm to find the shortest path from the node with index
    srcIndex (0 based) in the network (CS312Graph object) to all other nodes

    :param queue_type: The type of queue to use in this algorithm
    :return: 'prev' array from which to derive each path from the starting index
    """

    # SETUP

    # get all nodes
    nodes = network.getNodes() # DO NOT MODIFY

    # map nodes to their index for quick lookup
    node_to_index_map = {}
    i = 0
    for u in nodes:                                             # ]- O(n)
        node_to_index_map[u] = i
        i += 1

    # initialize dist array (for all u in V: dist(u) <- infinity)
    dist = [float('inf')] * len(nodes)

    # initialize prev array (for all u in V: prev(u) <- NULL)
    prev = [None] * len(nodes)

    # dist(s) <- 0 # self.source contains the srcIndex
    dist[src_index] = 0

    # H.makequeue(V) {distances as keys}
    H = queue_type.make_queue(dist)

    # ALGORITHM

    # while H is not empty do
    while (u_index := H.delete_min()) is not None:
        # for all edges (u, v) in E do    # just edges from the current node
        for edge in (nodes[u_index]).neighbors:
            v_index = node_to_index_map[edge.dest]

            assert u_index == node_to_index_map[edge.src]   # TODO Deleteme

            # if dist(v) > dist(u) + l(u, v) then
            alt_path_dist = dist[u_index] + edge.length
            if dist[v_index] > alt_path_dist:
                # dist(v) <- dist(u) + l(u, v)
                dist[v_index] = alt_path_dist
                # prev(v) <- u
                prev[v_index] = u_index
                # H.decreasekey(v)
                H.decrease_weight(v_index, alt_path_dist)

    return dist, prev


class QueueInterface(ABC):
    """
    An interface for a Priority Queue
    It also keeps track of the node_id corresponding to each weight
    """

    @staticmethod
    @abstractmethod
    def make_queue(weights: list):
        """
        Creates a queue with the weights from the list\n
        NOTE - the node_id for each weight is its index in the list (0 based)

        :param weights: The list of weights from which to make the queue; the index of each weight is its id
        :return An queue filled with the given weights
        """
        pass

    @abstractmethod
    def insert(self, node_id: int, weight: float):
        """
        Inserts an item into the priority queue\n
        NOTE - The dict will store node_id->weight

        :param node_id: The index/id of the node to insert
        :param weight: The current weight of the node to insert
        """
        pass

    @abstractmethod
    def delete_min(self):
        """
        Deletes the min item from the priority queue

        :return: the node_id of the node with the min weight, None if the queue is empty
        """
        pass

    @abstractmethod
    def decrease_weight(self, node_id: int, new_weight):
        """
        Decreases the item node_id's value to new_weight and percolates as necessary

        :param node_id: The index/id of the node for which to decrease the weight
        :param new_weight: The new weight value for the node specified by node_id
        """
        pass


class ArrayQueue(QueueInterface):
    # just implement it as a map

    def __init__(self, index_to_weight_map: dict):
        """
        Do not call directly; call make_queue() instead.

        :param index_to_weight_map: The initial map of 0 based index id mapped to their distances
        """
        self.index_to_weight_map = index_to_weight_map

    def __str__(self):
        return str(self.index_to_weight_map)

    @staticmethod
    def make_queue(weights: list):
        index_to_weight_map = {}

        # fill the map with node_id (it's index) -> weight
        for node_index in range(len(weights)):
            index_to_weight_map[node_index] = weights[node_index]

        return ArrayQueue(index_to_weight_map)

    def insert(self, node_id: int, weight: float):
        # item should not already be in the map
        assert(self.index_to_weight_map.get(node_id) is None)

        self.index_to_weight_map[node_id] = weight

    def delete_min(self):
        min_id = None
        min_weight = float('inf')

        # find the node with the min weight
        for node_id in self.index_to_weight_map:
            if self.index_to_weight_map[node_id] <= min_weight:
                # found the new lowest item
                min_id = node_id
                min_weight = self.index_to_weight_map[node_id]

        # remove the node with the min weight
        if min_id is not None:
            self.index_to_weight_map.pop(min_id)

        return min_id

    def decrease_weight(self, node_id: int, new_weight):
        # item should already be in the map
        assert(self.index_to_weight_map.get(node_id) is not None)

        # decrease the weight of the node
        self.index_to_weight_map[node_id] = new_weight


class HeapQueue(QueueInterface):

    def __init__(self, heap_array: list, index_to_weight_map: dict):
        """
        Do not call directly; call make_queue() instead.

        :param index_to_weight_map: The initial map of 0 based index id mapped to their distances
        """
        self.heap_array = heap_array
        self.index_to_weight_map = index_to_weight_map

    def __str__(self):
        pass
        # TODO

    @staticmethod
    def make_queue(weights: list):
        # create the array to represent a heap
        heap_array = weights.copy()                                         # ]- O(n)

        # fill the map with node_id (it's initial index) -> weight
        index_to_weight_map = {}
        # TODO add
        for node_index in range(len(weights)):
            index_to_weight_map[node_index] = weights[node_index]

        # create a heap queue with the array and map
        heap_queue = HeapQueue(heap_array, index_to_weight_map)

        # reorder the heap_array as a min heap
        heap_queue._build_min_heap()                                        # ]- O(n)

        return heap_queue

    def insert(self, node_id: int, weight: float):
        # append the item to the heap_array
        # percolate up starting at the index of the item just added (last index)
        pass

    def delete_min(self):
        # set the value of the first item (index 0) to the value of the item at the last index
        # delete the last item in the array
        # percolate down from the first item (index 0)
        pass

    def decrease_weight(self, node_id: int, new_weight):
        # get the node's old_weight from the dictionary
        # update the node's weight in the dictionary
        # if new_weight < old_weight
        #   percolate up from node_id
        # else if new_weight > old_weight
        #   percolate down at old_weight
        pass

    # HEAP METHODS

    def _build_min_heap(self):
        # for each node with at least one child (from high to low)
        for curr_root_index in range((len(self.heap_array) // 2) - 1, -1, -1):
            # heapify the subtree
            self._percolate_down(curr_root_index)

    def _percolate_down(self, curr_index: int):
        # get left child index
        # get right child index

        # set min_index to curr_index
        # set min_val to value at curr_index

        # if left_child_index is not None and value at left_child_index is less than min_val
        #   set min_index to left_child_index
        #   set min_val to value at left_child_index
        # if right child index is not None and value at right_child_index is less than min_val
        #   set min_index to right_child_index
        #   set min_val to value at left_child_index

        # if min_index is not curr_index
        #   swap values at curr_index and min_index
        #   recursively call _percolate_down on the child's tree
        pass

    def _percolate_up(self, curr_index):
        # if the parent index is greater than the curr_index
        #   swap values at curr_index and parent_index
        #   if get the parent index is not None (top of tree)
        #       recursively call _percolate_up on the parent_index
        pass

    def _get_left_child_index(self, index: int):
        # gets the index of the left child and returns None if it doesn't exist
        left_index = (index * 2) + 1

        return left_index if left_index < len(self.heap_array) else None

    def _get_right_child_index(self, index: int):
        # gets the index of the right child and returns None if it doesn't exist
        right_index = (index * 2) + 1

        return right_index if right_index < len(self.heap_array) else None

    @staticmethod
    def _get_parent_index(index: int):
        # gets the index of the parent and returns None if already at top of tree
        parent_index = ceil(index / 2) - 1

        return parent_index if parent_index != -1 else None

    def _swap(self, i: int, j: int):
        # swaps the values in self.heap_array at indexes i and j
        temp = self.heap_array[i]
        self.heap_array[i] = j
        self.heap_array[i] = temp

        # TODO update the map of id->index
