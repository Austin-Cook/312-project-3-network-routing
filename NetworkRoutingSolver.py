#!/usr/bin/python3


from CS312Graph import *
import time
from abc import ABC, abstractmethod
from typing import Type


class NetworkRoutingSolver:
    def __init__(self):
        # the array of previous nodes from last run of computeShortestPaths()
        # prev: [None, 0, 1, 2, 0, 6, 2, 6]
        # is    [  -,  a, b, c, a, g, c, g]
        self.prev = None

        # the array of distances to each node from the starting node from last run of computeShortestPaths()
        # dist: [0, 1, 3, 4, 4, 6, 5, 6]
        self.dist = None

    def initializeNetwork(self, network):
        assert (type(network) == CS312Graph)
        self.network = network

    def getShortestPath(self, destIndex):
        self.dest = destIndex
        # TODO: RETURN THE SHORTEST PATH FOR destIndex
        #       INSTEAD OF THE DUMMY SET OF EDGES BELOW
        #       IT'S JUST AN EXAMPLE OF THE FORMAT YOU'LL 
        #       NEED TO USE

        # # old code
        # path_edges = []
        # total_length = 0
        # node = self.network.nodes[self.source]
        # edges_left = 3
        # while edges_left > 0:
        #     edge = node.neighbors[2]
        #     path_edges.append((edge.src.loc, edge.dest.loc, '{:.0f}'.format(edge.length)))
        #     total_length += edge.length
        #     node = edge.dest
        #     edges_left -= 1

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
        self.source = srcIndex
        t1 = time.time()
        # TODO: RUN DIJKSTRA'S TO DETERMINE SHORTEST PATHS.
        #       ALSO, STORE THE RESULTS FOR THE SUBSEQUENT
        #       CALL TO getShortestPath(dest_index)
        self._dijkstra(ArrayQueue)
        # TODO use proper queue

        print("self.prev: ", self.prev)
        print("self.dist: ", self.dist)


        t2 = time.time()
        return (t2 - t1)

    def _dijkstra(self, queue_type):
        """
        Runs Dijkstra's algorithm to find the shortest path from the node with index
        srcIndex (0 based) in the network (CS312Graph object) to all other nodes
        NOTE self.source must be set prior to running this call computeShortestPaths()

        :param Q: The queue to use in this algorithm
        :return: 'prev' array from which to derive each path from the starting index
        """

        # SETUP

        # get all nodes
        nodes = self.network.getNodes() # DO NOT MODIFY

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
        dist[self.source] = 0

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
                if dist[v_index] > alt_path_dist:   # TODO How to represent this correctly??
                    # dist(v) <- dist(u) + l(u, v)
                    dist[v_index] = alt_path_dist
                    # prev(v) <- u
                    prev[v_index] = u_index
                    # H.decreasekey(v)
                    H.decrease_weight(v_index, alt_path_dist)
        # set self.dist and self.prev
        self.dist = dist
        self.prev = prev


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

        # fill the map with index_in_distances->distance_at_index
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

    def __init__(self, heap_array, index_to_weight_map: dict):
        """
        Do not call directly; call make_queue() instead.

        :param index_to_weight_map: The initial map of 0 based index id mapped to their distances
        """
        # self.index_to_weight_map = index_to_weight_map
        self.heap_array = heap_array

    def __str__(self):
        pass
        # TODO

    @staticmethod
    def make_queue(weights: list):
        pass
        # make a copy of weights
        heap_array = weights.copy()

        # TODO heapify
        # _build_min_heap(heap_array) # ]- O(n)

        # return HeapQueue(heap_array)

    def insert(self, node_id: int, weight: float):
        pass

    def delete_min(self):
        pass

    def decrease_weight(self, node_id: int, new_weight):
        pass

    # heap methods

    def _build_min_heap(self):
        pass

    def _heapify(self):
        pass
