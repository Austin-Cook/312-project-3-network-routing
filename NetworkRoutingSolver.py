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

        # the CS312Graph containing all nodes and edges
        self.network = None

    def initializeNetwork(self, network):
        assert (type(network) == CS312Graph)
        self.network = network

    def getShortestPath(self, destIndex):
        # work backwards from last edge to first edge to get path
        path_edges = []
        total_length = 0

        # navigate the pre-filled prev array to build a path to destIndex
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
        t1 = time.time()
        self.dist, self.prev = _dijkstra(network=self.network,
                                         queue_type=(HeapQueue if use_heap else ArrayQueue),
                                         src_index=srcIndex)
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
    nodes = network.getNodes()  # DO NOT MODIFY

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

    def __init__(self, id_to_weight_map: dict):
        """
        Do not call directly; call make_queue() instead.

        :param id_to_weight_map: The initial map of 0 based index id mapped to their distances
        """
        self.id_to_weight_map = id_to_weight_map

    def __str__(self):
        return str(self.id_to_weight_map)

    @staticmethod
    def make_queue(weights: list):
        id_to_weight_map = {}

        # fill the map with node_id (it's index) -> weight
        for node_index in range(len(weights)):
            id_to_weight_map[node_index] = weights[node_index]

        return ArrayQueue(id_to_weight_map)

    def insert(self, node_id: int, weight: float):
        # item should not already be in the map
        assert(self.id_to_weight_map.get(node_id) is None)

        self.id_to_weight_map[node_id] = weight

    def delete_min(self):
        min_id = None
        min_weight = float('inf')

        # find the node with the min weight
        for node_id in self.id_to_weight_map:
            if self.id_to_weight_map[node_id] <= min_weight:
                # found the new lowest item
                min_id = node_id
                min_weight = self.id_to_weight_map[node_id]

        # remove the node with the min weight
        if min_id is not None:
            self.id_to_weight_map.pop(min_id)

        return min_id

    def decrease_weight(self, node_id: int, new_weight):
        # item should already be in the map
        assert(self.id_to_weight_map.get(node_id) is not None)

        # decrease the weight of the node
        self.id_to_weight_map[node_id] = new_weight


class HeapQueue(QueueInterface):

    def __init__(self, heap_array: list, id_to_weight_map: dict, id_to_index_map: dict):
        """
        Do not call directly; call make_queue() instead.

        :param id_to_weight_map: The initial map of 0 based id's mapped to their distances
        :param id_to_index_map: The initial map of 0 based id's mapped to their weights
        """
        # each index contains the id (the initial 0 based index)
        self.heap_array = heap_array
        # id (initial index): weight
        self.id_to_weight_map = id_to_weight_map
        # id (initial index): curr_index
        self.id_to_index_map = id_to_index_map

    def __str__(self):
        return ("heap_array: " + str(self.heap_array) +
                "\nid_to_weight_map: " + str(self.id_to_weight_map) +
                "\nid_to_index_map: " + str(self.id_to_index_map))

    @staticmethod
    def make_queue(weights: list):
        # create the array to represent a heap (each index contains the id/initial index in the array)
        heap_array = [i for i in range(len(weights))]                      # ]- O(n)

        # id (initial index): weight
        id_to_weight_map = {}
        # id (initial index): curr_index
        id_to_index_map = {}

        # initialize the maps
        for node_index in range(len(weights)):
            # initializes as id (initial index): initial weight
            id_to_weight_map[node_index] = weights[node_index]
            # initialized as index: index (the value will change as the heap is altered)
            id_to_index_map[node_index] = node_index

        # create a heap queue with the array and map
        heap_queue = HeapQueue(heap_array, id_to_weight_map, id_to_index_map)

        # reorder the heap_array as a min heap
        heap_queue._build_min_heap()                                        # ]- O(n)

        return heap_queue

    def insert(self, node_id: int, weight: float):
        insert_index = len(self.heap_array) # end of the array

        # make room for one more node
        self.heap_array.append(None)

        # place the item at the end of the heap_array and set the mappings to weight and index
        self._set(insert_index, node_id, weight)

        # percolate up starting at the index of the item just added (last index)
        self._percolate_up(insert_index)

    def delete_min(self):
        if len(self.heap_array) < 1:
            # heap already empty
            return None

        last_item_index = len(self.heap_array) - 1
        old_min_id = self.heap_array[0]

        # set the value of the first item (index 0) to the value of the item at the last index
        last_item_id = self.heap_array[last_item_index]
        last_item_weight = self.id_to_weight_map[last_item_id]
        self._set(0, last_item_id, last_item_weight)

        # clear map entries for the old first item
        self.id_to_weight_map.pop(old_min_id)     # ]- O(1)
        self.id_to_index_map.pop(old_min_id)      # ]- O(1)

        # remove last item from heap
        self.heap_array.pop()                       # ]- O(1)

        # percolate down from the first item (index 0)
        if len(self.heap_array) > 0:
            self._percolate_down(0)

        return old_min_id

    def decrease_weight(self, node_id: int, new_weight):
        # get the node's old_weight from the dictionary
        old_weight = self.id_to_weight_map[node_id]
        index = self.id_to_index_map[node_id]

        # update the node's weight in the dictionary
        self.id_to_weight_map[node_id] = new_weight

        if new_weight < old_weight:
            # percolate up from current index of node_id
            self._percolate_up(index)
        elif new_weight > old_weight:
            # percolate down from current index of node_id
            self._percolate_down(index)

    # HEAP METHODS

    def _set(self, index_to_set: int, node_id: int, weight: float):
        # places a node in the heap at the specified index, and updates the mappings to weight and index

        # sets the heap value at index_to_set to the node_id
        self.heap_array[index_to_set] = node_id

        # update id_to_weight_map with the new weight
        self.id_to_weight_map[node_id] = weight

        # update id_to_index_map with the new index
        self.id_to_index_map[node_id] = index_to_set

    def _swap(self, index_i: int, index_j: int):
        # swaps the values in self.heap_array at indexes i and j and updates the id mapping to weight and index

        # get previous i values
        old_index_i_id = self.heap_array[index_i]
        old_index_i_weight = self.id_to_weight_map[old_index_i_id]

        # get previous j values
        old_index_j_id = self.heap_array[index_j]
        old_index_j_weight = self.id_to_weight_map[old_index_j_id]

        # set i to old j values and update mappings for old_index_j_id
        self._set(index_i, old_index_j_id, old_index_j_weight)

        # set j to old i values and update mappings for old_index_i_id
        self._set(index_j, old_index_i_id, old_index_i_weight)

    def _build_min_heap(self):
        # order all subtrees such that the lowest weight is the parent

        # for each node with at least one child (from high to low)
        for curr_root_index in range((len(self.heap_array) // 2) - 1, -1, -1):
            # heapify the subtree
            self._percolate_down(curr_root_index)

    def _percolate_down(self, curr_index: int):
        # get left and right child indexes and id's
        left_child_index = self._get_left_child_index(curr_index)
        right_child_index = self._get_right_child_index(curr_index)

        # set min_index to curr_index
        min_index = curr_index
        curr_id = self.heap_array[curr_index]
        # set min_weight to weight at curr_id
        min_weight = self.id_to_weight_map[curr_id]

        # get child with the lowest weight
        if left_child_index is not None:
            # left child exists
            left_child_id = self.heap_array[left_child_index]
            left_child_weight = self.id_to_weight_map[left_child_id]
            if left_child_weight < min_weight:
                # left_child has new lowest weight
                min_index = left_child_index
                min_weight = left_child_weight

        if right_child_index is not None:
            # right child exists
            right_child_id = self.heap_array[right_child_index]
            right_child_weight = self.id_to_weight_map[right_child_id]
            if right_child_weight < min_weight:
                # right_child has new lowest weight
                min_index = right_child_index

        # swap the parent with the lower child
        if min_index != curr_index:
            # parent is not already the lowest

            # swap the parent with the lowest child
            self._swap(curr_index, min_index)

            # recursively call _percolate_down on the child's tree
            self._percolate_down(min_index)

    def _percolate_up(self, curr_index):
        parent_index = self._get_parent_index(curr_index)

        if parent_index is not None:
            # parent exists
            parent_id = self.heap_array[parent_index]
            curr_id = self.heap_array[curr_index]
            if self.id_to_weight_map[parent_id] > self.id_to_weight_map[curr_id]:
                # parent weight > curr_weight

                # swap the node with its parent
                self._swap(curr_index, parent_index)

                # recursively call _percolate_up on the parent_index
                self._percolate_up(parent_index)

    def _get_left_child_index(self, index: int):
        # gets the index of the left child and returns None if it doesn't exist
        left_index = (index * 2) + 1

        return left_index if left_index < len(self.heap_array) else None

    def _get_right_child_index(self, index: int):
        # gets the index of the right child and returns None if it doesn't exist
        right_index = (index * 2) + 2

        return right_index if right_index < len(self.heap_array) else None

    @staticmethod
    def _get_parent_index(index: int):
        # gets the index of the parent and returns None if already at top of tree
        parent_index = ceil(index / 2) - 1

        return parent_index if parent_index != -1 else None
