from NetworkRoutingSolver import NetworkRoutingSolver, ArrayQueue, HeapQueue
from CS312Graph import CS312GraphNode

def run_tests():
    # [0, inf, inf, inf]
    # dist = [0, float('inf'), float('inf'), float('inf')]
    # array_queue = ArrayQueue.make_queue(dist)
    #
    # array_queue.insert(4, 16)
    #
    # print(array_queue)
    #
    # array_queue.decrease_weight(2, 100)
    # print(array_queue)

    # dist = [float('inf'), 2, float('inf'), float('inf'), 5]
    dist = [0, 1, 2, 3, 4, 5, 6, 7]
    heap_queue = HeapQueue.make_queue(dist)

    print(heap_queue)
    print("")

    heap_queue.decrease_weight(0, 5.5)
    print(heap_queue)
    print("")



    #network = graph...


if __name__ == "__main__":
    run_tests()
