from operations.selection import heap_push,_sift_down


def main():
    #values = List[Float32](0.,1.,2.)
    #indices = List[Int](0,1,2)
    #_sift_down(values,indices,0,1)
    #print(values.__repr__())
    # Seed heap with 3 items
    var values  = List[Float32](0.0, 1.0, 2.0)
    var indices = List[Int](0, 1, 2)

    # Reserve one extra slot so heap_push can write in‑place
    values.append(0.0)     # dummy placeholders
    indices.append(0)

    # Push the pair (0.5, 3) — 0.5 is smaller than root (0.0),
    # so the heap order will adjust.
    heap_push(values, indices, 0.5, 3, 3)

    print("heap values :", values.__repr__())
    print("heap indices:", indices.__repr__())
