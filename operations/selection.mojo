from collections import List

# ---------------------------------
#  Heap utilities for sorted Top‑k
# ---------------------------------

def _sift_down(
        mut values:  List[Float32],
        mut indices: List[Int],
        startpos: Int, pos: Int
    ):
    var new_val =  values[pos]
    var new_idx =  indices[pos]
    var p = pos
    while p > startpos:
        var parent = (p - 1) >> 1
        if new_val < values[parent]:
            values[p]  = values[parent]
            indices[p] = indices[parent]
            p = parent
            continue
        break
    values[p]  = new_val
    indices[p] = new_idx


def _sift_up(mut values: List[Float32], 
            mut indices: List[Int], 
            pos: Int, 
            length: Int):
    var startpos = pos
    var new_value = values[pos]
    var new_index = indices[pos]
    var childpos = 2 * pos + 1
    while childpos < length:
        var rightpos = childpos + 1
        if rightpos < length and values[rightpos] < values[childpos]:
            childpos = rightpos
        values[pos] = values[childpos]
        indices[pos] = indices[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    values[pos] = new_value
    indices[pos] = new_index
    _sift_down(values, indices, startpos, pos)


def heap_push(mut values: List[Float32], 
              mut indices: List[Int], 
              value: Float32, 
              index: Int, 
              length: Int):
    values[length] = value
    indices[length] = index
    _sift_down(values, indices, 0, length)


