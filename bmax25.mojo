from collections import List
from operations.selection import heap_push, _sift_down, _sift_up
from operations.scoring import compute_relevance_from_scores


def main():

    # ──────────────────────────────────────────────
    # 1· heap_push test (unchanged)
    # ──────────────────────────────────────────────
    var values  = List[Float32](0.0, 1.0, 2.0)
    var indices = List[Int](0, 1, 2)

    # reserve one extra slot so heap_push can write in‑place
    values.append(0.0)
    indices.append(0)

    # push a new (value, index) pair
    heap_push(values, indices, 0.5, 3, 3)

    print("after heap_push:")
    print("  values :", values.__repr__())
    print("  indices:", indices.__repr__())

    # ──────────────────────────────────────────────
    # 2· sift_up test (new)
    # ──────────────────────────────────────────────
    # build a valid min‑heap
    var h_vals = List[Float32](0.2, 0.8, 0.6, 1.5, 2.1)
    var h_idx  = List[Int](10,  11,  12,  13,  14)

    print("\noriginal heap:")
    print("  values :", h_vals.__repr__())
    print("  indices:", h_idx.__repr__())

    # deliberately break the heap property at the root
    h_vals[0] = 3.3
    h_idx[0]  = 99

    # restore heap order
    _sift_up(h_vals, h_idx, 0, len(h_vals))

    print("\nafter _sift_up:")
    print("  values :", h_vals.__repr__())
    print("  indices:", h_idx.__repr__())

    
    # Example miniature CSR index with two docs and three tokens
    var data     = List[Float32](1.2, 0.7, 2.5, 3.1)
    var indices1  = List[Int]    (0,   1,   0,   1)      # doc ids
    var indptr   = List[Int]    (0,   2,   3,   4)      # row ptr (3 tokens → 4 entries)
    var q_tokens = List[Int]    (0, 2)                  # query contains token 0 and 2

    var scores = compute_relevance_from_scores(data, indptr, indices1, 2, q_tokens) # 2 is num of docs

    # Expected: doc0 = 1.2 + 2.5 = 3.7, doc1 = 0.7 + 3.1 = 3.8
    print(scores.__repr__())  # → [3.7, 3.8]
