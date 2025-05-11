from collections import List

def compute_relevance_from_scores(
        data:         List[Float32],
        indptr:       List[Int],
        indices:      List[Int],
        num_docs:     Int,
        query_tokens: List[Int]
    ) -> List[Float32]:

    # allocate result vector
    var scores = List[Float32]()
    for _ in range(num_docs):
        scores.append(0.0)

    # ── main accumulation ─────────────────────────────────────
    for qi in range(len(query_tokens)):        # <-- use plain indices
        var t = query_tokens[qi]               #     t is now an Int
        var start = indptr[t]
        var end   = indptr[t + 1]

        for j in range(start, end):
            var doc_id = indices[j]
            scores[doc_id] += data[j]

    return scores

