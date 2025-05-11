from collections import List
from memory.span import Span          # ← real generic view type
from memory       import memset_zero  # still optional
from algorithm    import parallelize


fn compute_relevance_from_scores(
        data:         Span[Float32],
        indptr:       Span[Int32],
        indices:      Span[Int32],
        num_docs:     Int32,
        query_tokens: Span[Int32]
    ) -> List[Float32]:

    # ── contiguous output ───────────────────────────────────────────
    var scores     = List[Float32].filled(num_docs, 0.0)
    let scores_ptr = scores.as_mut_ptr()

    # ── raw pointers (no bounds checks in the hot loop) ─────────────
    let data_ptr    = data.unsafe_ptr()
    let indptr_ptr  = indptr.unsafe_ptr()
    let indices_ptr = indices.unsafe_ptr()

    @parameter
    fn add_term_rows(qi: Int32) -> None:
        let t     = query_tokens[qi]
        let start = indptr_ptr[t]
        let stop  = indptr_ptr[t + 1]

        var j = start
        while j < stop:
            let doc = indices_ptr[j]
            scores_ptr[doc] += data_ptr[j]
            j += 1

    # parallel over query terms (usually just a handful)
    parallelize[add_term_rows](len(query_tokens), /*num_workers=*/4)

    return scores
