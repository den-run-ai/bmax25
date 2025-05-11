from memory import memset_zero
from algorithm import parallelize

fn compute_relevance_from_scores(
        data:    Slice[Float32],     # ← zero‑cost view
        indptr:  Slice[Int32],
        indices: Slice[Int32],
        num_docs: Int32,
        query_tokens: Slice[Int32]
    ) -> List[Float32]:

    # --- contiguous output ----------------------------------------------------
    var scores = List[Float32].filled(num_docs, 0.0)   # single alloc
    let scores_ptr = scores.as_mut_ptr()               # raw pointer view

    # --- fast paths -----------------------------------------------------------
    let data_ptr    = data.as_ptr()
    let indptr_ptr  = indptr.as_ptr()
    let indices_ptr = indices.as_ptr()

    @parameter
    fn add_term_rows(qi: Int32) -> None:
        let t     = query_tokens[qi]
        let start = indptr_ptr[t]
        let stop  = indptr_ptr[t + 1]

        var j = start
        while j < stop:
            let doc = indices_ptr[j]
            scores_ptr[doc] += data_ptr[j]         # no bounds checks
            j += 1

    # Parallelise over query‑terms (usually only a handful of ints)
    parallelize[add_term_rows](query_tokens.len(), /*num_threads=*/4)

    return scores
