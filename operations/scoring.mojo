from collections import List
from algorithm   import parallelize
from memory      import memset_zero, size_of      # optional but handy


# ──────────────────────────────────────────────────────────────
#  Utility: make a zero‑filled list of a given length
# ──────────────────────────────────────────────────────────────
fn make_zeros(len: Int32) -> List[Float32]:
    var buf = List[Float32]()
    buf.reserve(len)
    for _ in range(len):
        buf.append(0.0)
    return buf


# ──────────────────────────────────────────────────────────────
#  Relevance accumulation for BM25‑style CSR index
# ──────────────────────────────────────────────────────────────
# • data         – BM25 weights stored row‑wise (CSR “data”)
# • indptr       – CSR row‑pointer (len = vocab_size + 1)
# • indices      – column indices for each weight (doc ids)
# • num_docs     – total number of documents in the corpus
# • query_tokens – list of token ids occurring in the query
#
# Returns a length‑num_docs vector of relevance scores.
# ──────────────────────────────────────────────────────────────
fn compute_relevance_from_scores(
        data:         List[Float32],
        indptr:       List[Int32],
        indices:      List[Int32],
        num_docs:     Int32,
        query_tokens: List[Int32]
    ) -> List[Float32]:

    # ── contiguous output (single alloc) ─────────────────────────
    var scores     = make_zeros(num_docs)
    var scores_ptr = scores.as_mut_ptr()          # *mut Float32

    # ── raw input pointers (no bounds checks) ───────────────────
    var data_ptr    = data.as_ptr()               # *const Float32
    var indptr_ptr  = indptr.as_ptr()             # *const Int32
    var indices_ptr = indices.as_ptr()            # *const Int32

    # The inner term‑accumulation kernel
    @parameter
    fn add_term_rows(qi: Int32) -> None:
        var t     = query_tokens[qi]
        var start = indptr_ptr[t]
        var stop  = indptr_ptr[t + 1]

        var j = start
        while j < stop:
            var doc = indices_ptr[j]
            scores_ptr[doc] += data_ptr[j]        # no bounds checks
            j += 1

    # Parallel over the (usually small) query‑token list
    parallelize[add_term_rows](len(query_tokens), /*num_workers=*/4)

    return scores
