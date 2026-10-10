#pragma once
#include <cstddef>
#include <vector>

namespace kvmem {

// Default top-k for T_i (ActKV Sec. 6.1 / Sec. 8: k = 20).
constexpr int kDefaultConfTopK = 20;

// ActKV token-level confidence (arXiv:2609.31395, Sec. 6.1):
//   T_i = -(1/k) * sum_{j=1..k} log P_i(j)
// with P_i the softmax of `logits` and the sum over the k LARGEST logits.
// T is large for a peaked distribution (every log P close to 0) and small for
// a flat one, so HIGHER T = MORE confident.
//
// `logits` must be the RAW model logits. Temperature, repeat penalties and
// grammar masking all reshape the distribution and would turn the signal into
// a function of the sampling configuration instead of the model -- temperature
// < 1 in particular inflates T and hides exactly the degradation this is
// meant to detect.
//
// Returns a quiet NaN when the input is unusable (null, n <= 0, k <= 0, or a
// degenerate distribution). Callers must drop NaN instead of feeding it to
// KvMemConfidenceMonitor: one NaN poisons every pooled minimum that contains
// it and the slope becomes NaN, which compares false against < 0 and silently
// disables budget growth.
float token_confidence(const float * logits, int n, int k = kDefaultConfTopK);

// ActKV confidence-driven adaptive budget (arXiv:2609.31395, Sec. 6.1 Alg. 2).
//
// Token-level confidence is derived from the predicted distribution:
//   T_i = -(1/k) * sum_{j=1..k} log P_i(j)      (k = 20 in the paper)
// i.e. the negative mean log-probability of the top-k tokens. A peaked
// distribution makes every log P close to 0, so T is LARGE for a confident
// token; a flat distribution makes them strongly negative, so T is SMALL.
// Higher T = more confident. (The caller computes T; this class consumes it.)
//
// Raw T is noisy, so it is aggregated in three steps:
//   1. R'_m = min over a sliding window of w tokens, stride s  (min pooling)
//   2. R    = the bottom b% of R'                              (worst segments)
//   3. alpha = first-order least-squares slope of R            (trend)
// Minimum pooling and bottom-b% selection jointly emphasise localised
// degradation -- the model restating missing context ("wait", "I guess",
// "let me check") when the compressed KV cache lost action-critical entries.
//
// Sec. 6.2: the budget is raised when alpha < 0 and NEVER lowered on recovery.
// "When confidence stabilizes or recovers, we retain the current budget, as
// this improvement suggests that the allocation is sufficient but does not
// establish that a smaller budget would suffice." Shrinking re-introduces the
// information loss that caused the confidence drop in the first place.
struct KvMemConfidenceConfig {
    uint32_t pool_size = 64;      // w: sliding window over token confidences
    uint32_t stride = 8;          // s: window stride between pooled minima
    float    bottom_frac = 0.10f; // b: bottom fraction of R' kept as R
    uint32_t min_pool = 8;        // min pooled points before a slope is trusted
};

class KvMemConfidenceMonitor {
public:
    explicit KvMemConfidenceMonitor(KvMemConfidenceConfig cfg = {});

    // Feed one token-level confidence T_i (higher = more confident).
    void observe(double token_conf);
    void reset();

    // Token confidences currently buffered (capped at cfg.pool_size).
    uint32_t size() const;
    // Pooled minima available for the fit (R').
    uint32_t pooled() const;
    uint32_t tokens() const;
    bool ready() const;

    // Least-squares slope alpha of R ordered by arrival time. Only valid when
    // ready(); returns 0.0 otherwise. alpha < 0 means confidence is decaying.
    double slope() const;
    // True when the confidence trend is negative: the KV budget is too small.
    bool should_increase_budget() const;

private:
    void pool_one();
    double min_window() const;

    KvMemConfidenceConfig cfg_;
    std::vector<double> tokens_;   // ring of the last w token confidences
    std::vector<double> pooled_;   // ring of the last N pooled minima (R')
    size_t token_head_ = 0;        // write position in tokens_ once full
    size_t pooled_head_ = 0;       // write position in pooled_ once full
    uint64_t n_tokens_ = 0;        // total token confidences observed
    uint64_t n_pooled_ = 0;        // total pooled minima produced
};

} // namespace kvmem
