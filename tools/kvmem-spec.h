#pragma once

// Shared draft-mtp wiring for llama-kvmem-cli / llama-kvmem-server.
// Does not reimplement llama.cpp's speculative state machine.

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <functional>
#include <string>
#include <vector>

struct kvmem_spec_opts {
    // Measured optimum sits at 4-5; see the comment in llama-kvmem-server.cpp.
    int32_t n_max = 4;
    int32_t n_min = 0;
    float   p_min = 0.0f;
    int32_t n_gpu_layers = 99;
    int32_t n_ctx = 0;
    int32_t n_batch = 512;
    int32_t n_ubatch = 512;
    int32_t n_threads = -1;
    int32_t n_threads_batch = -1;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool    kvmem_enabled = false;
    std::string draft_model; // optional sidecar GGUF; empty = welded nextn
    ggml_type type_k = GGML_TYPE_Q8_0;
    ggml_type type_v = GGML_TYPE_Q8_0;
    ggml_type draft_type = GGML_TYPE_COUNT; // inherit target K/V types unless overridden

    // Speculative implementations, in llama.cpp priority order. Multiple entries
    // are *fallback*, not additive: common_speculative_draft() walks impls and
    // stops at the first one that produces a non-empty draft
    // (llama.cpp/common/speculative.cpp, common_speculative_draft).
    std::vector<common_speculative_type> types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };

    // n-gram self-speculation. No draft model, no extra VRAM: the draft is a
    // verbatim copy of an earlier token run found in the target's own history.
    // Upstream defaults are size_n=12 / size_m=48; 48 makes the verify step a
    // 49-token batch that is mostly rejected, so we start at 16.
    uint16_t ngram_size_n    = 12;
    uint16_t ngram_size_m    = 16;
    uint16_t ngram_min_hits  = 1;
};

// Supported cache types: f16, f32, q8_0, q5_0, q4_0.
// q5_0 requires GGML_CUDA_FA_QUANTS=all (enabled by this project's build).
ggml_type kvmem_parse_cache_type(const char * s, bool * ok);
// Quantized K/V may independently use q8_0, q5_0 or q4_0; no float/quantized mixing.
bool kvmem_cache_types_ok(ggml_type type_k, ggml_type type_v);

// Which n-gram knob a command-line flag refers to.
enum kvmem_ngram_field {
    KVMEM_NGRAM_NONE = 0,
    KVMEM_NGRAM_SIZE_N,
    KVMEM_NGRAM_SIZE_M,
    KVMEM_NGRAM_MIN_HITS,
    // Upstream removed --spec-ngram-size-n/-m/-min-hits (llama.cpp master,
    // common/arg.cpp: "the argument has been removed. use the respective
    // --spec-ngram-*-size-n"). KVMem mirrors that removal instead of silently
    // reviving the dead names.
    KVMEM_NGRAM_REMOVED,
};

// Upstream llama.cpp n-gram flags (common/arg.cpp), one set per implementation:
//   --spec-ngram-simple-size-n / -size-m / -min-hits
//   --spec-ngram-map-k-size-n  / -size-m / -min-hits
//   --spec-ngram-map-k4v-size-n / -size-m / -min-hits
//   --spec-ngram-mod-n-max (draft length) / -n-min / -n-match (lookup length)
// plus the three names upstream deleted. Returns KVMEM_NGRAM_NONE when the flag
// is not an n-gram size flag.
kvmem_ngram_field kvmem_ngram_flag_field(const char * arg);

// Parse a --spec-type value; empty/false on unknown input.
// Accepted: none | draft-mtp | ngram-simple | ngram-map-k | ngram-map-k4v |
//           ngram-mod | ngram-cache
bool kvmem_parse_spec_type(const char * s, common_speculative_type * out);
const char * kvmem_spec_type_name(common_speculative_type t);

// True when any requested type needs a draft context (MTP / EAGLE / DFlash / DSpark).
// N-gram types never do - they are pure CPU history matching.
bool kvmem_spec_needs_draft_ctx(const std::vector<common_speculative_type> & types);

// Widest draft any requested type can produce; drives n_outputs_max / n_rs_seq.
int32_t kvmem_spec_draft_width(const kvmem_spec_opts & opts);

// Warn when n_rs_seq is big enough that the recurrent rollback planes eat the
// VRAM budget. Each unit of n_rs_seq reserves one extra recurrent-state plane
// (~144 MiB on Qwen3.8-27B / 64 recr layers). Measured on a 16 GB card:
// n_rs_seq 16 -> 45.8 tok/s, 24 -> 36.2, 32 -> 7.2 (overflow).
void kvmem_spec_warn_rs(uint32_t n_rs_seq);

// GDN replay (--kvmem-mtp-state replay, the server default when MTP is on) is
// only implemented for n_rs_seq in [1,5]: use_gdn_replay() in
// src/adapter/llama-memory-kvmem-hybrid.cpp throws
//     "GDN replay requires single-sequence GPU Qwen 27B with MTP 1-5 ..."
// once n_rs_seq exceeds 5. An n-gram draft width of 16 would otherwise abort
// context creation. Replay rolls back via its own transaction, not via the
// planes, so clamping to 5 costs nothing there.
uint32_t kvmem_spec_clamp_rs(uint32_t width, int mtp_state);

struct kvmem_spec_session {
    common_params spec_params;
    common_speculative_init_result_ptr init;
    common_speculative * spec = nullptr;
    llama_context * ctx_dft = nullptr;
    bool use_ckpt_tgt = false;
    bool use_ckpt_dft = false;
    bool use_gdn_replay = false;
    uint32_t n_rs_tgt = 0;
    bool ok = false;
};

bool kvmem_spec_start(kvmem_spec_session & sess,
                      llama_model * model_tgt,
                      llama_context * ctx_tgt,
                      const kvmem_spec_opts & opts);

void kvmem_spec_stop(kvmem_spec_session & sess);

// decode_span / spec_decode_span: client disconnected mid-prefill.
constexpr int KVMEM_DECODE_ABORT = -100;

// Decode [pos0, pos1) with explicit pos/seq_id so process() can catch up draft KV.
// `abort` if set: return KVMEM_DECODE_ABORT when it returns true (checked between batches).
int kvmem_spec_decode_span(llama_context * ctx,
                           common_speculative * spec,
                           const llama_token * toks,
                           int pos0, int pos1, int n_batch,
                           const char * what,
                           const std::function<bool()> & abort = {});

using kvmem_spec_on_token =
        std::function<void(llama_token id, const std::string & piece, bool from_draft)>;

struct kvmem_spec_gen_stats {
    int n_past = 0;
    int n_gen = 0;
    int n_drafted = 0;
    int n_accept = 0;
    int n_restore = 0;
    bool failed = false;
};

// `prompt` is the full prompt; the last token is id_last and must not already
// be in the target KV (same contract as llama.cpp speculative-simple).
kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        float temp,
        kvmem_spec_on_token on_token);

// Same as above, but uses a fully-specified sampler (grammar / lazy triggers).
// `abort` if set: stop the generate loop when it returns true (not a failure).
kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        common_params_sampling sparams,
        kvmem_spec_on_token on_token,
        const std::function<bool()> & abort = {},
        llama_pos position_offset = 0);
