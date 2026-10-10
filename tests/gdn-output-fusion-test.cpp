#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void fill(ggml_tensor * t, unsigned seed, float scale) {
    std::vector<float> data(ggml_nelements(t));
    for (float & x : data) {
        seed = seed * 1664525u + 1013904223u;
        x = (float(seed >> 8) / 8388608.0f - 1.0f) * scale;
    }
    ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
}

struct result {
    std::vector<float> data;
    double us;
    size_t buffer_bytes;
};

static result run(ggml_backend_t backend, ggml_backend_t cpu, const std::string & name, int tokens, int seqs,
                  int width, bool stride, bool gate_stride, bool row_weights, bool reverse,
                  bool observe, int repeats) {
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "context allocation failed");
    auto * graph = ggml_new_graph(ctx);
    auto * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width * (stride ? 2 : 1), 48, tokens, seqs);
    auto * gate_input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width * (gate_stride ? 2 : 1), 48, tokens, seqs);
    auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, row_weights ? 48 : 1);
    for (auto * t : {input, gate_input, weight}) {
        ggml_set_input(t);
        ggml_set_output(t);
    }
    auto * x_storage = ggml_scale(ctx, input, 0.75f);
    auto * g_storage = ggml_scale(ctx, gate_input, 1.25f);
    auto * x = ggml_view_4d(ctx, x_storage, width, 48, tokens, seqs,
                           x_storage->nb[1], x_storage->nb[2], x_storage->nb[3], 0);
    auto * gate = ggml_view_4d(ctx, g_storage, width, 48, tokens, seqs,
                              g_storage->nb[1], g_storage->nb[2], g_storage->nb[3], 0);
    ggml_build_forward_expand(graph, gate);
    auto * norm = ggml_rms_norm(ctx, x, 1e-6f);
    auto * weighted = ggml_mul(ctx, norm, weight);
    if (observe) ggml_set_output(weighted);
    auto * silu = ggml_silu(ctx, gate);
    auto * dst = reverse ? ggml_mul(ctx, silu, weighted) : ggml_mul(ctx, weighted, silu);
    ggml_set_name(dst, name.c_str());
    ggml_set_output(dst);
    ggml_build_forward_expand(graph, dst);
    ggml_backend_t backends[] = {backend, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr, "scheduler allocation failed");
    for (auto * t : {input, gate_input, weight}) ggml_backend_sched_set_tensor_backend(sched, t, backend);
    require(ggml_backend_sched_alloc_graph(sched, graph), "graph allocation failed");
    fill(input, 101, 2.0f);
    fill(gate_input, 202, 12.0f);
    fill(weight, 303, 1.5f);
    for (int i = 0; i < 5; ++i) {
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "warmup failed");
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "compute failed");
    }
    const auto stop = std::chrono::steady_clock::now();
    result out{std::vector<float>(ggml_nelements(dst)),
        std::chrono::duration<double, std::micro>(stop - start).count() / repeats,
        ggml_backend_sched_get_buffer_size(sched, backend)};
    ggml_backend_tensor_get(dst, out.data.data(), 0, ggml_nbytes(dst));
    for (float x : out.data) require(std::isfinite(x), "non-finite output");
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    try {
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 20;
        require(repeats > 0, "invalid repeat count");
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_name("CUDA0");
        if (!device) return 77;
        auto backend = ggml_backend_dev_init(device, nullptr);
        require(backend != nullptr, "CUDA backend initialization failed");
        auto cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        require(cpu != nullptr, "CPU backend initialization failed");
        for (const auto & name : {"t1", "t4", "t64", "t256", "t512", "seq2", "strided", "reverse",
                                 "width64", "gate_stride", "row_weights", "observed"}) {
            const std::string label(name);
            const int tokens = label == "t1" ? 1 : label == "t64" ? 64 : label == "t256" ? 256 : label == "t512" ? 512 : 4;
            const int seqs = label == "seq2" ? 2 : 1;
            const int width = label == "width64" ? 64 : 128;
            const bool stride = label == "strided";
            const bool gate_stride = label == "gate_stride";
            const bool row_weights = label == "row_weights";
            const bool reverse = label == "reverse";
            const bool observed = label == "observed";
            const auto reference = run(backend, cpu, "reference_" + label, tokens, seqs, width, stride, gate_stride, row_weights, reverse, true, repeats);
            const auto actual = run(backend, cpu, "candidate_" + label, tokens, seqs, width, stride, gate_stride, row_weights, reverse, observed, repeats);
            require(actual.data.size() == reference.data.size(), "output shape mismatch");
            if (std::memcmp(actual.data.data(), reference.data.data(), actual.data.size() * sizeof(float)) != 0) {
                size_t differences = 0;
                float max_error = 0;
                for (size_t i = 0; i < actual.data.size(); ++i) {
                    max_error = std::max(max_error, std::fabs(actual.data[i] - reference.data[i]));
                    if (actual.data[i] != reference.data[i] && differences++ < 4) {
                        std::fprintf(stderr, "DIFF %s[%zu]: %.9g / %.9g\n", name, i, reference.data[i], actual.data[i]);
                    }
                }
                std::fprintf(stderr, "DIFF count=%zu max_abs=%.9g\n", differences, max_error);
            }
            require(std::memcmp(actual.data.data(), reference.data.data(), actual.data.size() * sizeof(float)) == 0,
                    "fused output differs from two-kernel reference");
            std::printf("PASS %s: bitwise_equal floats=%zu reference_us=%.3f candidate_us=%.3f buffer=%zu/%zu\n",
                        name, actual.data.size(), reference.us, actual.us, reference.buffer_bytes, actual.buffer_bytes);
        }
        ggml_backend_free(backend);
        ggml_backend_free(cpu);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
