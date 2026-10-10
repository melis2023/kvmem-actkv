#pragma once

// Bounded streaming archive for process-local session snapshots. The caller
// supplies the transport (llama_file in the server, memory in host tests).
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace kvmem {
class SnapshotCorrupt : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
constexpr size_t snapshot_chunk = 1024 * 1024;
class SnapshotWriter {
public:
    using Sink = std::function<void(const void *, size_t)>;
    explicit SnapshotWriter(Sink sink = {}) : sink_(std::move(sink)) {}
    void write(const void * p, uint64_t n) {
        if (n > UINT64_MAX - bytes_) throw std::runtime_error("snapshot size overflow");
        if (!sink_) { bytes_ += n; return; } // sizing pass, no data copies
        auto src = static_cast<const uint8_t *>(p);
        while (n) {
            size_t part = size_t(std::min<uint64_t>(n, snapshot_chunk));
            sink_(src, part);
            for (size_t i = 0; i < part; ++i) hash_ = (hash_ ^ src[i]) * 1099511628211ull;
            bytes_ += part; src += part; n -= part;
        }
    }
    template<class T> void scalar(const T & value) {
        static_assert(std::is_arithmetic<T>::value, "scalar only");
        write(&value, sizeof(value));
    }
    template<class T> void vector(const std::vector<T> & values) {
        static_assert(std::is_arithmetic<T>::value, "scalar vector only");
        scalar(uint64_t(values.size()));
        write(values.data(), uint64_t(values.size()) * sizeof(T));
    }
    uint64_t bytes() const { return bytes_; }
    uint64_t hash() const { return hash_; }
private:
    Sink sink_;
    uint64_t bytes_ = 0, hash_ = 14695981039346656037ull;
};

class SnapshotReader {
public:
    using Source = std::function<void(void *, size_t)>;
    SnapshotReader(Source source, uint64_t bytes) : source_(std::move(source)), left_(bytes) {}
    void read(void * p, uint64_t n) {
        if (n > left_) throw SnapshotCorrupt("truncated session snapshot");
        auto dst = static_cast<uint8_t *>(p);
        while (n) {
            size_t part = size_t(std::min<uint64_t>(n, snapshot_chunk));
            source_(dst, part);
            for (size_t i = 0; i < part; ++i) hash_ = (hash_ ^ dst[i]) * 1099511628211ull;
            left_ -= part; dst += part; n -= part;
        }
    }
    template<class T> T scalar() {
        static_assert(std::is_arithmetic<T>::value, "scalar only");
        T value; read(&value, sizeof(value)); return value;
    }
    template<class T> void expect(T expected) {
        if (scalar<T>() != expected) throw SnapshotCorrupt("incompatible session snapshot");
    }
    template<class T> std::vector<T> vector(uint64_t max_count) {
        static_assert(std::is_arithmetic<T>::value, "scalar vector only");
        const auto n = scalar<uint64_t>();
        if (n > max_count || n > left_ / sizeof(T) || n > SIZE_MAX / sizeof(T))
            throw SnapshotCorrupt("invalid session snapshot vector length");
        std::vector<T> result(static_cast<size_t>(n));
        read(result.data(), n * sizeof(T)); return result;
    }
    uint64_t remaining() const { return left_; }
    uint64_t hash() const { return hash_; }
private:
    Source source_;
    uint64_t left_, hash_ = 14695981039346656037ull;
};
} // namespace kvmem
