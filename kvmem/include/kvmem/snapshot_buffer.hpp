#pragma once

#include "snapshot.hpp"

namespace kvmem {
// A frozen allocation, not a copy of its contents. Bindings remain valid only
// while their owning detached store is frozen (no inference/vector relocation).
// Reclaim whole allocations: writing slices of a vector cannot release its RAM.
struct SnapshotBuffer {
    void * object = nullptr;
    uint64_t count = 0, size = 0;
    uint64_t (*capacity)(const void *) = nullptr;
    void (*write)(const void *, SnapshotWriter &) = nullptr;
    void (*read)(void *, uint64_t, SnapshotReader &) = nullptr;
    void (*clear)(void *) = nullptr;
    void * accounting = nullptr;
    void (*account)(void *, int64_t) = nullptr;

    uint64_t ram_bytes() const { return capacity(object); }
    void release() const {
        if (account) account(accounting, -int64_t(size));
        clear(object);
    }
    void restore(SnapshotReader & in) const {
        read(object, count, in);
        if (account) account(accounting, int64_t(size));
    }
    template<class T> static SnapshotBuffer bind(std::vector<T> & values) {
        static_assert(std::is_arithmetic<T>::value, "plain vector required");
        SnapshotBuffer b;
        b.object = &values; b.count = values.size(); b.size = uint64_t(values.size()) * sizeof(T);
        b.capacity = [](const void * p) { return uint64_t(static_cast<const std::vector<T> *>(p)->capacity()) * sizeof(T); };
        b.write = [](const void * p, SnapshotWriter & out) { out.vector(*static_cast<const std::vector<T> *>(p)); };
        b.read = [](void * p, uint64_t count, SnapshotReader & in) {
            // Exact length comes from our in-memory manifest, never the file.
            in.expect(count);
            auto & v = *static_cast<std::vector<T> *>(p);
            std::vector<T> next(static_cast<size_t>(count));
            in.read(next.data(), count * sizeof(T));
            v.swap(next);
        };
        b.clear = [](void * p) { std::vector<T>().swap(*static_cast<std::vector<T> *>(p)); };
        return b;
    }
};
} // namespace kvmem
