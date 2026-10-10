#pragma once

#include "kvmem/snapshot_buffer.hpp"
#include "kvmem-session-files.h"
#include <memory>

struct kvmem_session_chunk {
    std::vector<kvmem::SnapshotBuffer> buffers;
    bool in_ram = true;
    uint64_t payload_bytes() const {
        uint64_t n = 32; // magic, session id, generation, chunk id, buffer count
        for (const auto & b : buffers) n += 8 + b.size;
        return n;
    }
    uint64_t disk_bytes() const { return payload_bytes() + 8; }
    uint64_t restore_bytes() const {
        uint64_t n = 0; for (const auto & b : buffers) n += b.size; return n;
    }
    uint64_t ram_bytes() const {
        if (!in_ram) return 0;
        uint64_t n = 0; for (const auto & b : buffers) n += b.ram_bytes(); return n;
    }
};

// The authoritative, process-local manifest. Metadata/ownership never moves
// to an untrusted file. Both sessions can have mixed residency after a failed
// transfer; only an entirely restored manifest may be thawed for inference.
struct kvmem_session_payload {
    int id;
    uint64_t generation;
    bool invalid = false;
    std::vector<kvmem_session_chunk> chunks;

    kvmem_session_payload(int id, uint64_t generation, const std::vector<kvmem::SnapshotBuffer> & buffers,
                          uint64_t group_bytes = 16 * 1024 * 1024) : id(id), generation(generation) {
        uint64_t group = 0;
        for (const auto & b : buffers) {
            if (chunks.empty() || (group && (b.size >= group_bytes || group > group_bytes - b.size))) {
                chunks.emplace_back(); group = 0;
            }
            chunks.back().buffers.push_back(b); group += b.size + 8;
        }
    }
    uint64_t ram_bytes() const {
        uint64_t n = 0; for (const auto & c : chunks) n += c.ram_bytes(); return n;
    }
    uint64_t restore_bytes() const {
        uint64_t n = 0; for (const auto & c : chunks) if (!c.in_ram) n += c.restore_bytes(); return n;
    }
    uint64_t disk_bytes() const {
        uint64_t n = 0; for (const auto & c : chunks) n += c.disk_bytes(); return n;
    }
    uint64_t metadata_bytes() const {
        uint64_t n = sizeof(*this) + chunks.capacity() * sizeof(kvmem_session_chunk);
        for (const auto & c : chunks) n += c.buffers.capacity() * sizeof(kvmem::SnapshotBuffer);
        return n;
    }
    bool complete() const {
        if (invalid) return false;
        for (const auto & c : chunks) if (!c.in_ram) return false;
        return true;
    }
    void write_header(kvmem::SnapshotWriter & out, uint32_t index) const {
        out.scalar(uint64_t(0x324b4e554843564bull)); // KVCHUNK2
        out.scalar(int32_t(id)); out.scalar(generation); out.scalar(index);
        out.scalar(uint64_t(chunks.at(index).buffers.size()));
    }
    void check_header(kvmem::SnapshotReader & in, uint32_t index) const {
        try {
            in.expect(uint64_t(0x324b4e554843564bull)); in.expect(int32_t(id));
            in.expect(generation); in.expect(index);
            in.expect(uint64_t(chunks.at(index).buffers.size()));
        } catch (const std::runtime_error & e) { throw kvmem_session_corrupt(e.what()); }
    }
    void spill(kvmem_session_files & files, uint32_t index) {
        auto & c = chunks.at(index);
        if (invalid) throw std::runtime_error("invalid session payload");
        if (!c.in_ram) return;
        if (!files.ready(id, index)) {
            if (!files.erase_chunk(id, index)) throw std::runtime_error("cannot remove unfinished session chunk");
            files.save_chunk(id, index, c.payload_bytes(), [&](kvmem::SnapshotWriter & out) {
                write_header(out, index);
                for (const auto & b : c.buffers) b.write(b.object, out);
            });
        }
        // The complete, checksummed destination was published before release.
        for (const auto & b : c.buffers) b.release();
        c.in_ram = false;
    }
    void restore(kvmem_session_files & files, uint32_t index) {
        auto & c = chunks.at(index);
        if (invalid) throw kvmem_session_corrupt("invalid session payload");
        if (!c.in_ram) {
            size_t loaded = 0;
            try {
                files.load_chunk(id, index, [&](kvmem::SnapshotReader & in) {
                    check_header(in, index);
                    for (const auto & b : c.buffers) {
                        b.restore(in); ++loaded;
                    }
                });
            } catch (...) {
                // Never retain bytes from a file whose final hash failed.
                for (size_t i = 0; i < loaded; ++i) c.buffers[i].release();
                throw;
            }
            c.in_ram = true;
        }
        // If deletion fails, keep BOTH copies and the charge. A retry only
        // deletes this duplicate; it never overwrites a validated RAM copy.
        if (!files.erase_chunk(id, index)) throw std::runtime_error("cannot remove restored session chunk");
    }
};

struct kvmem_session_move {
    kvmem_session_payload * payload;
    uint32_t chunk;
    bool restore;
    uint64_t ram, disk;
};

struct kvmem_session_plan {
    enum class path { disk_first, ram_first, exchange } route = path::disk_first;
    std::vector<kvmem_session_move> moves;
    uint64_t peak_extra_ram = 0, peak_disk = 0;
};

// Pure planning: no allocation of KV, file writes, deletion or eviction.
// ram_free is actual machine headroom after a safety margin, NOT the soft cap.
// disk_used/disk_limit may include *planned* LRU removals; execute them only
// after this function has proved that the entire transfer can be scheduled.
inline kvmem_session_plan kvmem_plan_session_transfer(kvmem_session_files & files,
        kvmem_session_payload * target, const std::vector<kvmem_session_payload *> & outgoing,
        uint64_t ram_free, uint64_t disk_used, uint64_t disk_limit) {
    std::vector<kvmem_session_move> reads, writes;
    uint64_t read_ram = 0, write_disk = 0;
    if (target) {
        if (target->invalid) throw kvmem_session_corrupt("invalid target session");
        for (uint32_t i = 0; i < target->chunks.size(); ++i) {
            const auto & c = target->chunks[i];
            if (!c.in_ram && !files.ready(target->id, i)) throw kvmem_session_corrupt("missing session chunk");
            if (!c.in_ram || files.contains_chunk(target->id, i)) {
                const uint64_t ram = c.in_ram ? 0 : c.restore_bytes();
                reads.push_back({target, i, true, ram, files.chunk_bytes(target->id, i)}); read_ram += ram;
            }
        }
    }
    for (auto * p : outgoing) {
        if (p->invalid) throw std::runtime_error("invalid outgoing session");
        for (uint32_t i = 0; i < p->chunks.size(); ++i) if (p->chunks[i].in_ram) {
            const auto & c = p->chunks[i];
            if (files.contains_chunk(p->id, i) && !files.ready(p->id, i))
                throw std::runtime_error("unfinished session chunk still occupies disk");
            const uint64_t disk = files.ready(p->id, i) ? 0 : c.disk_bytes();
            writes.push_back({p, i, false, c.ram_bytes(), disk}); write_disk += disk;
        }
    }
    if (disk_used > disk_limit) throw std::runtime_error("session disk space changed before transfer");
    kvmem_session_plan result;
    result.route = write_disk <= disk_limit - disk_used ? kvmem_session_plan::path::disk_first :
        read_ram <= ram_free ? kvmem_session_plan::path::ram_first : kvmem_session_plan::path::exchange;
    result.peak_disk = disk_used;
    const uint64_t initial_free = ram_free;
    // Mark completed moves, rather than erasing the front of a long vector.
    std::vector<bool> read_done(reads.size()), write_done(writes.size());
    while (result.moves.size() < reads.size() + writes.size()) {
        bool progress = false;
        auto schedule = [&](const std::vector<kvmem_session_move> & pending, std::vector<bool> & done) {
            for (size_t i = 0; i < pending.size(); ++i) {
                if (done[i]) continue;
                const auto & m = pending[i];
                if (m.restore ? m.ram > ram_free : m.disk > disk_limit - disk_used) continue;
                result.moves.push_back(m); done[i] = true; progress = true;
                if (m.restore) { ram_free -= m.ram; disk_used -= m.disk; }
                else { ram_free += m.ram; disk_used += m.disk; }
                result.peak_disk = std::max(result.peak_disk, disk_used);
                result.peak_extra_ram = std::max(result.peak_extra_ram, initial_free > ram_free ? initial_free - ram_free : 0);
            }
        };
        if (result.route == kvmem_session_plan::path::ram_first) {
            schedule(reads, read_done); schedule(writes, write_done);
        } else {
            schedule(writes, write_done); schedule(reads, read_done);
        }
        if (!progress) throw std::runtime_error("insufficient temporary RAM/disk space for session exchange; caches retained");
    }
    return result;
}

inline void kvmem_execute_session_transfer(kvmem_session_files & files, const kvmem_session_plan & plan,
                                          const std::function<uint64_t()> & available_ram = {}) {
    for (const auto & move : plan.moves) {
        if (move.restore) {
            if (available_ram && move.ram > available_ram())
                throw std::runtime_error("available RAM decreased during session transfer; caches retained");
            move.payload->restore(files, move.chunk);
        } else move.payload->spill(files, move.chunk);
    }
}
