#include "kvmem/raw_kv_store.hpp"
#include "kvmem/snapshot.hpp"
#include "kvmem-session-files.h"
#include <cstring>
#include <iostream>

static void check(bool ok) { if (!ok) throw std::runtime_error("snapshot test failed"); }
template<class F> static void rejects(F f) {
    try { f(); } catch (const std::exception &) { return; }
    throw std::runtime_error("expected snapshot failure");
}

static void large_file() {
    // Exercise the >4 GiB boundary with 1 MiB of working memory, rather than
    // relying on a model's context length to produce a sufficiently big file.
    const uint64_t size = (uint64_t(4) << 30) + 137;
    const auto root = std::filesystem::temp_directory_path() /
        ("kvmem-large-snapshot-test-" + std::to_string(std::random_device{}()));
    std::vector<uint8_t> chunk(kvmem::snapshot_chunk, 0xa7);
    {
        kvmem_session_files files(root, size+8);
        files.save(1, size, [&](kvmem::SnapshotWriter & out) {
            for (uint64_t left = size; left;) {
                const auto n = std::min<uint64_t>(left, chunk.size());
                out.write(chunk.data(), n); left -= n;
            }
        });
        check(files.bytes() == size+8 && std::filesystem::file_size(files.path(1)) == size+8);
        files.load(1, [&](kvmem::SnapshotReader & in) {
            while (in.remaining()) {
                const auto n = std::min<uint64_t>(in.remaining(), chunk.size());
                in.read(chunk.data(), n);
                check(chunk[0] == 0xa7 && chunk[size_t(n)-1] == 0xa7);
            }
        });
    }
    std::filesystem::remove(root / kvmem_session_cache_dir::root_lock_name());
    std::filesystem::remove(root);
    std::cout << ">4 GiB streaming file roundtrip and quota passed\n";
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--large-file") { large_file(); return 0; }
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 3; cfg.n_embd_k = 64; cfg.n_embd_v = 64; cfg.block_tokens = 64;
    cfg.k_gpu_row_bytes = 68; cfg.v_gpu_row_bytes = 128;
    kvmem::RawKvStore source(cfg), restored(cfg);
    constexpr uint32_t tokens = 9021;
    std::vector<uint8_t> k(tokens*68), v(tokens*128);
    std::vector<float> means(tokens*64);
    for (size_t i = 0; i < k.size(); ++i) k[i] = uint8_t(i*19+7);
    for (size_t i = 0; i < v.size(); ++i) v[i] = uint8_t(i*13+3);
    for (size_t i = 0; i < means.size(); ++i) means[i] = float(i % 37)-18.5f;
    for (uint32_t layer : {0u, 2u}) {
        source.write_layer_k_gpu(0, tokens, layer, k.data());
        source.write_layer_v_gpu(0, tokens, layer, v.data());
        source.write_layer_mean_k(0, tokens, layer, means.data());
    }
    const auto write = [&](kvmem::SnapshotWriter & out) { source.snapshot_write(out); };
    kvmem::SnapshotWriter size; write(size);
    std::vector<uint8_t> archive;
    size_t max_chunk = 0;
    kvmem::SnapshotWriter out([&](const void * p, size_t n) {
        max_chunk = std::max(max_chunk, n);
        const auto * begin = (const uint8_t *)p; archive.insert(archive.end(), begin, begin+n);
    });
    write(out);
    check(size.bytes() == archive.size() && out.bytes() == size.bytes());
    check(size.bytes() > kvmem::snapshot_chunk && max_chunk <= kvmem::snapshot_chunk);
    auto read = [&](const std::vector<uint8_t> & data, kvmem::RawKvStore & raw) {
        size_t offset = 0;
        kvmem::SnapshotReader in([&](void * p, size_t n) { std::memcpy(p, data.data()+offset, n); offset += n; }, data.size());
        raw.snapshot_read(in, (tokens+63)/64);
        check(in.remaining() == 0); return in.hash();
    };
    check(read(archive, restored) == out.hash());
    for (uint32_t block = 0; block < (tokens+63)/64; ++block) {
        const uint32_t n = std::min(64u, tokens-block*64);
        check(!restored.has_k_gpu(block, 1));
        for (uint32_t layer : {0u, 2u}) {
            std::vector<uint8_t> got_k(n*68), got_v(n*128);
            check(restored.copy_k_gpu(block, layer, got_k.data(), n));
            check(restored.copy_v_gpu(block, layer, got_v.data(), n));
            check(!std::memcmp(got_k.data(), k.data()+block*64*68, got_k.size()));
            check(!std::memcmp(got_v.data(), v.data()+block*64*128, got_v.size()));
            float a[64], b[64]; source.mean_k(block, layer, a); restored.mean_k(block, layer, b);
            check(!std::memcmp(a, b, sizeof(a)));
        }
    }
    check(source.allocated_bytes() <= source.capacity_bytes(tokens, 2));
    const auto before = restored.allocated_bytes();
    auto truncated = archive; truncated.resize(truncated.size()-7);
    rejects([&] { read(truncated, restored); });
    check(restored.allocated_bytes() == before); // transactional raw load
    auto corrupt = archive; corrupt[0] ^= 1;
    rejects([&] { read(corrupt, restored); });
    auto bad_length = archive;
    // Header is 4 uint32 + block uint32 + 3 uint64, then the block count.
    const uint64_t insane = UINT64_MAX;
    std::memcpy(bad_length.data()+4*5+3*8, &insane, sizeof(insane));
    rejects([&] { read(bad_length, restored); });

    const auto root = std::filesystem::temp_directory_path() /
        ("kvmem-snapshot-test-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directory(root);
    const auto sentinel = root / "user-file"; std::ofstream(sentinel) << "keep";
    std::filesystem::path run;
    {
        const uint64_t file_bytes = size.bytes()+8;
        kvmem_session_files files(root, file_bytes); run = files.directory();
        kvmem_session_files other(root, file_bytes);
        check(run != other.directory());
        files.save(1, size.bytes(), [&](kvmem::SnapshotWriter & writer) {
            check(files.bytes() == file_bytes); write(writer);
        });
        check(files.bytes() == file_bytes && std::filesystem::file_size(files.path(1)) == file_bytes);
        rejects([&] { files.save(2, size.bytes(), write); });
        check(!files.contains(2));
        files.load(1, [&](kvmem::SnapshotReader & in) { restored.snapshot_read(in, (tokens+63)/64); });
        {
            std::fstream file(files.path(1), std::ios::binary|std::ios::in|std::ios::out);
            file.seekg(-1, std::ios::end); char byte; file.read(&byte, 1); byte ^= 1;
            file.seekp(-1, std::ios::end); file.write(&byte, 1);
        }
        rejects([&] { files.load(1, [&](kvmem::SnapshotReader & in) { restored.snapshot_read(in, (tokens+63)/64); }); });
        check(files.erase(1) && files.bytes() == 0);
        rejects([&] { files.save(2, size.bytes(), [&](kvmem::SnapshotWriter & writer) {
            writer.scalar(uint32_t(1)); throw std::runtime_error("injected short write");
        }); });
        check(files.bytes() == 0 && !files.contains(2));
        files.save(3, size.bytes(), write);
        std::filesystem::resize_file(files.path(3), 23);
        rejects([&] { files.load(3, [&](kvmem::SnapshotReader & in) { restored.snapshot_read(in, (tokens+63)/64); }); });
        // Destruction removes only this run's tracked files.
        check(std::filesystem::exists(sentinel));
    }
    check(!std::filesystem::exists(run) && std::filesystem::exists(sentinel));
    std::filesystem::remove(sentinel);
    std::filesystem::remove(root / kvmem_session_cache_dir::root_lock_name());
    std::filesystem::remove(root);
    std::cout << "snapshot roundtrip, bounds, checksum, quota, short write and cleanup passed\n";
}
