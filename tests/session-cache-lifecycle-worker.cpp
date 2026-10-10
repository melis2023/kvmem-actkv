// Small real process used by the lifecycle test: killing it bypasses all C++
// destructors, just as killing the server does. No model or GPU is needed.
#include "kvmem-session-files.h"
#include <iostream>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: worker ROOT hold|partial");
        std::string command;
        if (!std::getline(std::cin, command) || command != "start") return 2;
        kvmem_session_files files(std::filesystem::u8path(argv[1]), 4 * 1024 * 1024);
        const auto ready = [&] {
            const auto & c = files.startup_cleanup();
            std::cout << "READY\t" << files.directory().u8string() << '\t' << c.removed_runs << '\t'
                      << c.removed_bytes << '\t' << c.active_runs << '\t' << c.skipped_runs << '\t'
                      << c.errors << std::endl;
        };
        const std::vector<uint8_t> payload(4096, 0xa7);
        files.save(1, payload.size(), [&](kvmem::SnapshotWriter & out) { out.write(payload.data(), payload.size()); });
        if (std::string(argv[2]) == "partial") {
            const std::vector<uint8_t> chunk(1024 * 1024, 0x5b);
            files.save(2, 2 * chunk.size(), [&](kvmem::SnapshotWriter & out) {
                out.write(chunk.data(), chunk.size());
                ready();
                std::getline(std::cin, command); // parent kills us during this write
                throw std::runtime_error("partial worker must be killed");
            });
        } else {
            ready();
            while (std::getline(std::cin, command) && command != "quit") {
                if (command != "check") throw std::runtime_error("unknown command");
                std::vector<uint8_t> restored(payload.size());
                files.load(1, [&](kvmem::SnapshotReader & in) { in.read(restored.data(), restored.size()); });
                if (restored != payload) throw std::runtime_error("live snapshot changed");
                std::cout << "OK" << std::endl;
            }
        }
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}
