#include "kvmem/session_memory.hpp"
#include <algorithm>
#include <fstream>
#include <string>
#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <unistd.h>
#endif

namespace kvmem {
uint64_t session_memory_available() {
    uint64_t available = 0;
#ifdef _WIN32
    MEMORYSTATUSEX status{}; status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) available = std::min(status.ullAvailPhys, status.ullAvailPageFile);
#elif defined(__APPLE__)
    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page_size = 0;
    const auto host = mach_host_self();
    if (host_page_size(host, &page_size) == KERN_SUCCESS &&
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count) == KERN_SUCCESS)
        available = uint64_t(stats.free_count + stats.inactive_count) * page_size;
    mach_port_deallocate(mach_task_self(), host);
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string key, rest;
    while (meminfo >> key) {
        if (key == "MemAvailable:") { meminfo >> available; available *= 1024; break; }
        std::getline(meminfo, rest);
    }
#ifdef _SC_AVPHYS_PAGES
    if (!available) {
        const auto pages = sysconf(_SC_AVPHYS_PAGES), size = sysconf(_SC_PAGESIZE);
        if (pages > 0 && size > 0) available = uint64_t(pages) * uint64_t(size);
    }
#endif
    // Honor cgroup v2 limits, including parent limits and namespace roots.
    const std::filesystem::path root("/sys/fs/cgroup");
    auto limit_at = [&](const std::filesystem::path & path) {
        std::ifstream max_file(path / "memory.max"), used_file(path / "memory.current");
        uint64_t maximum = 0, used = 0;
        if (max_file >> maximum && used_file >> used)
            available = std::min(available, maximum > used ? maximum - used : 0);
    };
    limit_at(root);
    std::ifstream groups("/proc/self/cgroup");
    while (std::getline(groups, rest)) if (rest.rfind("0::/", 0) == 0) {
        const auto relative = std::filesystem::path(rest.substr(4)).lexically_normal();
        bool safe = !relative.is_absolute();
        for (const auto & part : relative) if (part == "..") safe = false;
        if (safe) for (auto path = root / relative; path != root && !path.empty(); path = path.parent_path()) limit_at(path);
    }
#endif
    // Leave working space for I/O, manifest bookkeeping and concurrent activity.
    const uint64_t margin = std::min<uint64_t>(256 * 1024 * 1024, available / 8);
    return available - margin;
}
}
