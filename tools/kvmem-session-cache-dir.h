#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// Locks are tied to OS handles, never to PIDs or file timestamps. The root lock
// file is permanent: unlinking it could let two processes lock different inodes.
class kvmem_session_cache_lock {
public:
    kvmem_session_cache_lock() = default;
    kvmem_session_cache_lock(const kvmem_session_cache_lock &) = delete;
    kvmem_session_cache_lock & operator=(const kvmem_session_cache_lock &) = delete;
    ~kvmem_session_cache_lock() { close(); }

    bool open(const std::filesystem::path & path, bool create, bool wait) {
        path_ = path;
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, create ? OPEN_ALWAYS : OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) fail(GetLastError());
        BY_HANDLE_FILE_INFORMATION info{};
        if (!GetFileInformationByHandle(handle_, &info)) fail(GetLastError());
        if ((info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) || info.nNumberOfLinks != 1)
            fail(ERROR_INVALID_DATA);
        OVERLAPPED offset{};
        if (!LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK | (wait ? 0 : LOCKFILE_FAIL_IMMEDIATELY), 0, 1, 0, &offset)) {
            const auto error = GetLastError();
            close();
            if (!wait && error == ERROR_LOCK_VIOLATION) return false;
            fail(error);
        }
#else
        fd_ = ::open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW | (create ? O_CREAT : 0), 0600);
        if (fd_ < 0) fail(errno);
        struct stat info{};
        if (fstat(fd_, &info)) fail(errno);
        if (!S_ISREG(info.st_mode) || info.st_nlink != 1) fail(EINVAL);
        while (flock(fd_, LOCK_EX | (wait ? 0 : LOCK_NB))) {
            const int error = errno;
            if (error == EINTR) continue;
            close();
            if (!wait && (error == EWOULDBLOCK || error == EAGAIN)) return false;
            fail(error);
        }
#endif
        return true;
    }

    void close() noexcept {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) { CloseHandle(handle_); handle_ = INVALID_HANDLE_VALUE; }
#else
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
    }

    void mark() {
        const std::string magic = marker();
#ifdef _WIN32
        DWORD written = 0;
        if (!WriteFile(handle_, magic.data(), DWORD(magic.size()), &written, nullptr)) fail(GetLastError());
        if (written != magic.size()) fail(ERROR_WRITE_FAULT);
        if (!FlushFileBuffers(handle_)) fail(GetLastError());
#else
        size_t offset = 0;
        while (offset < magic.size()) {
            const auto n = ::write(fd_, magic.data() + offset, magic.size() - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) fail(n < 0 ? errno : EIO);
            offset += size_t(n);
        }
        if (fsync(fd_)) fail(errno);
#endif
    }

    bool marked() {
        char buffer[64]{};
        size_t size = 0;
#ifdef _WIN32
        DWORD read = 0;
        if (!ReadFile(handle_, buffer, sizeof(buffer), &read, nullptr)) fail(GetLastError());
        size = read;
#else
        ssize_t n;
        do { n = pread(fd_, buffer, sizeof(buffer), 0); } while (n < 0 && errno == EINTR);
        if (n < 0) fail(errno);
        size = size_t(n);
#endif
        return std::string(buffer, size) == marker();
    }

private:
    static const char * marker() { return "KVMem session cache owner v1\n"; }
    [[noreturn]] void fail(int error) {
        close();
        throw std::filesystem::filesystem_error("session cache lock", path_,
            std::error_code(error,
#ifdef _WIN32
                std::system_category()
#else
                std::generic_category()
#endif
            ));
    }
    std::filesystem::path path_;
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

class kvmem_session_cache_dir {
public:
    using log_fn = std::function<void(bool, const std::string &)>;
    struct cleanup_result {
        uint64_t removed_runs = 0, removed_bytes = 0, active_runs = 0, skipped_runs = 0, errors = 0;
    };
    static const char * root_lock_name() { return ".kvmem-cache.lock"; }
    static const char * owner_name() { return ".kvmem-owner"; }

    explicit kvmem_session_cache_dir(const std::filesystem::path & root, log_fn log = {}) : log_(std::move(log)) {
        std::filesystem::create_directories(root);
        root_ = std::filesystem::canonical(root);
        kvmem_session_cache_lock root_lock;
        root_lock.open(root_ / root_lock_name(), true, true);
        // Scanning, registration and graceful cleanup all hold the root lock.
        // An incomplete registration therefore cannot be mistaken for an orphan.
        for (const auto & entry : std::filesystem::directory_iterator(root_)) {
            const auto path = entry.path();
            if (path.filename().u8string().rfind("run-", 0) != 0) continue;
            try {
                if (!plain(path, true) || !plain(path / owner_name(), false)) { ++result_.skipped_runs; continue; }
                kvmem_session_cache_lock owner;
                if (!owner.open(path / owner_name(), false, false)) { ++result_.active_runs; continue; }
                if (!owner.marked()) { ++result_.skipped_runs; continue; }
                cleanup(path, owner, result_);
            } catch (const std::exception & e) { ++result_.errors; report(true, e.what()); }
        }
        report(false, "startup cleanup root=" + root_.u8string() +
            " removed_runs=" + std::to_string(result_.removed_runs) +
            " removed_bytes=" + std::to_string(result_.removed_bytes) +
            " active_runs=" + std::to_string(result_.active_runs) +
            " skipped_runs=" + std::to_string(result_.skipped_runs) +
            " errors=" + std::to_string(result_.errors));
        for (int attempt = 0; attempt < 32; ++attempt) {
            dir_ = root_ / ("run-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
                            "-" + std::to_string(std::random_device{}()));
            if (!std::filesystem::create_directory(dir_)) continue;
            try {
                std::filesystem::permissions(dir_, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);
                owner_.open(dir_ / owner_name(), true, true);
                owner_.mark(); // publish before any snapshot can be written
                return;
            } catch (...) {
                owner_.close();
                std::error_code ec;
                std::filesystem::remove(dir_ / owner_name(), ec);
                std::filesystem::remove(dir_, ec);
                throw;
            }
        }
        throw std::runtime_error("cannot create private session cache directory");
    }

    ~kvmem_session_cache_dir() {
        try {
            kvmem_session_cache_lock root_lock;
            root_lock.open(root_ / root_lock_name(), true, true);
            cleanup_result ignored;
            cleanup(dir_, owner_, ignored);
        } catch (const std::exception & e) { report(true, e.what()); }
    }
    kvmem_session_cache_dir(const kvmem_session_cache_dir &) = delete;
    kvmem_session_cache_dir & operator=(const kvmem_session_cache_dir &) = delete;
    const std::filesystem::path & directory() const { return dir_; }
    const cleanup_result & startup_cleanup() const { return result_; }
    void report(bool warning, const std::string & message) const noexcept {
        try {
            if (log_) log_(warning, message);
            else std::fprintf(stderr, "KVMEM session cache %s%s\n", warning ? "warning: " : "", message.c_str());
        } catch (...) {} // diagnostics must not interrupt destruction
    }

private:
    static bool plain(const std::filesystem::path & path, bool directory) {
        std::error_code ec;
        const auto status = std::filesystem::symlink_status(path, ec);
        if (ec == std::errc::no_such_file_or_directory) return false;
        if (ec) throw std::filesystem::filesystem_error("inspect session cache", path, ec);
        if (directory ? !std::filesystem::is_directory(status) : !std::filesystem::is_regular_file(status)) return false;
#ifdef _WIN32
        // Junctions and other reparse points must also be excluded.
        const auto attributes = GetFileAttributesW(path.c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES)
            throw std::filesystem::filesystem_error("inspect session cache", path,
                std::error_code(GetLastError(), std::system_category()));
        if (attributes & FILE_ATTRIBUTE_REPARSE_POINT) return false;
#endif
        return true;
    }
    static bool chunk_name(const std::string & name) {
        const auto dash = name.find('-'), dot = name.find('.');
        if (dash == std::string::npos || dot == std::string::npos || dash == 0 || dot <= dash + 1) return false;
        if (name.substr(dot) != ".kv" && name.substr(dot) != ".tmp") return false;
        for (size_t i = 0; i < dot; ++i) if (i != dash && (name[i] < '0' || name[i] > '9')) return false;
        return true;
    }
    void cleanup(const std::filesystem::path & dir, kvmem_session_cache_lock & owner, cleanup_result & result) {
        std::vector<std::pair<std::filesystem::path, uint64_t>> files;
        for (const auto & entry : std::filesystem::directory_iterator(dir)) {
            const auto path = entry.path();
            if (path.filename() == owner_name()) continue;
            // Never recursively remove a run directory or follow links. Unknown
            // entries make the entire directory ineligible for automatic cleanup.
            if (!chunk_name(path.filename().u8string()) || !plain(path, false)) {
                ++result.skipped_runs;
                report(true, "cleanup skipped unexpected entry: " + path.u8string());
                return;
            }
            files.emplace_back(path, std::filesystem::file_size(path));
        }
        bool failed = false;
        const auto remove = [&](const std::filesystem::path & path) {
            std::error_code ec;
            const bool removed = std::filesystem::remove(path, ec);
            if (ec) { ++result.errors; failed = true; report(true, "cannot remove " + path.u8string() + ": " + ec.message()); }
            return removed;
        };
        for (const auto & file : files) if (remove(file.first)) result.removed_bytes += file.second;
        if (failed) return; // keep the marker so a later startup can retry
        owner.close(); // Windows cannot unlink the open owner; root lock excludes scanners.
        if (!remove(dir / owner_name()) && failed) return;
        if (remove(dir)) ++result.removed_runs;
    }

    std::filesystem::path root_, dir_;
    kvmem_session_cache_lock owner_;
    log_fn log_;
    cleanup_result result_;
};
