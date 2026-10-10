#pragma once
#include <cstdint>

namespace kvmem {
// Physical/commit headroom for transient transfers, not a process reservation.
// Returns zero when the OS cannot supply a reliable estimate.
uint64_t session_memory_available();
}
