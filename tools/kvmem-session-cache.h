#pragma once

// Included after ServerState and conversation bookkeeping. Frozen allocations
// stay in place; failures leave resumable RAM/disk manifests, never attached
// partial stores. No whole-session serialization buffer is allocated.
static kvmem_session_payload & session_freeze(ServerState & st, int id) {
    auto & conv = st.conv.at(id);
    if (conv.payload) return *conv.payload;
    if (id == st.conv_active) throw std::runtime_error("cannot freeze active session");
    std::vector<kvmem::SnapshotBuffer> buffers;
    std::set<const MultimodalCheckpointData *> unique;
    auto checkpoint = [&](const std::shared_ptr<const MultimodalCheckpointData> & p) {
        if (!p || !unique.insert(p.get()).second) return;
        // Created mutable by multimodal_checkpoint(). Live/rollback aliases
        // were cleared after parking. Only this conversation owns references.
        auto & data = *const_cast<MultimodalCheckpointData *>(p.get());
        auto add = [&](auto & vector) {
            if (!vector.capacity()) return;
            auto b = kvmem::SnapshotBuffer::bind(vector);
            if (data.accounting) {
                b.accounting = data.accounting.get();
                b.account = [](void * p, int64_t delta) {
                    auto & a = *static_cast<MultimodalCheckpointAccounting *>(p);
                    if (delta < 0) a.live_bytes -= size_t(-delta); else a.live_bytes += size_t(delta);
                    a.peak_bytes = std::max(a.peak_bytes, a.live_bytes);
                };
            }
            buffers.push_back(b);
        };
        add(data.recurrent); add(data.draft_carry); add(data.tail_mean);
    };
    for (const auto & c : conv.mm_checkpoints) checkpoint(c.data);
    checkpoint(conv.mm_live_checkpoint);
    for (auto * v : {&conv.gdn_ckpt, &conv.gdn_carry, &conv.gdn_query_carry, &conv.gdn_ckpt_query})
        if (v->capacity()) buffers.push_back(kvmem::SnapshotBuffer::bind(*v));
    const int32_t store = st.conv_table.find(id)->store_id;
    llama_kvmem_store_freeze(store, buffers);
    try {
        conv.payload = std::make_unique<kvmem_session_payload>(id, ++st.session_generation, buffers);
    } catch (...) { llama_kvmem_store_thaw(store); throw; }
    conv.disk_gen = !conv.gdn_ckpt.empty(); conv.disk_query = !conv.gdn_ckpt_query.empty();
    conv.cold = true; // also covers partially migrated and frozen, still-hot stores
    return *conv.payload;
}

static void session_refresh_bytes(ServerState & st) {
    for (const auto & entry : st.conv_table.entries())
        st.conv_table.set_bytes(entry.id, conversation_bytes(st, entry.id));
}

static void session_warn_budget(ServerState & st) {
    const bool over = st.conv_limits.max_bytes && st.conv_table.bytes_total() > st.conv_limits.max_bytes;
    if (over && !st.conv_budget_warned)
        LOG_WRN("srv    KVMEM session RAM soft limit exceeded (bytes=%llu cap=%llu); retaining active KV and cold metadata\n",
            (unsigned long long)st.conv_table.bytes_total(), (unsigned long long)st.conv_limits.max_bytes);
    st.conv_budget_warned = over;
}

// Plan the final placement BEFORE choosing a transfer order or applying LRU.
// Retention uses measured RAM and exact restore sizes, just like RAM-only mode.
// Worst-case future generation estimates must not evict reusable sessions.
static void session_balance(ServerState & st, int target) {
    const auto started = std::chrono::steady_clock::now();
    auto & files = *st.session_files;
    auto * incoming = target >= 0 && target != st.conv_active ? st.conv.at(target).payload.get() : nullptr;
    session_refresh_bytes(st);
    const uint64_t restore = incoming ? incoming->restore_bytes() : 0;
    uint64_t projected = st.conv_table.bytes_total() + restore;
    const uint64_t ram_free = kvmem::session_memory_available();
    uint64_t freed_ram = 0;
    std::vector<kvmem_session_payload *> outgoing;
    for (int id : st.conv_table.lru_order()) {
        if (id == target || id == st.conv_active) continue;
        const bool soft_pressure = st.conv_limits.max_bytes && projected > st.conv_limits.max_bytes;
        if (!soft_pressure && restore <= ram_free + freed_ram) break;
        auto & p = session_freeze(st, id);
        if (p.invalid || !p.ram_bytes()) continue;
        projected -= std::min(projected, p.ram_bytes()); freed_ram += p.ram_bytes();
        outgoing.push_back(&p);
    }
    // Unfinished temp files still have their original RAM source. A failed
    // cleanup remains an I/O error, never an eviction request.
    for (auto * p : outgoing) for (uint32_t i = 0; i < p->chunks.size(); ++i)
        if (p->chunks[i].in_ram && files.contains_chunk(p->id, i) && !files.ready(p->id, i) && !files.erase_chunk(p->id, i))
            throw std::runtime_error("cannot remove unfinished session snapshot");

    const uint64_t disk_available = files.available();
    const uint64_t disk_limit = std::min(files.limit(), files.bytes() + std::min(disk_available, UINT64_MAX - files.bytes()));
    uint64_t final_disk = files.bytes() - (incoming ? files.session_bytes(target) : 0);
    for (auto * p : outgoing) final_disk += p->disk_bytes() - files.session_bytes(p->id);
    uint64_t disk_used = files.bytes(), evicted_ram = 0;
    std::vector<int> evictions;
    // Only FINAL capacity pressure permits LRU. Credit target files that will
    // be freed by restore, including the RAM 10 / target 15 / disk quota 20 case.
    for (int id : st.conv_table.lru_order()) {
        if (final_disk <= disk_limit) break;
        if (id == target || id == st.conv_active) continue;
        const auto it = std::find_if(outgoing.begin(), outgoing.end(), [&](auto * p) { return p->id == id; });
        const uint64_t saved = it != outgoing.end() ? (*it)->disk_bytes() : files.session_bytes(id);
        if (!saved) continue;
        final_disk -= saved; disk_used -= files.session_bytes(id);
        evicted_ram += conversation_bytes(st, id); evictions.push_back(id);
        if (it != outgoing.end()) outgoing.erase(it);
    }
    if (final_disk > disk_limit) throw std::runtime_error("cannot free final session disk capacity");
    const auto plan = kvmem_plan_session_transfer(files, incoming, outgoing,
        kvmem::session_memory_available() + evicted_ram, disk_used, disk_limit);
    const char * route = plan.route == kvmem_session_plan::path::ram_first ? "ram_first" :
        plan.route == kvmem_session_plan::path::exchange ? "exchange" : "disk_first";
    kvmem_diag("KVMEM_TRACE session_transfer target=%d path=%s moves=%zu peak_disk=%llu final_disk=%llu\n",
        target, route, plan.moves.size(), (unsigned long long)plan.peak_disk, (unsigned long long)final_disk);
    for (int id : evictions)
        if (!conversation_evict(st, id, "disk_lru")) throw std::runtime_error("cannot remove LRU session snapshot");
    try {
        kvmem_execute_session_transfer(files, plan, kvmem::session_memory_available);
    } catch (...) { session_refresh_bytes(st); throw; }
    session_refresh_bytes(st);
    for (auto * p : outgoing) {
        ++st.conv_counts.spills;
        kvmem_diag("KVMEM_TRACE session_spill id=%d disk_bytes=%llu ram_bytes=%llu ms=%.2f\n", p->id,
            (unsigned long long)files.session_bytes(p->id), (unsigned long long)st.conv_table.find(p->id)->bytes,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count());
    }
    if (incoming) {
        if (!incoming->complete() || files.contains(target)) throw std::runtime_error("incomplete session restore");
        llama_kvmem_store_thaw(st.conv_table.find(target)->store_id);
        auto & conv = st.conv.at(target); conv.payload.reset(); conv.cold = false;
        st.conv_table.set_bytes(target, conversation_bytes(st, target));
        ++st.conv_counts.restores;
        kvmem_diag("KVMEM_TRACE session_restore id=%d rows=%u ms=%.2f\n", target, conv.stored,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count());
    }
}

// Post-request enforcement must not invalidate the completed active cache.
static void session_make_room(ServerState & st, int target) {
    try { session_balance(st, target); }
    catch (const std::exception & e) {
        ++st.conv_counts.disk_errors; session_refresh_bytes(st);
        LOG_WRN("srv    KVMEM idle session migration deferred: %s\n", e.what());
    }
    session_warn_budget(st);
}

static void session_begin_request(ServerState & st, const kvmem_prompt & prompt,
                                  const std::string & client_id, int predict) {
    const uint64_t rows = uint64_t(prompt.tokens.size()) + uint64_t(std::max(0, predict)) +
        (st.spec.ok ? uint64_t(std::max(0, st.spec_n_max)) + 1 : 0);
    if (rows > UINT32_MAX) throw std::invalid_argument("session token count overflow");
    const uint64_t reserve = llama_kvmem_store_capacity(uint32_t(rows));
    std::vector<kvmem_store_match> matches;
    for (const auto & entry : st.conv_table.entries()) matches.push_back(conversation_match(st, entry.id, prompt));
    auto limits = st.conv_limits; limits.max_bytes = 0; limits.attached = st.conv_active;
    const auto selection = kvmem_store_select(matches, int(prompt.tokens.size())-(st.spec.ok ? 1 : 0), st.spec.ok, client_id, limits);
    int target = selection.id;
    if (target != st.conv_active && st.conv_active >= 0) {
        if (!st.mm_committed || st.conv_table.find(st.conv_active)->store_id != llama_kvmem_store_current())
            throw std::runtime_error("session switch requires a committed active conversation");
        const int outgoing = st.conv_active;
        conversation_swap(st, st.conv.at(outgoing));
        if (!llama_kvmem_store_park()) {
            conversation_swap(st, st.conv.at(outgoing));
            if (!llama_kvmem_store_n_tokens() && !st.cached_tokens.empty()) memory_clear_all(st);
            throw std::runtime_error("could not park active session");
        }
        st.conv_active = -1;
        st.mm_live_checkpoint.reset(); st.mm_rollback.reset(); st.mm_rollback_prompt.reset(); st.mm_pending_query.reset();
        if (!llama_kvmem_store_rows(st.conv_table.find(outgoing)->store_id)) conversation_drop_payload(st.conv.at(outgoing));
        st.conv_table.set_bytes(outgoing, conversation_bytes(st, outgoing));
    }
    for (int victim : st.conv_table.lru_order()) {
        if (st.conv_table.count() + (target < 0 ? 1 : 0) <= st.conv_limits.max_stores) break;
        if (victim != target && victim != st.conv_active) conversation_evict(st, victim, "lru");
    }
    if (st.conv_table.count() + (target < 0 ? 1 : 0) > st.conv_limits.max_stores)
        throw std::runtime_error("cannot free session count capacity");
    try {
        // A hot/fresh request does not depend on optional idle demotion. The
        // RAM cap stays soft even when that background retention work fails.
        if (target < 0 || !st.conv.at(target).payload) session_make_room(st, target);
        else session_balance(st, target);
    } catch (const kvmem_session_corrupt & e) {
        ++st.conv_counts.disk_errors;
        LOG_WRN("srv    KVMEM session id=%d corrupt: %s; cache miss\n", target, e.what());
        if (target < 0 || !conversation_evict(st, target, "invalid_snapshot"))
            throw std::runtime_error("cannot remove invalid session snapshot");
        target = -1;
        session_balance(st, target);
    } catch (...) {
        ++st.conv_counts.disk_errors; session_refresh_bytes(st); throw;
    }
    if (target < 0) {
        const int32_t store = llama_kvmem_store_create();
        if (store < 0) throw std::runtime_error("cannot allocate session store");
        target = st.conv_table.add(store); st.conv.emplace(target, kvmem_conversation{});
    }
    if (target != st.conv_active) {
        const int32_t store = st.conv_table.find(target)->store_id;
        const bool restored = llama_kvmem_store_switch(store);
        if (llama_kvmem_store_current() != store) throw std::runtime_error("cannot attach session store");
        conversation_swap(st, st.conv.at(target)); st.conv_active = target;
        st.mm_live_checkpoint.reset();
        if (!restored) memory_clear_all(st);
        ++st.conv_counts.switches;
    }
    st.conv_table.touch(target, ++st.conv_clock);
    if (selection.id == target) ++st.conv_counts.extends; else ++st.conv_counts.forks;
    conversation_publish(st);
    kvmem_diag("KVMEM_TRACE session_select id=%d keep=%d reserve=%llu\n", target,
        selection.id == target ? selection.keep : 0, (unsigned long long)reserve);
}
