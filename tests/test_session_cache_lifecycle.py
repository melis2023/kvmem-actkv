"""Process-level cache cleanup regression, usable on Windows and POSIX."""
import ctypes
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest

WORKER = str(Path(sys.argv.pop(1)).resolve())
OWNER = ".kvmem-owner"
ROOT_LOCK = ".kvmem-cache.lock"


def reparse_point(path):
    """True when path carries a reparse tag (symlink, junction, mount point).

    Python's is_symlink() is not enough on Windows: some configurations let
    symlink creation succeed while producing a plain directory or file with no
    reparse tag at all. Nothing is raised and is_symlink() reports False, so a
    test that only catches winerror 1314 silently tests nothing.
    """
    if os.name != "nt":
        return os.path.islink(str(path))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetFileAttributesW.argtypes = [ctypes.c_wchar_p]
    kernel.GetFileAttributesW.restype = ctypes.c_uint32
    attributes = kernel.GetFileAttributesW(str(path))
    return attributes != 0xFFFFFFFF and bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


def line(process):
    result = queue.Queue()
    threading.Thread(target=lambda: result.put(process.stdout.readline()), daemon=True).start()
    try:
        value = result.get(timeout=15)
    except queue.Empty:
        raise AssertionError("worker did not respond within 15 seconds") from None
    if not value:
        raise AssertionError("worker exited before replying: " + process.stderr.read())
    return value.rstrip("\n")


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kvmem-cache-lifecycle-")
        self.root = Path(self.temp.name) / "cache"
        self.root.mkdir()
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=15)
        self.temp.cleanup()

    def spawn(self, mode="hold", start=True):
        process = subprocess.Popen([WORKER, str(self.root), mode], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8")
        self.processes.append(process)
        if start:
            self.send(process, "start")
        return process

    @staticmethod
    def send(process, command):
        process.stdin.write(command + "\n")
        process.stdin.flush()

    def ready(self, process):
        fields = line(process).split("\t")
        self.assertEqual(fields[0], "READY")
        return Path(fields[1]), dict(zip(("runs", "bytes", "active", "skipped", "errors"), map(int, fields[2:])))

    def stop(self, process, kill=False):
        if kill:
            process.kill()
        else:
            self.send(process, "quit")
        _, errors = process.communicate(timeout=15)
        if not kill:
            self.assertEqual(process.returncode, 0, errors)
        return errors

    def orphan(self, mode="hold"):
        process = self.spawn(mode)
        run, _ = self.ready(process)
        self.stop(process, kill=True)
        self.assertTrue(run.exists())
        return run

    def test_graceful_exit(self):
        sentinel = self.root / "user-file"
        sentinel.write_text("keep")
        process = self.spawn()
        run, _ = self.ready(process)
        self.stop(process)
        self.assertFalse(run.exists())
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertTrue((self.root / ROOT_LOCK).is_file())

    def test_killed_process_and_partial_write(self):
        run = self.orphan("partial")
        self.assertTrue((run / "2-0.tmp").exists())
        size = sum(path.stat().st_size for path in run.iterdir() if path.name != OWNER)
        process = self.spawn()
        _, stats = self.ready(process)
        self.assertFalse(run.exists())
        self.assertEqual(stats["runs"], 1)
        self.assertEqual(stats["bytes"], size)
        self.stop(process)

    def test_live_process_is_preserved_and_loadable(self):
        live = self.spawn()
        live_run, _ = self.ready(live)
        orphan = self.orphan()
        cleaner = self.spawn()
        _, stats = self.ready(cleaner)
        self.assertTrue(live_run.exists())
        self.assertFalse(orphan.exists())
        self.assertEqual(stats["active"], 1)
        self.send(live, "check")
        self.assertEqual(line(live), "OK")
        self.stop(cleaner)
        self.stop(live)

    def test_simultaneous_startups(self):
        orphan = self.orphan()
        workers = [self.spawn(start=False) for _ in range(12)]
        for process in workers:
            self.send(process, "start")
        records = [self.ready(process) for process in workers]
        self.assertEqual(len({run for run, _ in records}), len(workers))
        self.assertFalse(orphan.exists())
        self.assertEqual(sum(stats["runs"] for _, stats in records), 1)
        for process in workers:
            self.send(process, "check")
        for process in workers:
            self.assertEqual(line(process), "OK")
        # Concurrent graceful exits also serialize against new startups.
        for process in workers:
            self.send(process, "quit")
        newcomer = self.spawn()
        self.ready(newcomer)
        for process in workers:
            _, errors = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, errors)
        self.stop(newcomer)

    def test_legacy_unknown_and_foreign_contents_are_preserved(self):
        legacy = self.root / "run-legacy"
        legacy.mkdir()
        (legacy / "1-0.kv").write_text("old snapshot")
        unknown = self.root / "run-unknown"
        unknown.mkdir()
        (unknown / OWNER).write_text("not our format")
        (unknown / "1-0.kv").write_text("foreign snapshot")
        orphan = self.orphan()
        (orphan / "user-file").write_text("keep")
        process = self.spawn()
        _, stats = self.ready(process)
        self.assertEqual(stats["skipped"], 3)
        self.assertTrue((orphan / "1-0.kv").exists())
        self.assertEqual((legacy / "1-0.kv").read_text(), "old snapshot")
        self.assertEqual((unknown / "1-0.kv").read_text(), "foreign snapshot")
        self.assertEqual((orphan / "user-file").read_text(), "keep")
        self.stop(process)

    def test_symlinks_are_not_followed(self):
        external = Path(self.temp.name) / "external"
        external.mkdir()
        target = external / "1-0.kv"
        target.write_text("keep outside cache")
        orphan = self.orphan()
        links = [(self.root / "run-link", True), (orphan / "2-0.kv", False)]
        for link, is_directory in links:
            try:
                link.symlink_to(external if is_directory else target, target_is_directory=is_directory)
            except OSError as error:
                if os.name != "nt" or getattr(error, "winerror", None) != 1314:
                    raise
        if not all(reparse_point(link) for link, _ in links):
            # Directory junctions do not require Windows' symlink privilege and
            # are reparse points too, which is what the cleanup code checks for.
            for link, _ in links:
                if link.is_symlink() or link.is_file():
                    link.unlink()
                elif link.is_dir():
                    link.rmdir()
                result = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(external)],
                                        capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(reparse_point(self.root / "run-link"), "run-link is not a reparse point")
        self.assertTrue(reparse_point(orphan / "2-0.kv"), "2-0.kv is not a reparse point")
        process = self.spawn()
        _, stats = self.ready(process)
        self.assertEqual(stats["skipped"], 2)
        self.assertTrue((orphan / "1-0.kv").exists())
        self.assertEqual(target.read_text(), "keep outside cache")
        self.stop(process)

    def test_delete_failure_is_logged_and_retried(self):
        orphan = self.orphan()
        target = orphan / "1-0.kv"
        handle = None
        if os.name == "nt":
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
            kernel.CreateFileW.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.CreateFileW(str(target), 0x80000000, 3, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
        elif os.geteuid() == 0:
            self.skipTest("permission denial requires an unprivileged POSIX user")
        else:
            orphan.chmod(0o500)
        try:
            process = self.spawn()
            _, stats = self.ready(process)
            self.assertGreater(stats["errors"], 0)
            self.assertTrue((orphan / OWNER).exists())
            self.assertTrue(target.exists())
            errors = self.stop(process)
            self.assertIn(str(target), errors)
            self.assertIn("cannot remove", errors)
        finally:
            if handle is not None:
                kernel.CloseHandle(handle)
            else:
                orphan.chmod(0o700)
        retry = self.spawn()
        _, stats = self.ready(retry)
        self.assertEqual(stats["runs"], 1)
        self.assertFalse(orphan.exists())
        self.stop(retry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
