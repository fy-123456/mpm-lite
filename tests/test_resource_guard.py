import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.resource_guard import GIB, StoragePaused, StorageStatus, inspect_storage, prepare_warp_cache


class TestResourceGuard(unittest.TestCase):
    def test_normal_storage_keeps_requested_cache(self):
        with patch(
            "utils.resource_guard.inspect_storage",
            return_value=StorageStatus(10 * GIB, 20 * GIB, 30 * GIB, False, False),
        ):
            self.assertEqual(prepare_warp_cache("/tmp/phase1-cache", "/data"), "/tmp/phase1-cache")

    def test_low_system_storage_moves_cache_to_data_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "system-cache"
            data = root / "data"
            source.mkdir()
            (source / "kernel.bin").write_text("cache")
            with patch(
                "utils.resource_guard.inspect_storage",
                return_value=StorageStatus(1 * GIB, 20 * GIB, 21 * GIB, True, False),
            ):
                result = Path(prepare_warp_cache(source, data))
            self.assertEqual(result, data / "mpm-lite-warp-cache")
            self.assertEqual((result / "kernel.bin").read_text(), "cache")
            self.assertFalse(source.exists())

    def test_total_low_storage_pauses(self):
        with patch(
            "utils.resource_guard.inspect_storage",
            return_value=StorageStatus(2 * GIB, 2 * GIB, 4 * GIB, False, True),
        ):
            with self.assertRaises(StoragePaused):
                prepare_warp_cache("/tmp/phase1-cache", "/data")

    def test_inspect_storage_reports_current_root(self):
        status = inspect_storage()
        self.assertGreater(status.system_free_bytes, 0)
        self.assertEqual(status.data_free_bytes, None)
        self.assertEqual(status.paused, status.total_free_bytes < 5 * GIB)


if __name__ == "__main__":
    unittest.main()
