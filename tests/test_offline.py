import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from offline.runtime import local_path, caltech_folder_frames, preflight
from offline.package import SplitWriter


class OfflineTests(unittest.TestCase):
    def test_windows_and_relative_dataset_paths(self):
        self.assertEqual(str(local_path(r"D:\research datasets\MRI", Path("/bundle"))),
                         "/mnt/d/research datasets/MRI")
        self.assertEqual(local_path("datasets/MRI", Path("/bundle")), Path("/bundle/datasets/MRI"))

    def test_caltech_partition_is_disjoint_reproducible_and_excludes_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("faces", "birds", "BACKGROUND_Google"):
                folder = root / name
                folder.mkdir()
                for index in range(40):
                    (folder / f"image_{index:04d}.jpg").touch()
            parts, names = caltech_folder_frames(root, 0.1, 42)
            again, _ = caltech_folder_frames(root, 0.1, 42)
            self.assertEqual(names, ["birds", "faces"])
            self.assertEqual([len(part) for part in parts], [54, 6, 20])
            sets = [set(part.id) for part in parts]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            for part, repeated in zip(parts, again):
                self.assertTrue(part.equals(repeated))

    def test_preflight_missing_data_fails_with_local_path(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(datasets=["pathmnist"], medmnist_root=directory)
            with self.assertRaisesRegex(FileNotFoundError, "pathmnist.npz"):
                preflight(settings, require_models=False)

    def test_split_zip_reassembles_as_valid_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            sink = SplitWriter(Path(directory) / "sample.zip", limit=73)
            try:
                with zipfile.ZipFile(sink, "w") as archive:
                    archive.writestr("test.txt", "payload" * 100)
            finally:
                sink.close()
            data = b"".join(path.read_bytes() for path in sink.parts)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                self.assertEqual(archive.read("test.txt"), b"payload" * 100)
                self.assertIsNone(archive.testzip())

    def test_notebook_has_no_active_download_or_install_calls(self):
        import ast
        notebook = json.loads(next(ROOT.glob("*.ipynb")).read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ("download_and_prepare", "dataset_download", "check_call", "get_file"), index)
                    if node.func.attr == "from_pretrained":
                        self.assertTrue(any(k.arg == "local_files_only" and isinstance(k.value, ast.Constant)
                                            and k.value.value is True for k in node.keywords), index)
                for keyword in node.keywords:
                    if keyword.arg == "download":
                        self.assertIs(keyword.value.value, False, index)


if __name__ == "__main__":
    unittest.main()
