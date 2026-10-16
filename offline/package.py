"""Create the small source ZIP or the complete, split offline transfer ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HOME = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_files():
    # Git knows source membership; do not accidentally ship data, credentials or .git.
    result = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=HOME)
    return sorted({HOME / p for p in result.decode().split("\0") if p and not p.startswith("downloads/")
                   and not p.endswith(".zip") and (HOME / p).is_file()})


class SplitWriter:
    """Unseekable ZIP sink; no extra multi-GB unsplit copy on the build host."""
    def __init__(self, base, limit=1500 * 1024 * 1024):
        self.base, self.limit = base, limit
        self.total = self.part_size = self.number = 0
        self.stream = None
        self.parts = []

    def tell(self):
        return self.total

    def write(self, data):
        original = len(data)
        view = memoryview(data)
        while view:
            if self.stream is None or self.part_size == self.limit:
                if self.stream:
                    self.stream.close()
                self.number += 1
                path = self.base.with_name(self.base.name + f".part{self.number:04d}")
                self.parts.append(path)
                self.stream = path.open("wb")
                self.part_size = 0
            count = min(len(view), self.limit - self.part_size)
            self.stream.write(view[:count])
            self.total += count
            self.part_size += count
            view = view[count:]
        return original

    def flush(self):
        if self.stream:
            self.stream.flush()

    def close(self):
        if self.stream:
            self.stream.close()


def verify():
    manifest = HOME / "offline/payload-manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Full payload checksum manifest missing")
    files = json.loads(manifest.read_text())
    for relative, expected in files.items():
        path = HOME / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing/corrupt payload file: {relative}")
    print(f"PASS: {len(files)} payload files verified with SHA256")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    files = source_files()
    out = HOME / "dist"
    out.mkdir(exist_ok=True)
    if not args.full:
        target = out / "CBM-offline-setup.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, str(Path("CBM") / path.relative_to(HOME)))
        with zipfile.ZipFile(target) as archive:
            if archive.testzip():
                raise ValueError("Source ZIP integrity check failed")
        print(target)
        return
    payload = [HOME / "offline" / p for p in (
        "environment.tar.gz", "build-info.json", "resolved-requirements.txt", "conda-explicit.txt")]
    payload += [p for p in (HOME / "offline/models").rglob("*") if p.is_file()
                and ".cache" not in p.relative_to(HOME / "offline/models").parts]
    if not all(p.is_file() for p in payload) or not any(p.suffix == ".safetensors" for p in payload):
        raise FileNotFoundError("Build the complete environment AND pretrained models before packaging")
    manifest = HOME / "offline/payload-manifest.json"
    # Cover source instructions as well as binary assets.
    files = sorted(set(files + payload))
    files = [p for p in files if p != manifest]
    manifest.write_text(json.dumps({str(p.relative_to(HOME)): digest(p) for p in files}, indent=2) + "\n")
    files.append(manifest)
    base = out / "CBM-offline-full.zip"
    if list(out.glob(base.name + ".part*")):
        raise FileExistsError("Old split payload exists; move/remove it before building again")
    sink = SplitWriter(base)
    try:
        # environment.tar.gz is compressed already; weights gain little from DEFLATE.
        with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in files:
                archive.write(path, str(Path("CBM") / path.relative_to(HOME)))
    finally:
        sink.close()
    (out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sink.parts))
    import shutil
    shutil.copyfile(HOME / "offline/assemble.ps1", out / "assemble.ps1")
    print(f"Full ZIP: {len(sink.parts)} parts, {sink.total:,} bytes; transfer ALL parts")


if __name__ == "__main__":
    main()
