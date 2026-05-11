from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Protocol
from urllib.parse import quote

import requests
from PIL import Image
from tqdm import tqdm


USER_AGENT = "compressai-nano-expand-dataset/1.1"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CATEGORIES = ("natural", "anime", "texture")
CATEGORY_RATIOS = {
    "natural": 0.60,
    "anime": 0.30,
    "texture": 0.10,
}
CATEGORY_LABELS = {
    "natural": "natural scene",
    "anime": "anime illustration",
    "texture": "texture/detail",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    sha256: str
    category: str
    width: int
    height: int
    source_url: str = ""


@dataclass(frozen=True)
class DownloadResult:
    record: ImageRecord | None
    reason: str | None = None


class UrlProvider(Protocol):
    def next_url(self, attempt_id: int) -> str:
        ...


class PicsumProvider:
    """極其穩定的隨機圖片提供者，用於取代失效的 Unsplash。"""
    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)

    def next_url(self, attempt_id: int) -> str:
        return f"https://picsum.photos/{self.image_size}/{self.image_size}?random={attempt_id}"


class ThemedProvider:
    """支持標籤篩選的穩定圖片提供者，用於獲取紋理與文檔數據。"""
    def __init__(self, image_size: int, keywords: str) -> None:
        self.image_size = int(image_size)
        self.keywords = keywords.strip().replace(" ", ",")

    def next_url(self, attempt_id: int) -> str:
        return f"https://loremflickr.com/{self.image_size}/{self.image_size}/{self.keywords}?lock={attempt_id}"


class SafebooruProvider:
    """線程安全的 Safebooru 高清插圖提供者。"""
    def __init__(
        self,
        tags: str,
        timeout: float,
        page_start: int = 0,
        page_limit: int = 100,
    ) -> None:
        self.tags = tags
        self.timeout = float(timeout)
        self.page_limit = max(1, min(int(page_limit), 100))
        self._next_page = int(page_start)
        self._urls: deque[str] = deque()
        self._lock = Lock()

    def next_url(self, attempt_id: int) -> str:
        del attempt_id
        with self._lock:
            while not self._urls:
                self._fetch_page()
            return self._urls.popleft()

    def _fetch_page(self) -> None:
        params = {
            "page": "dapi", "s": "post", "q": "index", "json": "1",
            "limit": self.page_limit, "pid": self._next_page, "tags": self.tags,
        }
        self._next_page += 1
        response = requests.get(
            "https://safebooru.org/index.php",
            params=params, headers={"User-Agent": USER_AGENT}, timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        posts = payload if isinstance(payload, list) else payload.get("post", [])
        urls = []
        for post in posts:
            raw_url = post.get("file_url") or post.get("sample_url") or post.get("preview_url")
            if not raw_url: continue
            if raw_url.startswith("//"): raw_url = f"https:{raw_url}"
            elif raw_url.startswith("/"): raw_url = f"https://safebooru.org{raw_url}"
            urls.append(raw_url)
        random.shuffle(urls)
        self._urls.extend(urls)
        if not self._urls: raise RuntimeError("Safebooru returned no usable image URLs.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists(): return {}
    manifest: dict[str, dict[str, str]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"): continue
        parts = line.split("\t")
        if len(parts) < 7: continue
        filename, split, category, digest, width, height, source_url = parts[:7]
        manifest[digest] = {"filename": filename, "split": split, "category": category, "width": width, "height": height, "source_url": source_url}
    return manifest


def scan_existing_images(roots: list[Path], manifest: dict[str, dict[str, str]], min_size: int) -> tuple[list[ImageRecord], set[str]]:
    records: list[ImageRecord] = []
    seen_hashes: set[str] = set()
    failures: Counter[str] = Counter()
    for root in roots:
        if not root.exists(): continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS: continue
            try:
                digest = sha256_file(path)
                if digest in seen_hashes:
                    failures["existing_duplicate"] += 1
                    continue
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
            except Exception:
                failures["existing_invalid"] += 1
                continue
            if width <= min_size or height <= min_size:
                failures["existing_too_small"] += 1
                continue
            meta = manifest.get(digest, {})
            category = meta.get("category", "existing")
            if category not in CATEGORIES: category = "existing"
            seen_hashes.add(digest)
            records.append(ImageRecord(path=path, sha256=digest, category=category, width=width, height=height, source_url=meta.get("source_url", "")))
    if failures:
        detail = ", ".join(f"{key}={value}" for key, value in failures.items())
        print(f"existing scan skipped: {detail}")
    return records, seen_hashes


def compute_category_targets(total_count: int) -> dict[str, int]:
    raw = {category: total_count * CATEGORY_RATIOS[category] for category in CATEGORIES}
    targets = {category: int(math.floor(value)) for category, value in raw.items()}
    remainder = total_count - sum(targets.values())
    ranked = sorted(CATEGORIES, key=lambda category: raw[category] - targets[category], reverse=True)
    for category in ranked[:remainder]: targets[category] += 1
    return targets


def allocate_unknown_existing(targets: dict[str, int], known_counts: Counter[str], unknown_count: int) -> dict[str, int]:
    missing = {category: max(targets[category] - known_counts.get(category, 0), 0) for category in CATEGORIES}
    for _ in range(unknown_count):
        category = max(missing, key=lambda item: missing[item])
        if missing[category] <= 0: break
        missing[category] -= 1
    return missing


def build_provider(category: str, args: argparse.Namespace) -> UrlProvider:
    if category == "natural": return PicsumProvider(args.image_size)
    if category == "texture": return ThemedProvider(args.image_size, args.texture_keywords)
    if category == "anime": return SafebooruProvider(tags=args.safebooru_tags, timeout=args.timeout)
    raise ValueError(f"Unknown category: {category}")


def validate_and_save_image(content: bytes, output_path: Path, category: str, source_url: str, min_size: int, jpeg_quality: int) -> DownloadResult:
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except Exception: return DownloadResult(record=None, reason="pil_open_failed")
    width, height = image.size
    if width <= min_size or height <= min_size: return DownloadResult(record=None, reason="too_small")
    try:
        rgb = image.convert("RGB")
        rgb.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)
    except Exception: return DownloadResult(record=None, reason="jpeg_save_failed")
    digest = sha256_file(output_path)
    return DownloadResult(record=ImageRecord(path=output_path, sha256=digest, category=category, width=width, height=height, source_url=source_url))


def download_one(category: str, attempt_id: int, provider: UrlProvider, pool_dir: Path, min_size: int, timeout: float, retries: int, jpeg_quality: int) -> DownloadResult:
    last_reason = "unknown"
    for _ in range(max(1, retries)):
        try: url = provider.next_url(attempt_id)
        except Exception: return DownloadResult(record=None, reason="url_provider_failed")
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
        except requests.RequestException:
            last_reason = "download_failed"; continue
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and not content_type.startswith("image/"):
            last_reason = "not_image"; continue
        output_path = pool_dir / f"{category}_{attempt_id:07d}_{time.time_ns()}.jpg"
        result = validate_and_save_image(response.content, output_path=output_path, category=category, source_url=response.url, min_size=min_size, jpeg_quality=jpeg_quality)
        if result.record is not None: return result
        last_reason = result.reason or last_reason
        output_path.unlink(missing_ok=True)
    return DownloadResult(record=None, reason=last_reason)


def collect_category(category: str, needed: int, provider: UrlProvider, pool_dir: Path, seen_hashes: set[str], args: argparse.Namespace) -> list[ImageRecord]:
    if needed <= 0: return []
    records, failures, pending, attempt_id = [], Counter(), set(), 1
    max_attempts = args.max_attempts_per_category or max(needed * 12, args.threads * 2)
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        with tqdm(total=needed, desc=f"{category} downloads", unit="img") as progress:
            while len(records) < needed:
                while len(pending) < args.threads and attempt_id <= max_attempts and len(records) < needed:
                    pending.add(executor.submit(download_one, category, attempt_id, provider, pool_dir, args.min_size, args.timeout, args.retries, args.jpeg_quality))
                    attempt_id += 1
                if not pending: break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try: result = future.result()
                    except Exception: failures["worker_exception"] += 1; continue
                    if result.record is None: failures[result.reason or "unknown"] += 1; continue
                    if result.record.sha256 in seen_hashes:
                        failures["duplicate"] += 1; result.record.path.unlink(missing_ok=True); continue
                    seen_hashes.add(result.record.sha256); records.append(result.record); progress.update(1)
                    if len(records) >= needed: break
            for future in pending: future.cancel()
    return records


def finalize_split(records: list[ImageRecord], output_dir: Path, train_ratio: float, total_count: int, seed: int) -> tuple[int, int, list[ImageRecord]]:
    train_dir, val_dir, manifest_path = output_dir / "train", output_dir / "val", output_dir / "dataset_manifest.tsv"
    unique: dict[str, ImageRecord] = {}
    for record in records:
        if record.path.exists() and record.sha256 not in unique: unique[record.sha256] = record
    selected = list(unique.values())
    random.Random(seed).shuffle(selected)
    if len(selected) > total_count: selected = selected[:total_count]
    train_count = int(round(len(selected) * train_ratio))
    tmp_root = output_dir / f"_split_tmp_{time.time_ns()}"
    tmp_train, tmp_val = tmp_root / "train", tmp_root / "val"
    tmp_train.mkdir(parents=True); tmp_val.mkdir(parents=True)
    try:
        for index, record in enumerate(selected, start=1):
            split_dir = tmp_train if index <= train_count else tmp_val
            target = split_dir / f"img_{index:04d}.jpg"
            shutil.move(str(record.path), target)
            selected[index - 1] = ImageRecord(path=target, sha256=record.sha256, category=record.category, width=record.width, height=record.height, source_url=record.source_url)
        if train_dir.exists(): shutil.rmtree(train_dir)
        if val_dir.exists(): shutil.rmtree(val_dir)
        tmp_train.rename(train_dir); tmp_val.rename(val_dir)
        lines = ["# filename\tsplit\tcategory\tsha256\twidth\theight\tsource_url"]
        for idx, rec in enumerate(selected, start=1):
            s = "train" if idx <= train_count else "val"
            lines.append(f"img_{idx:04d}.jpg\t{s}\t{rec.category}\t{rec.sha256}\t{rec.width}\t{rec.height}\t{rec.source_url}")
        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally: shutil.rmtree(tmp_root, ignore_errors=True)
    return train_count, len(selected) - train_count, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand compressai-nano dataset.")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-attempts-per-category", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--texture-keywords", type=str, default="texture,pattern,abstract,document")
    parser.add_argument("--safebooru-tags", type=str, default="highres rating:safe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir, pool_dir = args.output_dir, args.output_dir / "_expanded_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    manifest = parse_manifest(output_dir / "dataset_manifest.tsv")
    existing_records, seen_hashes = scan_existing_images([output_dir / "train", output_dir / "val", pool_dir], manifest, args.min_size)
    targets = compute_category_targets(args.count)
    existing_counts = Counter(record.category for record in existing_records)
    missing = allocate_unknown_existing(targets, existing_counts, existing_counts.get("existing", 0))
    
    print("Target status:")
    for cat in CATEGORIES: print(f"  {cat}: target={targets[cat]}, missing={missing[cat]}")

    all_records = list(existing_records)
    for cat in CATEGORIES:
        all_records.extend(collect_category(cat, missing[cat], build_provider(cat, args), pool_dir, seen_hashes, args))

    train_count, val_count, selected = finalize_split(all_records, output_dir, args.train_ratio, args.count, args.seed)
    print(f"✅ Dataset prepared: {train_count} train, {val_count} val. Manifest saved.")

if __name__ == "__main__":
    main()