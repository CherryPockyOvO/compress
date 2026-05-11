from __future__ import annotations

import argparse
import hashlib
import os
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

import requests
from PIL import Image
from tqdm import tqdm


IMAGE_USER_AGENT = "compressai-nano-data-prep/1.0"


@dataclass(frozen=True)
class DownloadedImage:
    path: Path
    sha256: str
    width: int
    height: int
    source_url: str


@dataclass(frozen=True)
class DownloadResult:
    image: DownloadedImage | None
    reason: str | None = None


class UrlProvider(Protocol):
    def next_url(self, attempt_id: int) -> str:
        ...


class UnsplashSourceProvider:
    """URL provider for the no-key legacy Unsplash random image endpoint.

    The random endpoint redirects to an image CDN URL. A unique signature is
    added so repeated requests do not simply reuse a cached redirect.
    """

    def __init__(self, image_size: int, query: str | None) -> None:
        self.image_size = int(image_size)
        self.query = query.strip().replace(" ", ",") if query else ""

    def next_url(self, attempt_id: int) -> str:
        # Picsum 接口极其稳定，且不需要 API Key
        # 语法：https://picsum.photos/宽/高?random=序号
        return f"https://picsum.photos/{self.image_size}/{self.image_size}?random={attempt_id}"


class UnsplashApiProvider:
    """Thread-safe provider for the official Unsplash API random endpoint."""

    def __init__(
        self,
        access_key: str,
        image_size: int,
        query: str | None,
        timeout: float,
        batch_size: int = 30,
    ) -> None:
        if not access_key:
            raise ValueError(
                "Unsplash API mode requires --unsplash-access-key or "
                "the UNSPLASH_ACCESS_KEY environment variable."
            )
        self.access_key = access_key
        self.image_size = int(image_size)
        self.query = query
        self.timeout = float(timeout)
        self.batch_size = max(1, min(int(batch_size), 30))
        self._urls: deque[str] = deque()
        self._lock = Lock()

    def next_url(self, attempt_id: int) -> str:
        del attempt_id
        with self._lock:
            if not self._urls:
                self._fetch_batch()
            return self._urls.popleft()

    def _fetch_batch(self) -> None:
        params: dict[str, str | int] = {
            "count": self.batch_size,
            "content_filter": "high",
        }
        if self.query:
            params["query"] = self.query

        response = requests.get(
            "https://api.unsplash.com/photos/random",
            params=params,
            headers={
                "Authorization": f"Client-ID {self.access_key}",
                "Accept-Version": "v1",
                "User-Agent": IMAGE_USER_AGENT,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        photos = payload if isinstance(payload, list) else [payload]

        for photo in photos:
            urls = photo.get("urls") or {}
            raw_url = urls.get("raw") or urls.get("full") or urls.get("regular")
            if raw_url:
                separator = "&" if "?" in raw_url else "?"
                self._urls.append(
                    f"{raw_url}{separator}auto=format&fit=max"
                    f"&w={self.image_size}&q=95"
                )

        if not self._urls:
            raise RuntimeError("Unsplash API returned no usable image URLs.")


def build_url_provider(args: argparse.Namespace) -> UrlProvider:
    if args.source == "unsplash-api":
        access_key = args.unsplash_access_key or os.getenv("UNSPLASH_ACCESS_KEY", "")
        return UnsplashApiProvider(
            access_key=access_key,
            image_size=args.image_size,
            query=args.query,
            timeout=args.timeout,
            batch_size=args.api_batch_size,
        )
    return UnsplashSourceProvider(image_size=args.image_size, query=args.query)


def ensure_clean_split_dirs(train_dir: Path, val_dir: Path, overwrite: bool) -> None:
    """Create output split folders and avoid mixing old and new datasets."""

    for directory in (train_dir, val_dir):
        if directory.exists() and any(directory.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"{directory} is not empty. Pass --overwrite to replace it."
                )
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def validate_and_save_image(
    content: bytes,
    output_path: Path,
    min_size: int,
    jpeg_quality: int,
) -> DownloadResult:
    """Open bytes with PIL, filter by resolution, convert to RGB, and save JPEG."""

    try:
        image = Image.open(BytesIO(content))
        image.load()
    except Exception:
        return DownloadResult(image=None, reason="pil_open_failed")

    width, height = image.size
    if width <= min_size or height <= min_size:
        return DownloadResult(image=None, reason="too_small")

    try:
        rgb = image.convert("RGB")
        rgb.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)
    except Exception:
        return DownloadResult(image=None, reason="jpeg_save_failed")

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return DownloadResult(
        image=DownloadedImage(
            path=output_path,
            sha256=digest,
            width=width,
            height=height,
            source_url="",
        )
    )


def download_one(
    attempt_id: int,
    provider: UrlProvider,
    tmp_dir: Path,
    min_size: int,
    timeout: float,
    retries: int,
    jpeg_quality: int,
) -> DownloadResult:
    """Download one candidate image and return None when it should be skipped."""

    last_reason = "unknown"
    for _ in range(max(1, retries)):
        try:
            url = provider.next_url(attempt_id)
        except Exception:
            return DownloadResult(image=None, reason="url_provider_failed")

        try:
            response = requests.get(
                url,
                headers={"User-Agent": IMAGE_USER_AGENT},
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.RequestException:
            last_reason = "download_failed"
            continue

        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and not content_type.startswith("image/"):
            last_reason = "not_image"
            continue

        tmp_path = tmp_dir / f"candidate_{attempt_id:06d}_{time.time_ns()}.jpg"
        result = validate_and_save_image(
            response.content,
            output_path=tmp_path,
            min_size=min_size,
            jpeg_quality=jpeg_quality,
        )
        if result.image is not None:
            image = result.image
            return DownloadResult(
                image=DownloadedImage(
                    path=image.path,
                    sha256=image.sha256,
                    width=image.width,
                    height=image.height,
                    source_url=response.url,
                )
            )
        last_reason = result.reason or last_reason

    return DownloadResult(image=None, reason=last_reason)


def collect_images(args: argparse.Namespace, tmp_dir: Path) -> list[DownloadedImage]:
    """Run a bounded multi-threaded download loop until enough images pass filters."""

    provider = build_url_provider(args)
    failures: Counter[str] = Counter()
    accepted: list[DownloadedImage] = []
    seen_hashes: set[str] = set()
    pending = set()
    attempt_id = 1
    max_attempts = args.max_attempts or args.count * 8

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        with tqdm(total=args.count, desc="accepted images", unit="img") as progress:
            while len(accepted) < args.count:
                while (
                    len(pending) < args.threads
                    and attempt_id <= max_attempts
                    and len(accepted) < args.count
                ):
                    future = executor.submit(
                        download_one,
                        attempt_id,
                        provider,
                        tmp_dir,
                        args.min_size,
                        args.timeout,
                        args.retries,
                        args.jpeg_quality,
                    )
                    pending.add(future)
                    attempt_id += 1

                if not pending:
                    break

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        result = future.result()
                    except Exception:
                        failures["worker_exception"] += 1
                        continue

                    if result.image is None:
                        failures[result.reason or "unknown"] += 1
                        continue

                    if result.image.sha256 in seen_hashes:
                        failures["duplicate"] += 1
                        result.image.path.unlink(missing_ok=True)
                        continue

                    seen_hashes.add(result.image.sha256)
                    accepted.append(result.image)
                    progress.update(1)
                    progress.set_postfix(
                        attempts=attempt_id - 1,
                        failures=sum(failures.values()),
                    )

                    if len(accepted) >= args.count:
                        break

            for future in pending:
                future.cancel()

    if len(accepted) < args.count:
        failure_text = ", ".join(f"{k}={v}" for k, v in failures.most_common())
        raise RuntimeError(
            f"Only collected {len(accepted)}/{args.count} valid images after "
            f"{attempt_id - 1} attempts. Failures: {failure_text or 'none'}"
        )

    failure_text = ", ".join(f"{k}={v}" for k, v in failures.most_common())
    print(f"downloaded valid images: {len(accepted)}")
    print(f"skipped candidates: {failure_text or 'none'}")
    return accepted


def split_and_move(
    images: list[DownloadedImage],
    train_dir: Path,
    val_dir: Path,
    train_ratio: float,
    seed: int,
) -> tuple[int, int]:
    """Shuffle accepted images and move them into train/val folders."""

    rng = random.Random(seed)
    rng.shuffle(images)

    train_count = int(round(len(images) * train_ratio))
    train_images = images[:train_count]
    val_images = images[train_count:]

    global_index = 1
    for image in train_images:
        target = train_dir / f"img_{global_index:04d}.jpg"
        shutil.move(str(image.path), target)
        global_index += 1

    for image in val_images:
        target = val_dir / f"img_{global_index:04d}.jpg"
        shutil.move(str(image.path), target)
        global_index += 1

    return len(train_images), len(val_images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare a 1000-image train/val dataset."
    )
    parser.add_argument("--count", type=int, default=900)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--source",
        choices=("unsplash-source", "unsplash-api"),
        default="unsplash-source",
        help="Use no-key source.unsplash.com redirects or the official Unsplash API.",
    )
    parser.add_argument("--unsplash-access-key", type=str, default=None)
    parser.add_argument("--query", type=str, default="landscape,nature,city,architecture")
    parser.add_argument("--image-size", type=int, default=1400)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--api-batch-size", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing data/train and data/val before writing new files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")

    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"
    tmp_dir = args.output_dir / "_tmp_downloads"

    ensure_clean_split_dirs(train_dir, val_dir, overwrite=args.overwrite)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        images = collect_images(args, tmp_dir)
        train_count, val_count = split_and_move(
            images,
            train_dir=train_dir,
            val_dir=val_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("dataset prepared")
    print(f"train: {train_count} images -> {train_dir}")
    print(f"val  : {val_count} images -> {val_dir}")


if __name__ == "__main__":
    main()
