from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
from itertools import chain
from pathlib import Path
from threading import Event
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm


DATASET_NAME = "CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq"
OFFICIAL_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
USER_AGENT = "compressai-nano-anime-hf/1.0"
IMAGE_KEY_CANDIDATES = (
    "image",
    "jpg",
    "png",
    "file",
    "file_name",
    "filename",
    "image_url",
    "url",
    "source",
)
DANBOORU_CDN_EXTENSIONS = ("jpg", "png", "webp", "gif")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_existing_images(
    output_dir: Path,
    prefix: str,
    min_size: int,
) -> tuple[int, int, int, set[str]]:
    hashes: set[str] = set()
    split_counts = {"train": 0, "val": 0}
    max_index = 0

    for split in split_counts:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for path in split_dir.glob(f"{prefix}*.jpg"):
            try:
                digest = sha256_file(path)
                if digest in hashes:
                    continue
                with Image.open(path) as image:
                    image.load()
                    if image.width <= min_size or image.height <= min_size:
                        continue

                stem = path.stem
                suffix = stem.removeprefix(prefix)
                if suffix.isdigit():
                    max_index = max(max_index, int(suffix))

                hashes.add(digest)
                split_counts[split] += 1
            except OSError:
                continue
            except Exception:
                continue

    return split_counts["train"], split_counts["val"], max_index + 1, hashes


def find_image_key(example: dict[str, Any]) -> str | None:
    keys = list(example.keys())
    for key in IMAGE_KEY_CANDIDATES:
        if key in example:
            return key

    for key, value in example.items():
        if isinstance(value, Image.Image):
            return key
        if isinstance(value, str):
            lower = value.lower()
            if lower.startswith(("http://", "https://")):
                return key
            if any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
                return key
        if isinstance(value, dict) and ("bytes" in value or "path" in value):
            return key

    if "md5" in example or "id" in example:
        return None

    raise KeyError(f"Could not infer image source from dataset keys: {keys}")


def open_image_from_url(
    url: str,
    timeout: float,
    stop_event: Event | None = None,
) -> Image.Image | None:
    if stop_event is not None and stop_event.is_set():
        return None
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        if stop_event is not None and stop_event.is_set():
            return None
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and not content_type.startswith("image/"):
            return None
        return Image.open(BytesIO(response.content))
    except Exception:
        return None


def danbooru_cdn_urls(md5: str) -> list[str]:
    md5 = md5.strip().lower()
    if len(md5) < 4:
        return []
    prefix_a = md5[:2]
    prefix_b = md5[2:4]
    return [
        f"https://cdn.donmai.us/original/{prefix_a}/{prefix_b}/{md5}.{ext}"
        for ext in DANBOORU_CDN_EXTENSIONS
    ]


def fetch_danbooru_api_urls(
    post_id: Any,
    timeout: float,
    stop_event: Event | None = None,
) -> list[str]:
    if post_id is None or (stop_event is not None and stop_event.is_set()):
        return []
    try:
        response = requests.get(
            f"https://danbooru.donmai.us/posts/{post_id}.json",
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        if stop_event is not None and stop_event.is_set():
            return []
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    urls = []
    for key in ("file_url", "large_file_url", "preview_file_url"):
        url = payload.get(key)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            urls.append(url)
    return urls


def danbooru_record_to_image(
    example: dict[str, Any],
    timeout: float,
    stop_event: Event | None = None,
) -> Image.Image | None:
    md5 = str(example.get("md5") or "").strip().lower()
    for url in danbooru_cdn_urls(md5):
        if stop_event is not None and stop_event.is_set():
            return None
        image = open_image_from_url(url, timeout=timeout, stop_event=stop_event)
        if image is not None:
            return image

    for url in fetch_danbooru_api_urls(
        example.get("id"),
        timeout=timeout,
        stop_event=stop_event,
    ):
        if stop_event is not None and stop_event.is_set():
            return None
        image = open_image_from_url(url, timeout=timeout, stop_event=stop_event)
        if image is not None:
            return image

    return None


def value_to_image(
    value: Any,
    timeout: float,
    stop_event: Event | None = None,
) -> Image.Image | None:
    if stop_event is not None and stop_event.is_set():
        return None
    try:
        if isinstance(value, Image.Image):
            return value

        if isinstance(value, bytes):
            return Image.open(BytesIO(value))

        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(BytesIO(value["bytes"]))
            if value.get("path"):
                path_or_url = str(value["path"])
                return value_to_image(path_or_url, timeout, stop_event=stop_event)

        if isinstance(value, str):
            if value.startswith(("http://", "https://")):
                return open_image_from_url(
                    value,
                    timeout=timeout,
                    stop_event=stop_event,
                )
            path = Path(value)
            if path.exists():
                return Image.open(path)
    except Exception:
        return None

    return None


def example_to_image(
    example: dict[str, Any],
    image_key: str | None,
    timeout: float,
    stop_event: Event | None = None,
) -> Image.Image | None:
    if image_key is not None:
        image = value_to_image(
            example.get(image_key),
            timeout=timeout,
            stop_event=stop_event,
        )
        if image is not None:
            return image
    return danbooru_record_to_image(
        example,
        timeout=timeout,
        stop_event=stop_event,
    )


def encode_jpeg(image: Image.Image, min_size: int, jpeg_quality: int) -> bytes | None:
    try:
        image.load()
        width, height = image.size
        if width <= min_size or height <= min_size:
            return None

        buffer = BytesIO()
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
        )
        return buffer.getvalue()
    except Exception:
        return None


def download_and_encode_example(
    example: dict[str, Any],
    image_key: str | None,
    timeout: float,
    min_size: int,
    jpeg_quality: int,
    stop_event: Event | None = None,
) -> tuple[bytes | None, str | None, str | None]:
    if stop_event is not None and stop_event.is_set():
        return None, None, "cancelled"

    image = example_to_image(
        example,
        image_key=image_key,
        timeout=timeout,
        stop_event=stop_event,
    )
    if image is None:
        reason = (
            "cancelled"
            if stop_event is not None and stop_event.is_set()
            else "bad_image"
        )
        return None, None, reason

    data = encode_jpeg(image, min_size=min_size, jpeg_quality=jpeg_quality)
    if data is None:
        reason = (
            "cancelled"
            if stop_event is not None and stop_event.is_set()
            else "too_small_or_encode_failed"
        )
        return None, None, reason

    return data, sha256_bytes(data), None


def clear_old_prefixed_files(output_dir: Path, prefix: str) -> None:
    for split in ("train", "val"):
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for path in split_dir.glob(f"{prefix}*.jpg"):
            path.unlink(missing_ok=True)


def choose_split(
    train_count: int,
    val_count: int,
    target_train: int,
    target_val: int,
) -> str:
    if train_count >= target_train:
        return "val"
    if val_count >= target_val:
        return "train"

    train_fill = train_count / max(target_train, 1)
    val_fill = val_count / max(target_val, 1)
    return "train" if train_fill <= val_fill else "val"


def save_to_split(
    data: bytes,
    output_dir: Path,
    prefix: str,
    index: int,
    split: str,
) -> Path:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{prefix}{index:05d}.jpg"
    tmp_path = split_dir / f".{prefix}{index:05d}.tmp"
    tmp_path.write_bytes(data)
    tmp_path.replace(path)
    return path


def normalize_endpoint(endpoint: str) -> str:
    return endpoint.strip().rstrip("/")


def resolve_hf_endpoint(args: argparse.Namespace) -> str:
    if args.disable_hf_mirror:
        return OFFICIAL_HF_ENDPOINT
    endpoint = args.hf_endpoint or os.getenv("HF_ENDPOINT") or DEFAULT_HF_MIRROR_ENDPOINT
    return normalize_endpoint(endpoint)


def resolve_hf_token(args: argparse.Namespace) -> str | None:
    return (
        args.hf_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract anime images from a HuggingFace Danbooru dataset."
    )
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--prefix", type=str, default="anime_hf_")
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42, help="Kept for CLI compatibility.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of concurrent Danbooru image download workers.",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=None,
        help="Maximum pending download jobs. Defaults to workers * 4.",
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help=(
            "HuggingFace Hub endpoint. Defaults to HF_ENDPOINT, then "
            "https://hf-mirror.com."
        ),
    )
    parser.add_argument(
        "--disable-hf-mirror",
        action="store_true",
        help="Use the official https://huggingface.co endpoint instead of a mirror.",
    )
    token_mirror_group = parser.add_mutually_exclusive_group()
    token_mirror_group.add_argument(
        "--allow-token-to-mirror",
        dest="allow_token_to_mirror",
        action="store_true",
        help="Allow sending the HuggingFace token to a non-official endpoint.",
    )
    token_mirror_group.add_argument(
        "--no-token-to-mirror",
        dest="allow_token_to_mirror",
        action="store_false",
        help="Do not send the HuggingFace token to a non-official endpoint.",
    )
    parser.set_defaults(allow_token_to_mirror=True)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-retries", type=int, default=5)
    parser.add_argument("--dataset-retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--max-scan",
        type=int,
        default=200000,
        help="Maximum dataset rows to scan before giving up.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing anime_hf_*.jpg files and rebuild the anime subset.",
    )
    return parser.parse_args()


def load_dataset_with_retries(args: argparse.Namespace):
    last_error: Exception | None = None
    endpoint = resolve_hf_endpoint(args)
    os.environ["HF_ENDPOINT"] = endpoint
    print(f"Using HuggingFace endpoint: {endpoint}")

    from datasets import load_dataset

    token = resolve_hf_token(args)
    is_official_endpoint = normalize_endpoint(endpoint) == OFFICIAL_HF_ENDPOINT
    if token and not is_official_endpoint and not args.allow_token_to_mirror:
        print(
            "HuggingFace token detected, but the endpoint is not official. "
            "Not sending token to the mirror."
        )
        token = None

    if token:
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
            print("Logged in to HuggingFace Hub with token.")
        except Exception as exc:
            print(f"Warning: HuggingFace login failed, continuing with token: {exc}")

    for attempt in range(1, args.dataset_retries + 1):
        try:
            try:
                return load_dataset(
                    DATASET_NAME,
                    split="train",
                    streaming=args.streaming,
                    token=token,
                )
            except TypeError:
                return load_dataset(
                    DATASET_NAME,
                    split="train",
                    streaming=args.streaming,
                    use_auth_token=token,
                )
        except Exception as exc:
            last_error = exc
            if attempt >= args.dataset_retries:
                break
            sleep_seconds = args.dataset_retry_sleep * attempt
            print(
                f"load_dataset failed ({type(exc).__name__}: {exc}). "
                f"Retrying in {sleep_seconds:.1f}s [{attempt}/{args.dataset_retries}]..."
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to load HuggingFace dataset: {last_error}") from last_error


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.prefetch is not None and args.prefetch <= 0:
        raise ValueError("--prefetch must be positive")

    prefetch = args.prefetch or args.workers * 4

    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        clear_old_prefixed_files(args.output_dir, args.prefix)

    train_count, val_count, next_index, seen_hashes = scan_existing_images(
        args.output_dir,
        args.prefix,
        args.min_size,
    )
    target_train = int(round(args.count * 0.8))
    target_val = args.count - target_train
    accepted_total = train_count + val_count

    print(
        f"Existing anime images: train={train_count}, val={val_count}, "
        f"total={accepted_total}/{args.count}"
    )
    if accepted_total >= args.count:
        print("Done. Existing files already satisfy requested count.")
        return

    print(f"Loading HuggingFace dataset: {DATASET_NAME}")
    dataset = load_dataset_with_retries(args)
    iterator = iter(dataset)

    try:
        first = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("Dataset is empty.") from exc

    image_key = find_image_key(first)
    if image_key is None:
        print("No image field found; using md5/id to download images from Danbooru.")
    else:
        print(f"Detected image field: {image_key}")
    print(f"Dataset keys: {list(first.keys())}")
    print(f"Danbooru download workers: {args.workers}, prefetch: {prefetch}")

    scanned = 0
    skipped = {
        "bad_image": 0,
        "too_small_or_encode_failed": 0,
        "duplicate": 0,
        "worker_exception": 0,
    }

    def save_accepted_image(data: bytes, digest: str) -> None:
        nonlocal train_count, val_count, next_index, accepted_total
        seen_hashes.add(digest)
        split = choose_split(train_count, val_count, target_train, target_val)
        save_to_split(
            data,
            output_dir=args.output_dir,
            prefix=args.prefix,
            index=next_index,
            split=split,
        )
        next_index += 1
        accepted_total += 1
        if split == "train":
            train_count += 1
        else:
            val_count += 1

    with tqdm(
        total=args.count,
        initial=accepted_total,
        desc="anime images",
        unit="img",
    ) as progress:
        example_iter = chain([first], iterator)
        pending = set()
        stop_event = Event()

        def desired_pending_count() -> int:
            remaining = args.count - accepted_total
            if remaining <= 0:
                return 0
            return min(prefetch, max(1, remaining * 2))

        def cancel_extra_pending(limit: int) -> None:
            while len(pending) > limit:
                cancelled = False
                for future in list(pending):
                    if future.cancel():
                        pending.remove(future)
                        cancelled = True
                        break
                if not cancelled:
                    break

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            nonlocal scanned
            if accepted_total >= args.count or scanned >= args.max_scan:
                return False
            try:
                example = next(example_iter)
            except StopIteration:
                return False

            pending.add(
                executor.submit(
                    download_and_encode_example,
                    dict(example),
                    image_key,
                    args.timeout,
                    args.min_size,
                    args.jpeg_quality,
                    stop_event,
                )
            )
            scanned += 1
            return True

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while len(pending) < desired_pending_count() and submit_next(executor):
                pass

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        data, digest, reason = future.result()
                    except Exception:
                        skipped["worker_exception"] += 1
                        continue

                    if data is None or digest is None:
                        skipped[reason or "bad_image"] = skipped.get(
                            reason or "bad_image", 0
                        ) + 1
                        continue

                    if digest in seen_hashes:
                        skipped["duplicate"] += 1
                        continue

                    save_accepted_image(data, digest)
                    progress.update(1)
                    if accepted_total >= args.count:
                        stop_event.set()
                        break

                cancel_extra_pending(desired_pending_count())
                while (
                    accepted_total < args.count
                    and len(pending) < desired_pending_count()
                    and submit_next(executor)
                ):
                    pass

                progress.set_postfix(
                    train=train_count,
                    val=val_count,
                    scanned=scanned,
                    pending=len(pending),
                    skipped=sum(skipped.values()),
                )

                if accepted_total >= args.count:
                    stop_event.set()
                    break

            stop_event.set()
            for future in pending:
                future.cancel()

    if accepted_total < args.count:
        detail = ", ".join(f"{key}={value}" for key, value in skipped.items())
        raise RuntimeError(
            f"Only collected {accepted_total}/{args.count} images after scanning "
            f"{scanned} rows. Skipped: {detail}"
        )

    print("Done.")
    print(f"Saved total: {accepted_total}")
    print(f"Train: {train_count} -> {train_dir}")
    print(f"Val: {val_count} -> {val_dir}")
    print(f"Scanned rows: {scanned}")
    print("Skipped:", ", ".join(f"{key}={value}" for key, value in skipped.items()))


if __name__ == "__main__":
    main()
