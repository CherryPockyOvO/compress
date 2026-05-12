from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
from itertools import chain
from pathlib import Path
from threading import Event
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from PIL import Image
from tqdm import tqdm


OFFICIAL_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
USER_AGENT = "compressai-nano-real-hf/1.0"

IMAGE_KEY_CANDIDATES = (
    "image",
    "input_image",
    "jpg",
    "png",
    "file",
    "file_name",
    "filename",
    "image_url",
    "url",
    "photo.image_url",
    "photo.url",
    "coco_url",
    "flickr_url",
)

DEFAULT_UNSPLASH_KEYWORDS = (
    "person",
    "human",
    "people",
    "portrait",
    "face",
    "hair",
    "woman",
    "man",
    "model",
    "fashion",
    "clothing",
    "apparel",
    "dress",
    "jacket",
    "shirt",
    "coat",
    "fabric",
    "textile",
    "leather",
    "texture",
    "indoor",
    "interior",
    "room",
    "home",
    "chair",
    "couch",
    "sofa",
    "window",
    "wall",
    "wood",
    "floor",
)


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str | None
    split: str
    count: int
    prefix: str
    min_size: int
    image_key: str | None = None
    text_keys: tuple[str, ...] = ()
    include_keywords: tuple[str, ...] = ()
    exclude_keywords: tuple[str, ...] = ()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_dataset_images(output_dir: Path) -> Iterable[Path]:
    for split in ("train", "val"):
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def scan_existing_hashes(output_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for path in iter_dataset_images(output_dir):
        try:
            hashes.add(sha256_file(path))
        except Exception:
            continue
    return hashes


def scan_existing_source_images(
    output_dir: Path,
    prefix: str,
    min_size: int,
) -> tuple[int, int, int]:
    split_counts = {"train": 0, "val": 0}
    max_index = 0

    for split in split_counts:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for path in split_dir.glob(f"{prefix}*.jpg"):
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.width < min_size or image.height < min_size:
                        continue
                stem = path.stem
                suffix = stem.removeprefix(prefix)
                if suffix.isdigit():
                    max_index = max(max_index, int(suffix))
                split_counts[split] += 1
            except Exception:
                continue

    return split_counts["train"], split_counts["val"], max_index + 1


def clear_old_prefixed_files(output_dir: Path, prefix: str) -> None:
    for split in ("train", "val"):
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for path in split_dir.glob(f"{prefix}*.jpg"):
            path.unlink(missing_ok=True)


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


def parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def get_by_path(example: dict[str, Any], path: str) -> Any:
    value: Any = example
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        parts: list[str] = []
        for nested in value.values():
            parts.extend(flatten_text(nested))
        return parts
    if isinstance(value, (list, tuple)):
        parts = []
        for nested in value:
            parts.extend(flatten_text(nested))
        return parts
    return []


def build_text_blob(example: dict[str, Any], text_keys: tuple[str, ...]) -> str:
    if text_keys:
        parts: list[str] = []
        for key in text_keys:
            parts.extend(flatten_text(get_by_path(example, key)))
        return " ".join(parts).lower()
    return " ".join(flatten_text(example)).lower()


def matches_keywords(
    example: dict[str, Any],
    include_keywords: tuple[str, ...],
    exclude_keywords: tuple[str, ...],
    text_keys: tuple[str, ...],
) -> bool:
    if not include_keywords and not exclude_keywords:
        return True

    text = build_text_blob(example, text_keys)
    if include_keywords and not any(keyword in text for keyword in include_keywords):
        return False
    if exclude_keywords and any(keyword in text for keyword in exclude_keywords):
        return False
    return True


def is_image_like_value(value: Any) -> bool:
    if isinstance(value, Image.Image):
        return True
    if isinstance(value, bytes):
        return True
    if isinstance(value, dict):
        if value.get("bytes") is not None or value.get("path"):
            return True
        if value.get("image_url"):
            return True
    if isinstance(value, str):
        lower = value.lower()
        if lower.startswith(("http://", "https://")):
            return True
        if any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
            return True
    return False


def find_nested_image_key(value: Any, prefix: str = "") -> str | None:
    if not isinstance(value, dict):
        return None

    for key, nested in value.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key in {"image", "input_image", "image_url"} and is_image_like_value(nested):
            return dotted

    for key, nested in value.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if is_image_like_value(nested):
            return dotted
        found = find_nested_image_key(nested, dotted)
        if found is not None:
            return found
    return None


def find_image_key(example: dict[str, Any]) -> str:
    for key in IMAGE_KEY_CANDIDATES:
        value = get_by_path(example, key)
        if is_image_like_value(value):
            return key

    found = find_nested_image_key(example)
    if found is not None:
        return found

    raise KeyError(f"Could not infer image source from dataset keys: {list(example.keys())}")


def unsplash_image_url(url: str, width: int) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "images.unsplash.com":
        return url
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("auto", "format")
    query.setdefault("fit", "max")
    query.setdefault("q", "95")
    query.setdefault("w", str(width))
    return urlunparse(parsed._replace(query=urlencode(query)))


def open_image_from_url(
    url: str,
    timeout: float,
    request_width: int,
    stop_event: Event | None = None,
) -> Image.Image | None:
    if stop_event is not None and stop_event.is_set():
        return None
    try:
        response = requests.get(
            unsplash_image_url(url, request_width),
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


def value_to_image(
    value: Any,
    timeout: float,
    request_width: int,
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
                return value_to_image(
                    value["path"],
                    timeout=timeout,
                    request_width=request_width,
                    stop_event=stop_event,
                )
            if value.get("image_url"):
                return value_to_image(
                    value["image_url"],
                    timeout=timeout,
                    request_width=request_width,
                    stop_event=stop_event,
                )

        if isinstance(value, str):
            if value.startswith(("http://", "https://")):
                return open_image_from_url(
                    value,
                    timeout=timeout,
                    request_width=request_width,
                    stop_event=stop_event,
                )
            path = Path(value)
            if path.exists():
                return Image.open(path)
    except Exception:
        return None

    return None


def encode_jpeg(
    image: Image.Image,
    min_size: int,
    jpeg_quality: int,
    max_aspect: float,
) -> bytes | None:
    try:
        image.load()
        width, height = image.size
        if width < min_size or height < min_size:
            return None
        aspect = max(width / max(height, 1), height / max(width, 1))
        if aspect > max_aspect:
            return None

        buffer = BytesIO()
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            subsampling=0,
        )
        return buffer.getvalue()
    except Exception:
        return None


def fetch_and_encode_example(
    example: dict[str, Any],
    image_key: str,
    source: SourceSpec,
    args: argparse.Namespace,
    stop_event: Event | None = None,
) -> tuple[bytes | None, str | None, str | None]:
    if stop_event is not None and stop_event.is_set():
        return None, None, "cancelled"

    if not matches_keywords(
        example,
        source.include_keywords,
        source.exclude_keywords,
        source.text_keys,
    ):
        return None, None, "keyword_filter"

    image = value_to_image(
        get_by_path(example, image_key),
        timeout=args.timeout,
        request_width=args.request_width,
        stop_event=stop_event,
    )
    if image is None:
        reason = "cancelled" if stop_event is not None and stop_event.is_set() else "bad_image"
        return None, None, reason

    data = encode_jpeg(
        image,
        min_size=source.min_size,
        jpeg_quality=args.jpeg_quality,
        max_aspect=args.max_aspect,
    )
    if data is None:
        reason = (
            "cancelled"
            if stop_event is not None and stop_event.is_set()
            else "too_small_or_encode_failed"
        )
        return None, None, reason

    return data, sha256_bytes(data), None


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


def load_dataset_with_retries(source: SourceSpec, args: argparse.Namespace):
    last_error: Exception | None = None
    endpoint = resolve_hf_endpoint(args)
    os.environ["HF_ENDPOINT"] = endpoint

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

    load_kwargs: dict[str, Any] = {
        "split": source.split,
        "streaming": args.streaming,
        "token": token,
        "trust_remote_code": args.trust_remote_code,
    }
    if source.config:
        load_kwargs["name"] = source.config

    for attempt in range(1, args.dataset_retries + 1):
        try:
            try:
                return load_dataset(source.dataset, **load_kwargs)
            except TypeError:
                legacy_kwargs = dict(load_kwargs)
                legacy_kwargs.pop("token", None)
                legacy_kwargs["use_auth_token"] = token
                return load_dataset(source.dataset, **legacy_kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= args.dataset_retries:
                break
            sleep_seconds = args.dataset_retry_sleep * attempt
            print(
                f"[{source.name}] load_dataset failed ({type(exc).__name__}: {exc}). "
                f"Retrying in {sleep_seconds:.1f}s [{attempt}/{args.dataset_retries}]..."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to load HuggingFace dataset {source.dataset}: {last_error}") from last_error


def collect_source(
    source: SourceSpec,
    args: argparse.Namespace,
    seen_hashes: set[str],
) -> None:
    if source.count <= 0:
        print(f"[{source.name}] skipped because target count is 0.")
        return

    train_count, val_count, next_index = scan_existing_source_images(
        args.output_dir,
        source.prefix,
        source.min_size,
    )
    target_train = int(round(source.count * args.train_ratio))
    target_val = source.count - target_train
    accepted_total = train_count + val_count

    print(
        f"[{source.name}] existing: train={train_count}, val={val_count}, "
        f"total={accepted_total}/{source.count}"
    )
    if accepted_total >= source.count:
        print(f"[{source.name}] done. Existing files already satisfy requested count.")
        return

    print(
        f"[{source.name}] loading {source.dataset}"
        + (f"/{source.config}" if source.config else "")
        + f" split={source.split}"
    )
    dataset = load_dataset_with_retries(source, args)
    iterator = iter(dataset)

    try:
        first = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"[{source.name}] dataset is empty.") from exc

    image_key = source.image_key or find_image_key(dict(first))
    print(f"[{source.name}] image field: {image_key}")
    print(f"[{source.name}] dataset keys: {list(dict(first).keys())}")
    print(f"[{source.name}] workers={args.workers}, prefetch={args.prefetch}")

    scanned = 0
    max_scan = args.max_scan or max(source.count * args.max_scan_multiplier, source.count)
    skipped: dict[str, int] = {
        "keyword_filter": 0,
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
            prefix=source.prefix,
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
        total=source.count,
        initial=accepted_total,
        desc=source.name,
        unit="img",
    ) as progress:
        example_iter = chain([first], iterator)
        pending = set()
        stop_event = Event()

        def desired_pending_count() -> int:
            remaining = source.count - accepted_total
            if remaining <= 0:
                return 0
            return min(args.prefetch, max(1, remaining * 2))

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
            if accepted_total >= source.count or scanned >= max_scan:
                return False
            try:
                example = next(example_iter)
            except StopIteration:
                return False

            pending.add(
                executor.submit(
                    fetch_and_encode_example,
                    dict(example),
                    image_key,
                    source,
                    args,
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
                        skipped[reason or "bad_image"] = skipped.get(reason or "bad_image", 0) + 1
                        continue

                    if digest in seen_hashes:
                        skipped["duplicate"] += 1
                        continue

                    save_accepted_image(data, digest)
                    progress.update(1)
                    if accepted_total >= source.count:
                        stop_event.set()
                        break

                cancel_extra_pending(desired_pending_count())
                while (
                    accepted_total < source.count
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

                if accepted_total >= source.count:
                    stop_event.set()
                    break

            stop_event.set()
            for future in pending:
                future.cancel()

    if accepted_total < source.count:
        detail = ", ".join(f"{key}={value}" for key, value in skipped.items())
        raise RuntimeError(
            f"[{source.name}] only collected {accepted_total}/{source.count} images "
            f"after scanning {scanned} rows. Skipped: {detail}"
        )

    print(f"[{source.name}] done.")
    print(f"[{source.name}] saved total: {accepted_total}")
    print(f"[{source.name}] train: {train_count}")
    print(f"[{source.name}] val: {val_count}")
    print(f"[{source.name}] scanned rows: {scanned}")
    print(f"[{source.name}] skipped: " + ", ".join(f"{key}={value}" for key, value in skipped.items()))


def build_sources(args: argparse.Namespace) -> list[SourceSpec]:
    ffhq_min_size = args.ffhq_min_size if args.ffhq_min_size is not None else max(args.min_size, 768)
    unsplash_min_size = args.unsplash_min_size if args.unsplash_min_size is not None else args.min_size
    coco_min_size = args.coco_min_size if args.coco_min_size is not None else args.min_size

    sources = [
        SourceSpec(
            name="ffhq",
            dataset=args.ffhq_dataset,
            config=args.ffhq_config,
            split=args.ffhq_split,
            count=args.ffhq_count,
            prefix=args.ffhq_prefix,
            min_size=ffhq_min_size,
            image_key=args.ffhq_image_key,
        ),
        SourceSpec(
            name="unsplash",
            dataset=args.unsplash_dataset,
            config=args.unsplash_config,
            split=args.unsplash_split,
            count=args.unsplash_count,
            prefix=args.unsplash_prefix,
            min_size=unsplash_min_size,
            image_key=args.unsplash_image_key,
            text_keys=parse_csv(args.unsplash_text_keys) or ("keywords", "photo.description", "ai.description"),
            include_keywords=parse_csv(args.unsplash_keywords) or DEFAULT_UNSPLASH_KEYWORDS,
            exclude_keywords=parse_csv(args.unsplash_exclude_keywords),
        ),
        SourceSpec(
            name="coco",
            dataset=args.coco_dataset,
            config=args.coco_config,
            split=args.coco_split,
            count=args.coco_count,
            prefix=args.coco_prefix,
            min_size=coco_min_size,
            image_key=args.coco_image_key,
            text_keys=parse_csv(args.coco_text_keys),
            include_keywords=parse_csv(args.coco_keywords),
            exclude_keywords=parse_csv(args.coco_exclude_keywords),
        ),
    ]

    if args.sources:
        enabled = set(parse_csv(args.sources))
        sources = [source for source in sources if source.name in enabled]
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a high-detail real-image training mix from HuggingFace datasets "
            "through a mirror endpoint."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--sources", type=str, default=None, help="Comma-separated subset: ffhq,unsplash,coco.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--min-size", type=int, default=512, help="Global fallback minimum side length.")
    parser.add_argument("--jpeg-quality", type=int, default=96)
    parser.add_argument("--max-aspect", type=float, default=3.0)
    parser.add_argument("--request-width", type=int, default=1600, help="Requested width for URL-backed images.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--prefetch", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--ffhq-count", type=int, default=12000)
    parser.add_argument("--ffhq-dataset", type=str, default="bitmind/ffhq")
    parser.add_argument("--ffhq-config", type=str, default=None)
    parser.add_argument("--ffhq-split", type=str, default="train")
    parser.add_argument("--ffhq-prefix", type=str, default="ffhq_hf_")
    parser.add_argument("--ffhq-min-size", type=int, default=None)
    parser.add_argument("--ffhq-image-key", type=str, default=None)

    parser.add_argument("--unsplash-count", type=int, default=6000)
    parser.add_argument("--unsplash-dataset", type=str, default="sentence-transformers/unsplash-lite")
    parser.add_argument("--unsplash-config", type=str, default=None)
    parser.add_argument("--unsplash-split", type=str, default="train")
    parser.add_argument("--unsplash-prefix", type=str, default="unsplash_hf_")
    parser.add_argument("--unsplash-min-size", type=int, default=None)
    parser.add_argument("--unsplash-image-key", type=str, default=None)
    parser.add_argument(
        "--unsplash-text-keys",
        type=str,
        default=None,
        help="Comma-separated metadata fields used for keyword filtering.",
    )
    parser.add_argument(
        "--unsplash-keywords",
        type=str,
        default=None,
        help="Comma-separated include keywords. Defaults to portrait/indoor/clothing/material terms.",
    )
    parser.add_argument("--unsplash-exclude-keywords", type=str, default=None)

    parser.add_argument("--coco-count", type=int, default=2000)
    parser.add_argument("--coco-dataset", type=str, default="phiyodr/coco2017")
    parser.add_argument("--coco-config", type=str, default=None)
    parser.add_argument("--coco-split", type=str, default="train")
    parser.add_argument("--coco-prefix", type=str, default="coco_hf_")
    parser.add_argument("--coco-min-size", type=int, default=None)
    parser.add_argument("--coco-image-key", type=str, default=None)
    parser.add_argument("--coco-text-keys", type=str, default=None)
    parser.add_argument("--coco-keywords", type=str, default=None)
    parser.add_argument("--coco-exclude-keywords", type=str, default=None)

    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help="HuggingFace Hub endpoint. Defaults to HF_ENDPOINT, then https://hf-mirror.com.",
    )
    parser.add_argument(
        "--disable-hf-mirror",
        action="store_true",
        help="Use official https://huggingface.co instead of a mirror.",
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
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dataset-retries", type=int, default=5)
    parser.add_argument("--dataset-retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="Maximum rows to scan per source. Defaults to count * max-scan-multiplier.",
    )
    parser.add_argument("--max-scan-multiplier", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.prefetch <= 0:
        raise ValueError("--prefetch must be positive")
    if args.jpeg_quality <= 0 or args.jpeg_quality > 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "val").mkdir(parents=True, exist_ok=True)

    endpoint = resolve_hf_endpoint(args)
    print(f"Using HuggingFace endpoint: {endpoint}")
    print(f"Output directory: {args.output_dir}")

    sources = build_sources(args)
    if not sources:
        raise ValueError("No sources enabled.")

    if args.overwrite:
        print("Overwrite enabled: clearing files created by this script.")
        for source in sources:
            clear_old_prefixed_files(args.output_dir, source.prefix)

    print("Scanning existing data hashes for dedupe...")
    seen_hashes = scan_existing_hashes(args.output_dir)
    print(f"Existing dedupe hashes: {len(seen_hashes)}")

    for source in sources:
        collect_source(source, args, seen_hashes)

    print("Done.")
    print(f"Train directory: {args.output_dir / 'train'}")
    print(f"Val directory: {args.output_dir / 'val'}")


if __name__ == "__main__":
    main()
