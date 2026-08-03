from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import List, Tuple

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
load_dotenv()

# ── S3 coordinates ──────────────────────────────────────────────────────────
BUCKET = "stage-stereo-pipeline-v2-data"
PREFIX = "normalized/akai/2026-07-20/akai-ego-003_2026-07-20_05-37-38/000/"

def _list_objects(
    s3_client,
    bucket: str,
    prefix: str,
) -> List[Tuple[str, int]]:
    """Return a list of (key, size_bytes) under *prefix*."""
    objects: List[Tuple[str, int]] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append((obj["Key"], obj["Size"]))
    return objects


def _file_verified(local_path: pathlib.Path, expected_size: int) -> bool:
    """Return True if *local_path* exists and matches *expected_size*."""
    if not local_path.exists():
        return False
    return local_path.stat().st_size == expected_size


def _download_one(
    s3_client,
    bucket: str,
    key: str,
    dest: pathlib.Path,
) -> None:
    """Download a single object to *dest* via a .part temp file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        s3_client.download_file(bucket, key, str(part))
        part.rename(dest)
    except Exception:
        # Clean up partial download on failure
        if part.exists():
            part.unlink()
        raise


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download every object in the configured S3 folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data") / BUCKET / PREFIX,
        help="Local directory for downloaded files (default: data/<bucket>/<prefix>)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List remote objects and exit (no downloads).",
    )
    args = parser.parse_args()

    # Treat PREFIX as a folder even if it was entered without a trailing slash.
    prefix = PREFIX.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    # ── connect ─────────────────────────────────────────────────────────
    print(f"⬇  Connecting to S3…")
    print(f"   Bucket     : {BUCKET}")
    print(f"   Prefix     : {prefix}")
    try:

        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN")
        region = os.getenv("AWS_REGION", "us-east-1")

        if bool(access_key) != bool(secret_key):
            sys.exit("❌ Set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env.")

        session_kwargs = {"region_name": region} if region else {}
        if access_key:
            # An AWS_PROFILE in .env may refer to a profile that is not present
            # on this machine. Explicit credentials must take precedence.
            os.environ.pop("AWS_PROFILE", None)
            os.environ.pop("AWS_DEFAULT_PROFILE", None)
            session_kwargs.update(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token,
            )
            print("   AWS Auth   : explicit credentials from .env")
        else:
            print(f"   AWS Profile: {os.getenv('AWS_PROFILE', 'default')}")
        session = boto3.Session(**session_kwargs)
        s3 = session.client("s3")
        # Quick connectivity / permissions check
        s3.head_bucket(Bucket=BUCKET)
    except NoCredentialsError:
        sys.exit(
            "❌ No AWS credentials found.\n"
            "On EC2, attach an IAM role.  Locally, configure AWS_PROFILE or a .env file."
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("403", "AccessDenied"):
            sys.exit(f"❌ Access denied to s3://{BUCKET}  – check IAM permissions.")
        raise

    print(f"✅ Connected to s3://{BUCKET}/{prefix}")

    # ── enumerate remote objects ────────────────────────────────────────
    objects = _list_objects(s3, BUCKET, prefix)

    objects = [
        (key, size)
        for key, size in objects
        if (rel := key[len(prefix):]) and not rel.endswith("/")
    ]
    if not objects:
        sys.exit(f"No objects found under s3://{BUCKET}/{prefix}")

    # ── list-only mode ──────────────────────────────────────────────────
    if args.list_only:
        total_bytes = 0
        for key, size in objects:
            rel = key[len(prefix):]  # strip the common prefix for readability
            print(f"  {size:>12,} B  {rel}")
            total_bytes += size
        print(f"\n{len(objects)} object(s), {total_bytes:,} bytes total")
        return

    # ── download ────────────────────────────────────────────────────────
    output_dir: pathlib.Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded, skipped, failed = 0, 0, 0
    for idx, (key, size) in enumerate(objects, 1):
        rel = key[len(prefix):]
        local = output_dir / rel
        tag = f"[{idx}/{len(objects)}]"

        if _file_verified(local, size):
            print(f"{tag} SKIP (verified) {rel}")
            skipped += 1
            continue

        print(f"{tag} Downloading {rel}  ({size:,} B) …", end=" ", flush=True)
        try:
            _download_one(s3, BUCKET, key, local)
            print("OK")
            downloaded += 1
        except Exception as exc:
            print(f"FAILED ({exc})")
            failed += 1

    print(
        f"\nDone.  downloaded={downloaded}  skipped={skipped}  failed={failed}"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
