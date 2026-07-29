"""Build and retrieve the Vercel-hosted hard-negative review application."""

from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from ourbrain_cv.manifest import read_manifest
from ourbrain_cv.reviews import deterministic_split

REMOTE_EXPORT_FIELDS = ("target_split", "review_note", "reviewer", "reviewed_at")
LOCAL_REVIEW_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_by_group(manifest_csv: Path) -> dict[str, str]:
    return {
        row["group_id"]: row["split"]
        for row in read_manifest(manifest_csv)
        if row.get("group_id") and row.get("split")
    }


def _candidate_path(value: str, review_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(review_dir)
    except ValueError as exc:
        raise ValueError(
            f"remote review candidate is outside the review directory: {candidate}"
        ) from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"remote review candidate does not exist: {candidate}")
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"unsupported remote review image type: {candidate.suffix}")
    return candidate


def _candidate_id(row: dict[str, str]) -> str:
    identity = "\0".join(
        row.get(field, "")
        for field in (
            "candidate_path",
            "source_image_path",
            "group_id",
            "left",
            "top",
            "right",
            "bottom",
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _remote_review_endpoint(base_url: str, resource: str) -> str:
    parsed = urlsplit(base_url)
    is_local_http = parsed.scheme == "http" and parsed.hostname in LOCAL_REVIEW_HOSTS
    if parsed.scheme != "https" and not is_local_http:
        raise ValueError(
            "remote review URL must use HTTPS "
            "(plain HTTP is allowed only for localhost development)"
        )
    if not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("remote review URL is invalid")
    return urljoin(base_url.rstrip("/") + "/", resource)


def _prepare_output(template_dir: Path, output_dir: Path) -> None:
    if template_dir == output_dir or template_dir in output_dir.parents:
        raise ValueError("remote review output must not contain the template directory")
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.name in {".vercel", ".env.local", ".gitignore"}:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        output_dir.mkdir(parents=True)
    shutil.copytree(template_dir, output_dir, dirs_exist_ok=True)


def build_remote_review_bundle(
    review_csv: str | Path,
    manifest_csv: str | Path,
    output_dir: str | Path,
    *,
    template_dir: str | Path = "web/remote-review-template",
    seed: int = 42,
) -> dict[str, Any]:
    """Create a Vercel source bundle whose candidate images upload to private Blob."""

    review_path = Path(review_csv).expanduser().resolve()
    manifest_path = Path(manifest_csv).expanduser().resolve()
    template_path = Path(template_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not review_path.is_file():
        raise FileNotFoundError(f"review CSV does not exist: {review_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    if not template_path.is_dir():
        raise FileNotFoundError(f"remote review template does not exist: {template_path}")
    if output_path in {Path.cwd().resolve(), review_path.parent, manifest_path.parent}:
        raise ValueError("remote review output must be a dedicated build directory")

    with review_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        source_rows = list(reader)
    required = {"candidate_path", "group_id", "review_label"}
    missing = sorted(required - set(source_fields))
    if missing:
        raise ValueError(f"review CSV is missing fields: {', '.join(missing)}")
    if not source_rows:
        raise ValueError("review CSV has no candidate rows")

    _prepare_output(template_path, output_path)
    candidate_dir = output_path / "private-candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    split_by_group = _split_by_group(manifest_path)
    rows: list[dict[str, str]] = []
    candidate_sources: list[tuple[Path, str, str]] = []
    seen_ids: set[str] = set()
    for source in source_rows:
        row = {field: source.get(field, "") for field in source_fields}
        candidate = _candidate_path(row["candidate_path"], review_path.parent)
        candidate_id = _candidate_id(row)
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate remote review candidate id: {candidate_id}")
        seen_ids.add(candidate_id)
        extension = candidate.suffix.lower()
        image_sha256 = _sha256(candidate)
        deployed_name = f"{candidate_id}-{image_sha256[:16]}{extension}"
        group_id = row["group_id"].strip()
        row.update(
            {
                "id": candidate_id,
                "imageSha256": image_sha256,
                "target_split": split_by_group.get(group_id)
                or deterministic_split(group_id, seed=seed),
                "review_label": row["review_label"].strip().lower(),
            }
        )
        rows.append(row)
        candidate_sources.append((candidate, deployed_name, extension))

    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    dataset_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    for row, (candidate, deployed_name, extension) in zip(
        rows, candidate_sources, strict=True
    ):
        shutil.copy2(candidate, candidate_dir / deployed_name)
        content_type = mimetypes.types_map.get(extension, "application/octet-stream")
        row.update(
            {
                "imageUrl": f"/api/candidate?id={row['id']}",
                "imageBlobPath": f"review-images/{dataset_id}/{deployed_name}",
                "imageContentType": content_type,
            }
        )
    export_fields = list(source_fields)
    for field in REMOTE_EXPORT_FIELDS:
        if field not in export_fields:
            export_fields.append(field)
    dataset_module = (
        f"export const DATASET_ID = {json.dumps(dataset_id)};\n"
        f"export const CANDIDATES = {json.dumps(rows, ensure_ascii=False)};\n"
        f"export const EXPORT_FIELDS = {json.dumps(export_fields)};\n"
    )
    library_dir = output_path / "lib"
    library_dir.mkdir(exist_ok=True)
    (library_dir / "dataset.js").write_text(dataset_module, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "candidates": len(rows),
        "review_csv": str(review_path),
        "review_csv_sha256": _sha256(review_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "output_dir": str(output_path),
        "private_image_bytes": sum(
            path.stat().st_size for path in candidate_dir.iterdir()
        ),
    }
    (output_path / "bundle-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def remote_review_status(
    base_url: str,
    *,
    token_env: str = "OURBRAIN_REVIEW_TOKEN",
    timeout: float = 30,
) -> dict[str, Any]:
    """Fetch authenticated progress from the deployed review service."""

    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"required environment variable is missing: {token_env}")
    endpoint = _remote_review_endpoint(base_url, "api/review-state")
    request = Request(  # noqa: S310 - caller explicitly selects the remote endpoint
        endpoint,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"remote review status failed ({exc.code}): {detail}") from exc
    if not isinstance(payload, dict) or "summary" not in payload:
        raise RuntimeError("remote review status returned an invalid response")
    return payload


def download_remote_review_csv(
    base_url: str,
    output_csv: str | Path,
    *,
    token_env: str = "OURBRAIN_REVIEW_TOKEN",
    timeout: float = 30,
) -> dict[str, Any]:
    """Download the server-side review state as a strict-import-compatible CSV."""

    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"required environment variable is missing: {token_env}")
    endpoint = _remote_review_endpoint(base_url, "api/review-state?format=csv")
    request = Request(  # noqa: S310 - caller explicitly selects the remote endpoint
        endpoint,
        headers={"Authorization": f"Bearer {token}", "Accept": "text/csv"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            content_type = response.headers.get_content_type()
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"remote review download failed ({exc.code}): {detail}") from exc
    if content_type != "text/csv":
        raise RuntimeError(f"remote review download returned {content_type}, not text/csv")
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(body)
    with output_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "output_csv": str(output_path),
        "rows": len(rows),
        "reviewed": sum(bool(row.get("review_label", "").strip()) for row in rows),
        "sha256": _sha256(output_path),
    }
