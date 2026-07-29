"""Generate a dependency-free local UI for hard-negative review."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import webbrowser
from collections import Counter
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ourbrain_cv.manifest import read_manifest
from ourbrain_cv.reviews import NEGATIVE_LABELS, deterministic_split

REVIEW_LABELS = ("negative", "crack", "uncertain")
REVIEW_LABEL_ALIASES = {
    **{label: "negative" for label in NEGATIVE_LABELS},
    "crack": "crack",
    "positive": "crack",
    "1": "crack",
    "uncertain": "uncertain",
    "unsure": "uncertain",
}


def normalize_review_label(value: str) -> str:
    """Normalize supported review labels while preserving a blank decision."""

    stripped = value.strip().lower()
    if not stripped:
        return ""
    return REVIEW_LABEL_ALIASES.get(stripped, "uncertain")


def _split_by_group(manifest_csv: str | Path | None) -> dict[str, str]:
    if manifest_csv is None:
        return {}
    return {
        row.get("group_id", ""): row.get("split", "")
        for row in read_manifest(manifest_csv)
        if row.get("group_id") and row.get("split")
    }


def _candidate_file_url(candidate_path: str, output_dir: Path) -> tuple[str, bool]:
    candidate = Path(candidate_path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    try:
        relative = candidate.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(
            f"candidate is outside the review server root: {candidate}. "
            "Place --output inside the candidate image directory."
        ) from exc
    image_url = "/".join(quote(part) for part in relative.parts)
    return image_url, candidate.is_file()


def _review_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = Counter(row["review_label"] or "unreviewed" for row in rows)
    by_split: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["target_split"] == split]
        split_labels = Counter(row["review_label"] or "unreviewed" for row in split_rows)
        by_split[split] = {
            "total": len(split_rows),
            "negative": split_labels["negative"],
            "crack": split_labels["crack"],
            "uncertain": split_labels["uncertain"],
            "unreviewed": split_labels["unreviewed"],
        }
    return {
        "total": len(rows),
        "reviewed": len(rows) - labels["unreviewed"],
        "negative": labels["negative"],
        "crack": labels["crack"],
        "uncertain": labels["uncertain"],
        "unreviewed": labels["unreviewed"],
        "missing_candidate_files": sum(row["candidate_exists"] != "true" for row in rows),
        "by_split": by_split,
    }


def build_negative_review_ui(
    review_csv: str | Path,
    output_html: str | Path,
    *,
    manifest_csv: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a self-contained local HTML reviewer and return review statistics.

    The UI never writes source data. Decisions are persisted in browser local
    storage and exported as a new CSV. Only labels exported as ``negative`` are
    accepted by :func:`ourbrain_cv.reviews.import_reviewed_negatives`.
    """

    review_path = Path(review_csv).expanduser().resolve()
    output_path = Path(output_html).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with review_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)
    required_fields = {"candidate_path", "group_id", "review_label"}
    missing_fields = sorted(required_fields - set(fieldnames))
    if missing_fields:
        raise ValueError(f"review CSV is missing fields: {', '.join(missing_fields)}")

    split_by_group = _split_by_group(manifest_csv)
    rows: list[dict[str, str]] = []
    for source in source_rows:
        row = {field: source.get(field, "") for field in fieldnames}
        group_id = row["group_id"].strip()
        target_split = split_by_group.get(group_id) or deterministic_split(
            group_id, seed=seed
        )
        image_url, candidate_exists = _candidate_file_url(
            row["candidate_path"], output_path.parent
        )
        row["review_label"] = normalize_review_label(row["review_label"])
        row["target_split"] = target_split
        row["image_url"] = image_url
        row["candidate_exists"] = "true" if candidate_exists else "false"
        rows.append(row)

    export_fields = [*fieldnames]
    if "target_split" not in export_fields:
        export_fields.append("target_split")
    summary = _review_summary(rows)
    storage_key = "ourbrain-negative-review-" + hashlib.sha256(
        str(review_path).encode()
    ).hexdigest()[:16]
    payload = _script_safe_json(rows)
    fields_payload = _script_safe_json(export_fields)
    summary_payload = json.dumps(summary, ensure_ascii=False)
    document = _render_document(
        rows_payload=payload,
        fields_payload=fields_payload,
        summary_payload=summary_payload,
        storage_key=storage_key,
        title=f"OurBrain Hard-negative Review · {review_path.name}",
    )
    output_path.write_text(document, encoding="utf-8")
    return {
        **summary,
        "review_csv": str(review_path),
        "review_html": str(output_path),
        "storage_key": storage_key,
    }


def create_review_server(
    review_html: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> tuple[ThreadingHTTPServer, str]:
    """Create a loopback-friendly static server for stable browser storage."""

    review_path = Path(review_html).expanduser().resolve()
    if not review_path.is_file():
        raise FileNotFoundError(f"review HTML does not exist: {review_path}")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("review server host must be loopback-only")
    handler = partial(SimpleHTTPRequestHandler, directory=str(review_path.parent))
    server = ThreadingHTTPServer((host, port), handler)
    _, bound_port = server.server_address[:2]
    url_host = "[::1]" if host == "::1" else host
    if port == 0:
        bound_port = int(bound_port)
    return server, f"http://{url_host}:{bound_port}/{quote(review_path.name)}"


def _script_safe_json(value: Any) -> str:
    """Serialize JSON without allowing an embedded value to close a script tag."""

    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def serve_review_ui(
    review_html: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Serve a generated review UI until interrupted."""

    server, url = create_review_server(review_html, host=host, port=port)
    print(
        json.dumps(
            {
                "review_url": url,
                "status": "serving",
                "stop": "Ctrl-C",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _render_document(
    *,
    rows_payload: str,
    fields_payload: str,
    summary_payload: str,
    storage_key: str,
    title: str,
) -> str:
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, Pretendard, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0b0f14;
      color: #e8edf2;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; }}
    header {{
      display: flex; align-items: center; justify-content: space-between; gap: 20px;
      padding: 16px 24px; border-bottom: 1px solid #26303b; background: #10161d;
      position: sticky; top: 0; z-index: 2;
    }}
    h1 {{ margin: 0; font-size: 18px; }}
    .progress {{ min-width: 280px; }}
    .progress-bar {{ height: 8px; background: #26303b; border-radius: 999px; overflow: hidden; }}
    .progress-fill {{ height: 100%; width: 0; background: #42d392; transition: width .15s; }}
    .progress-text {{ margin-top: 6px; color: #9aa9b8; font-size: 13px; text-align: right; }}
    main {{
      display: grid; grid-template-columns: minmax(520px, 1fr) 360px;
      gap: 24px; padding: 24px; max-width: 1400px; margin: 0 auto;
    }}
    .viewer, .panel {{ background: #111820; border: 1px solid #26303b; border-radius: 14px; }}
    .viewer {{ min-height: 620px; display: grid; place-items: center; padding: 18px; }}
    #candidate-image {{
      width: min(100%, 860px); max-height: 72vh; object-fit: contain;
      image-rendering: auto; border-radius: 8px; background: #070a0e;
    }}
    .panel {{ padding: 20px; align-self: start; position: sticky; top: 88px; }}
    .eyebrow {{ color: #42d392; font-weight: 700; font-size: 12px; text-transform: uppercase; }}
    #filename {{ margin: 7px 0 4px; font-size: 17px; word-break: break-all; }}
    #metadata {{ color: #9aa9b8; font-size: 13px; line-height: 1.65; white-space: pre-line; }}
    .instructions {{ color: #c3ccd5; line-height: 1.55; font-size: 14px; margin: 18px 0; }}
    .labels {{ display: grid; gap: 10px; }}
    button {{
      border: 1px solid #33404d; background: #18212b; color: #e8edf2;
      border-radius: 9px; padding: 12px 14px; cursor: pointer; font-weight: 700;
    }}
    button:hover {{ border-color: #637587; }}
    button.active {{ outline: 2px solid #42d392; border-color: #42d392; }}
    button.negative {{ background: #143427; }}
    button.crack {{ background: #3d1d24; }}
    button.uncertain {{ background: #3a3218; }}
    .nav {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }}
    .secondary {{ background: transparent; font-weight: 600; }}
    .summary {{
      margin-top: 20px; border-top: 1px solid #26303b; padding-top: 16px;
      font-size: 13px; color: #aeb9c4; line-height: 1.7;
    }}
    .summary table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    .summary th, .summary td {{ text-align: right; padding: 4px; }}
    .summary th:first-child, .summary td:first-child {{ text-align: left; }}
    .footer-actions {{ display: grid; gap: 10px; margin-top: 18px; }}
    .export {{ background: #236b4e; border-color: #42d392; }}
    .missing {{ color: #ff7a90; font-weight: 700; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; padding: 14px; }}
      .viewer {{ min-height: auto; }}
      .panel {{ position: static; }}
      header {{ align-items: flex-start; flex-direction: column; }}
      .progress {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <div class="progress">
      <div class="progress-bar"><div id="progress-fill" class="progress-fill"></div></div>
      <div id="progress-text" class="progress-text"></div>
    </div>
  </header>
  <main>
    <section class="viewer">
      <img id="candidate-image" alt="검수 후보 이미지">
    </section>
    <aside class="panel">
      <div id="split" class="eyebrow"></div>
      <h2 id="filename"></h2>
      <div id="metadata"></div>
      <p class="instructions">
        <strong>negative</strong>는 균열이 없다고 확신할 때만 선택하세요.
        균열 또는 균열 의심은 <strong>crack</strong>, 판단이 어려우면
        <strong>uncertain</strong>으로 남깁니다. N/C/U 단축키를 사용할 수 있습니다.
      </p>
      <div class="labels">
        <button class="negative" data-label="negative">N · 균열 없음</button>
        <button class="crack" data-label="crack">C · 균열/균열 의심</button>
        <button class="uncertain" data-label="uncertain">U · 판단 보류</button>
      </div>
      <div class="nav">
        <button id="previous" class="secondary">← 이전</button>
        <button id="next" class="secondary">다음 →</button>
      </div>
      <div class="nav">
        <button id="next-unreviewed" class="secondary">다음 미검수</button>
        <button id="clear-label" class="secondary">현재 라벨 해제</button>
      </div>
      <div id="summary" class="summary"></div>
      <div class="footer-actions">
        <button id="export" class="export">검수 CSV 내보내기</button>
      </div>
    </aside>
  </main>
  <script>
    const rows = {rows_payload};
    const exportFields = {fields_payload};
    const initialSummary = {summary_payload};
    const storageKey = {json.dumps(storage_key)};
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    const allowedLabels = new Set(["negative", "crack", "uncertain", ""]);
    rows.forEach(row => {{
      if (Object.prototype.hasOwnProperty.call(saved, row.candidate_path)) {{
        const savedLabel = saved[row.candidate_path];
        row.review_label = allowedLabels.has(savedLabel) ? savedLabel : "uncertain";
      }}
    }});
    let index = Math.min(
      Number(localStorage.getItem(storageKey + "-index") || 0),
      Math.max(0, rows.length - 1)
    );

    const image = document.getElementById("candidate-image");
    const filename = document.getElementById("filename");
    const metadata = document.getElementById("metadata");
    const split = document.getElementById("split");
    const progressFill = document.getElementById("progress-fill");
    const progressText = document.getElementById("progress-text");
    const summary = document.getElementById("summary");

    function persist() {{
      const labels = {{}};
      rows.forEach(row => {{
        if (row.review_label) labels[row.candidate_path] = row.review_label;
      }});
      localStorage.setItem(storageKey, JSON.stringify(labels));
      localStorage.setItem(storageKey + "-index", String(index));
    }}

    function counts() {{
      const result = {{
        reviewed: 0, negative: 0, crack: 0, uncertain: 0,
        bySplit: {{
          train: {{total: 0, negative: 0}},
          val: {{total: 0, negative: 0}},
          test: {{total: 0, negative: 0}}
        }}
      }};
      rows.forEach(row => {{
        if (row.review_label) {{
          result.reviewed += 1;
          result[row.review_label] += 1;
        }}
        if (result.bySplit[row.target_split]) {{
          result.bySplit[row.target_split].total += 1;
          if (row.review_label === "negative") result.bySplit[row.target_split].negative += 1;
        }}
      }});
      return result;
    }}

    function renderSummary(current) {{
      summary.innerHTML = `
        검수: <strong>${{current.reviewed}}</strong> / ${{rows.length}} ·
        정상 <strong>${{current.negative}}</strong> ·
        균열 ${{current.crack}} · 보류 ${{current.uncertain}}
        <table>
          <thead><tr><th>split</th><th>정상</th><th>후보</th></tr></thead>
          <tbody>
            ${{["train", "val", "test"].map(name =>
              `<tr><td>${{name}}</td><td>${{current.bySplit[name].negative}}</td><td>${{current.bySplit[name].total}}</td></tr>`
            ).join("")}}
          </tbody>
        </table>`;
    }}

    function render() {{
      if (!rows.length) {{
        filename.textContent = "검수 후보가 없습니다.";
        return;
      }}
      const row = rows[index];
      image.src = row.image_url;
      image.classList.toggle("missing", row.candidate_exists !== "true");
      filename.textContent = row.candidate_path.split("/").pop();
      split.textContent = `${{row.target_split}} · ${{index + 1}} / ${{rows.length}}`;
      metadata.textContent =
        `원본: ${{row.source_image_path}}\\n` +
        `좌표: (${{row.left}}, ${{row.top}}) – (${{row.right}}, ${{row.bottom}})` +
        (row.candidate_exists === "true" ? "" : "\\n후보 파일을 찾을 수 없습니다.");
      document.querySelectorAll("[data-label]").forEach(button => {{
        button.classList.toggle("active", button.dataset.label === row.review_label);
      }});
      const current = counts();
      const percent = rows.length ? current.reviewed / rows.length * 100 : 0;
      progressFill.style.width = `${{percent}}%`;
      progressText.textContent =
        `${{current.reviewed}} / ${{rows.length}} 검수 완료 ` +
        `(${{percent.toFixed(1)}}%)`;
      renderSummary(current);
      persist();
    }}

    function move(delta) {{
      index = (index + delta + rows.length) % rows.length;
      render();
    }}

    function setLabel(label) {{
      if (!rows.length) return;
      rows[index].review_label = label;
      persist();
      const next = rows.findIndex(
        (row, candidateIndex) => candidateIndex > index && !row.review_label
      );
      if (next >= 0) index = next;
      else if (index < rows.length - 1) index += 1;
      render();
    }}

    function nextUnreviewed() {{
      if (!rows.length) return;
      for (let offset = 1; offset <= rows.length; offset += 1) {{
        const candidate = (index + offset) % rows.length;
        if (!rows[candidate].review_label) {{
          index = candidate;
          render();
          return;
        }}
      }}
    }}

    function clearLabel() {{
      if (!rows.length) return;
      rows[index].review_label = "";
      render();
    }}

    function csvCell(value) {{
      let text = String(value ?? "");
      if (/^[=+\\-@\\t\\r]/.test(text)) text = "'" + text;
      return `"${{text.replaceAll('"', '""')}}"`;
    }}

    function exportCsv() {{
      const lines = [exportFields.map(csvCell).join(",")];
      rows.forEach(row => lines.push(
        exportFields.map(field => csvCell(row[field] || "")).join(",")
      ));
      const blob = new Blob(["\\ufeff", lines.join("\\r\\n")], {{type: "text/csv;charset=utf-8"}});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "negative_review_reviewed.csv";
      link.click();
      URL.revokeObjectURL(link.href);
    }}

    document.querySelectorAll("[data-label]").forEach(button =>
      button.addEventListener("click", () => setLabel(button.dataset.label))
    );
    document.getElementById("previous").addEventListener("click", () => move(-1));
    document.getElementById("next").addEventListener("click", () => move(1));
    document.getElementById("next-unreviewed").addEventListener("click", nextUnreviewed);
    document.getElementById("clear-label").addEventListener("click", clearLabel);
    document.getElementById("export").addEventListener("click", exportCsv);
    document.addEventListener("keydown", event => {{
      if (event.key.toLowerCase() === "n") setLabel("negative");
      else if (event.key.toLowerCase() === "c") setLabel("crack");
      else if (event.key.toLowerCase() === "u") setLabel("uncertain");
      else if (event.key === "ArrowLeft") move(-1);
      else if (event.key === "ArrowRight") move(1);
    }});
    void initialSummary;
    render();
  </script>
</body>
</html>
"""


__all__ = [
    "build_negative_review_ui",
    "create_review_server",
    "normalize_review_label",
    "serve_review_ui",
]
