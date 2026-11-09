from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-5.6-terra"
BASE_URL = "https://xcode.best/v1"
COUNTS = {"mv:direct": 93, "mv:harness": 93, "sr:direct": 100,
          "sr:harness": 100, "mv:official": 133, "sr:official": 143}
PUBLIC_KEYS = {
    "sample_id", "idx", "split", "subset", "row_index", "sub_task", "question",
    "choices", "options", "benchmark_prompt", "images", "reference_answer_visible",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, once=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("x") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if once:
        try:
            os.link(temp, path)
        finally:
            temp.unlink()
    else:
        os.replace(temp, path)


def read_keys(path):
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError("credential_file_requires_private_permissions")
    keys = []
    for line in path.read_text().splitlines():
        line = line.strip()
        assignment = re.fullmatch(r"(?:export\s+)?XCODE_GPT_API_KEY\s*=\s*(.*?)\s*", line)
        comment = re.fullmatch(r"#\s*(sk-[A-Za-z0-9_-]+)\s*", line)
        key = (assignment.group(1).strip("'\"") if assignment
               else comment.group(1) if comment else None)
        if key and key not in keys:
            keys.append(key)
    if not keys or any(any(c.isspace() for c in key) for key in keys):
        raise ValueError("missing_or_invalid_credentials")
    return {f"pool-{i + 1:02d}": key for i, key in enumerate(keys)}


def assert_public(value, keys=()):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if any(key and key in raw for key in keys):
        raise ValueError("credential_in_public_record")


def source_files():
    return {str(p.relative_to(ROOT)): file_digest(p)
            for p in sorted(ROOT.rglob("*")) if p.is_file()
            and p.suffix in {".py", ".txt", ".json"}
            and p.relative_to(ROOT).parts[0] not in {"runs", "deployment", "__pycache__"}}


def parse_official(text, options):
    """Label-only extraction independent of references; ambiguous prose is invalid."""
    text = text.strip()
    if not text:
        return None, "empty"
    labels = {o["label"] for o in options}
    texts = {o["text"].strip().casefold(): o["label"] for o in options}
    clean = re.sub(r"[*`]", "", text.splitlines()[-1]).strip()
    clean = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?::|is)\s*", "",
                   clean, flags=re.I).strip().rstrip(".!")
    if clean.casefold() in texts:
        return texts[clean.casefold()], "final_option_text"
    match = re.fullmatch(r"(?:\(([AB])\)|([AB])[.)]?)(?:\s+(.+))?", clean, flags=re.I)
    if match:
        label = (match[1] or match[2]).upper()
        suffix = (match[3] or "").strip().rstrip(".!").casefold()
        if label in labels and (not suffix or texts.get(suffix) == label):
            return label, "final_option_line"
    # Explicit provider JSON is valid even though it was not requested.
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        obj = json.loads(candidate)
        answer = obj.get("answer") if isinstance(obj, dict) else None
        if isinstance(answer, str):
            answer = answer.strip()
            if answer.upper() in labels:
                return answer.upper(), "explicit_json"
            if answer.casefold() in texts:
                return texts[answer.casefold()], "explicit_json_text"
    except ValueError:
        pass
    # A single parenthesized label at the beginning is the BLINK choice form.
    match = re.match(r"^\s*\(([AB])\)", text)
    if match and len(set(re.findall(r"\(([AB])\)", text))) == 1:
        return match[1], "leading_parenthesized_option"
    return None, "invalid"
