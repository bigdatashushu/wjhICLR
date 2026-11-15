import json
import pathlib
import re
import collections

TASK = pathlib.Path(__file__).resolve().parents[1]
BASE = pathlib.Path("/Users/bytedance/wangjh")
CANDIDATES = json.loads((TASK / "source/initial-candidates.json").read_text())["wins"]

def read(p):
    return json.loads(pathlib.Path(p).read_text())

def small(v):
    if isinstance(v, str):
        if v.startswith("data:image/"):
            return f"[原图内嵌，{len(v)} 字符]"
        return re.sub(r"data:image/[^\"\\\s]+", "[原图内嵌]", v)
    if isinstance(v, list):
        return [small(x) for x in v]
    if isinstance(v, dict):
        return {k: small(x) for k, x in v.items() if k not in ("native_payload", "generation")}
    return v

def gpt():
    out = []
    used = set()
    for c in CANDIDATES:
        if c["family"] not in ("BLINK", "CV3D") or not (c["method_has_answer"] and c["direct_has_answer"]):
            continue
        qid, ds = c["qid"], c["family"]
        if (ds, qid) in used:
            continue
        scorefile = pathlib.Path(c["source"])
        root = scorefile.parent.parent
        row = c["row"]
        pair = row["pair_identity"]
        ep = root / "episodes" / row["dataset"] / pair
        if not ep.is_dir():
            continue
        used.add((ds, qid))
        item = {"dataset": ds, "qid": qid, "score_source": str(scorefile), "root": str(ep), "arms": {}}
        for arm in ("harness", "direct"):
            p = ep / arm
            if not (p / "episode.json").exists():
                continue
            e = read(p / "episode.json")
            reqs = []
            for f in sorted((p / "requests").glob("*.json")):
                if f.name.endswith(".intent.json") or f.name.endswith(".response.json") or f.name.endswith(".execution.json"):
                    reqs.append({"file": f.name, "data": small(read(f))})
            item["arms"][arm] = {"result": small(e["result"]), "requests": reqs}
        out.append(item)
    (TASK / "review/gpt-candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    for x in out:
        h = x["arms"].get("harness", {}).get("result", {})
        print(x["dataset"], x["qid"], h.get("attempted_tool_calls"), h.get("succeeded_tool_calls"), h.get("basis"))

def omni():
    out = []
    for run in ["omni3d-rawdepth-vs-direct20-20261103-r04", "omni3d-rawdepth-vs-direct20-20261103-r05", "omni3d-repaired15-vs-direct-20261104-r01"]:
        root = BASE / "outputs/harness3d_four_datasets_20261031" / run
        scores = read(root / "scores.json")["scored"]
        groups = collections.defaultdict(dict)
        for r in scores:
            groups[r["sample_id"]][r["arm"]] = r
        inputfile = root / "inputs.public.json"
        if not inputfile.exists():
            inputfile = root / "inputs.json"
        inputs = {r["sample_id"]: r for r in read(inputfile)}
        for q, arms in groups.items():
            h, d = arms.get("skill", {}), arms.get("direct", {})
            if h.get("score") != 1 or d.get("score", 1) >= 1 or not d.get("legal"):
                continue
            item = {"dataset": "Omni3D", "qid": q, "root": str(root), "input": inputs.get(q), "scores": arms, "arms": {}}
            for arm in ("skill", "direct"):
                ep = root / "episodes" / f"{q}-{arm}"
                reqs = []
                for f in sorted(ep.glob("*.json")):
                    if any(t in f.name for t in (".messages.", ".response.", ".execution.")):
                        reqs.append({"file": f.name, "data": small(read(f))})
                item["arms"][arm] = reqs
            out.append(item)
    (TASK / "review/omni-candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    for x in out:
        print("Omni3D", x["qid"], x["scores"]["direct"]["score"], x["scores"]["skill"].get("tool_calls"), x["root"].split("/")[-1])

def vsi():
    out = []
    pairfile = BASE / "VSI相对方向33题DeepSeek官方MaxLow对比-20261113.json"
    pairs = read(pairfile)["pairs"]
    for r in pairs:
        if r["system_score"] != 1 or r["direct_score"] != 0:
            continue
        root = BASE / "wjhICLR/VSIQwenFlashRunner/runs" / r["selected_system_run"]
        for f in (root / "system/variants").glob("*/outcome.json"):
            o = read(f)
            if str(o.get("qa_id")) != str(r["qa_id"]):
                continue
            item = {"dataset": "VSIbench", "qid": str(r["qa_id"]), "pair": small(r), "root": str(root), "outcome": str(f),
                    "rounds": small(o.get("rounds")), "program_trace": small(o.get("program_trace")),
                    "episode_trace": small(o.get("episode_trace")), "answer_payload": o.get("answer_payload")}
            out.append(item)
            break
    (TASK / "review/vsi-candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    for x in out:
        print("VSIbench", x["qid"], "rounds", type(x["rounds"]).__name__, "program_trace", type(x["program_trace"]).__name__)

if __name__ == "__main__":
    gpt()
    omni()
    vsi()
