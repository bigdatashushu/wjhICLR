"""Score the complete retest, audit delivery, and close the remaining gateway."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT.parents[1] / "scripts"
COMPLETION = ROOT.parent / "rel-direction20-s06v110-storage-completion-c32-20261111-r18"
DIRECT = ROOT.parent / "rel-direction20-g1m0-c32-20261111-r14"
REPORT = Path("/Users/bytedance/wangjh/VSI相对方向20题S06新版复测-20261111.md")
sys.path.insert(0, str(SCRIPTS))
from control_small_comparison import remote, API_PYTHON


def run(path, *args):
    subprocess.run([sys.executable, str(path), *map(str, args)], check=True)


def main():
    run(SCRIPTS / "score_small_comparison.py", "--run-root", ROOT,
        "--report", REPORT, "--direct-run-root", DIRECT,
        "--system-completion-root", COMPLETION)
    run(ROOT / "audit_revision.py")
    run(SCRIPTS / "control_small_comparison.py", "cleanup", "--run-root", COMPLETION,
        "--gateway-root", "/home/cvailab/wangjh/harness3d/vsi-direction20-s06v110-gateway-20261111-r18")
    receipts = {}
    for root, port in ((ROOT, 27058), (COMPLETION, 27062)):
        cleanup = json.loads((root / "operational_cleanup.json").read_text())
        program = f"""
import json,socket,time
from pathlib import Path
pid={cleanup['pid']};port={port}
s=socket.socket();s.settimeout(2)
code=s.connect_ex(('127.0.0.1',port));s.close()
assert code!=0
assert not Path(f'/proc/{{pid}}').exists()
print(json.dumps({{'status':'closed','pid':pid,'port':port,'process_exited':True,
 'port_closed':True,'observed_at':time.time()}}))
"""
        value = json.loads(remote("cvailab@10.10.0.198",
                                  API_PYTHON + " -B -c " + shlex.quote(program)))
        (root / "gateway_closed_receipt.json").write_text(json.dumps(value, indent=2) + "\n")
        receipts[root.name] = value
    result = json.loads((ROOT / "comparison.json").read_text())
    result["summary"]["operational_cleanup"] = receipts
    for path in (ROOT / "comparison.json", REPORT.with_suffix(".json")):
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with REPORT.open("a") as handle:
        handle.write("\n两批网关均在 active=0 时核验 PID 和命令身份后停止，"
                     "进程退出及 27058/27062 端口关闭均已确认。"
                     "原批次的失败日志、未更新 summary、只读网关数据库备份和旁路核对账本保留在 r16；"
                     "完整 3 题补跑与零调用 preflight 回执保留在 r18。\n")
    s = result["summary"]
    print(json.dumps({
        "status": s["status"], "system_score": s["system_score"], "direct_score": s["direct_score"],
        "known_tokens": s["token_usage"]["system"]["total_tokens"],
        "change_vs_previous": s["revision_retest"]["score_delta"],
        "improved": s["revision_retest"]["improved_qa_ids"],
        "decreased": s["revision_retest"]["decreased_qa_ids"],
        "report": str(REPORT), "gateways_closed": len(receipts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
