"""Extract Ubuntu's Vulkan inspection tool inside the task, without system writes."""
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from bridge import simulator_environment
from protocol import ROOT, canonical, sha256

download_dir = ROOT / "cache/vulkan-tools"
runtime = ROOT / "runtime/vulkan-tools"
download_dir.mkdir(parents=True, exist_ok=True)
runtime.mkdir(parents=True, exist_ok=True)
environment = simulator_environment(ROOT)
subprocess.run([
    "apt-get", "-o", "Acquire::ForceIPv4=true", "-o", "Acquire::Retries=0",
    "download", "vulkan-tools=1.2.131.1+dfsg1-1",
], cwd=download_dir, env=environment, check=True)
packages = list(download_dir.glob("*.deb"))
if len(packages) != 1:
    raise RuntimeError("unexpected Vulkan package count")
subprocess.run(["dpkg-deb", "-x", str(packages[0]), str(runtime)],
               cwd=ROOT, env=environment, check=True)
print(canonical({"status": "ready", "package_sha256": sha256(packages[0].read_bytes()),
                 "binary_sha256": sha256((runtime / "usr/bin/vulkaninfo").read_bytes())}).decode())
