import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "f5cc53877677eb7fa0ef7091fe41536a777fee6a"
HEAD = "133b0409368afe8dc1751094d462efb4823176e6"
SOURCE = "src/accelerate/utils/operations.py"
TEST = "tests/test_broadcast_strided.py"

def run(label, tests):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests, f"--junitxml={label}.xml"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    root = ElementTree.parse(f"{label}.xml").getroot()
    counts = {key: sum(int(suite.get(key, 0)) for suite in root.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    counts["returncode"] = result.returncode
    print(label, counts, flush=True)
    return counts

for path in (SOURCE, TEST):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
fixed = Path(SOURCE).read_bytes()
try:
    Path(SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before = run("before", [TEST])
finally:
    Path(SOURCE).write_bytes(fixed)
after = run("after", [TEST])
existing = run("existing", ["tests/test_utils.py", "tests/test_data_loader.py"])
ops = subprocess.run(
    [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
     "-m", "accelerate.test_utils.scripts.test_ops"],
    env={**os.environ, "ACCELERATE_USE_CPU": "true"}, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
Path("existing-distributed-ops.log").write_text(ops.stdout)
print(ops.stdout, flush=True)
report = {"base": BASE, "head": HEAD, "before": before, "after": after,
          "existing": existing, "existing_distributed_ops_returncode": ops.returncode}
Path("native-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
assert before["returncode"] == 1 and before["failures"] > 0 and before["errors"] == 0, before
assert after == {"tests": 35, "failures": 0, "errors": 0, "skipped": 0, "returncode": 0}, after
assert existing["returncode"] == existing["failures"] == existing["errors"] == 0, existing
assert ops.returncode == 0
for path in (SOURCE, TEST):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
