import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "f5cc53877677eb7fa0ef7091fe41536a777fee6a"
HEAD = "a95ee5bdda6c74e93e59878a5e5e5db69d554950"
SOURCE = "src/accelerate/data_loader.py"
TEST = "tests/test_dispatcher_prefix_storage.py"

def run(label, tests):
    output = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests, f"--junitxml={label}.xml"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    Path(f"{label}.log").write_text(output.stdout)
    print(output.stdout, flush=True)
    root = ElementTree.parse(f"{label}.xml").getroot()
    counts = {key: sum(int(suite.get(key, 0)) for suite in root.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    counts["returncode"] = output.returncode
    print(label, counts, flush=True)
    return counts

def storage(label):
    result = subprocess.check_output([sys.executable, "validation/measure_dispatcher_prefix_storage.py"], text=True)
    Path(f"{label}-storage.json").write_text(result)
    print(label, result, flush=True)
    return json.loads(result.strip().splitlines()[-1])

for path in (SOURCE, TEST):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
fixed = Path(SOURCE).read_bytes()
try:
    Path(SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before = run("before", [TEST])
    before_storage = storage("before")
finally:
    Path(SOURCE).write_bytes(fixed)
after = run("after", [TEST])
after_storage = storage("after")
assert before == {"tests": 32, "failures": 12, "errors": 0, "skipped": 0, "returncode": 1}, before
assert after == {"tests": 32, "failures": 0, "errors": 0, "skipped": 0, "returncode": 0}, after
assert before_storage["cached_storage_bytes"] == 256 * 2048 * 4, before_storage
assert after_storage["cached_storage_bytes"] == 2048 * 4, after_storage
existing = run("existing", ["tests/test_data_loader.py"])
assert existing["returncode"] == existing["failures"] == existing["errors"] == 0, existing
for path in (SOURCE, TEST):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
report = {"base": BASE, "head": HEAD, "before": before, "after": after, "existing": existing,
          "before_storage": before_storage, "after_storage": after_storage}
Path("native-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
