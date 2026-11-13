import json
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

base = "3bd795f7cfb763b994b4ba8ba59e5a305a7c528a"
sources = ["src/accelerate/hooks.py", "src/accelerate/utils/modeling.py"]
patched = {p: pathlib.Path(p).read_bytes() for p in sources}
test = "tests/test_direct_tensor_dispatch.py"

def run(label, paths):
    result = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--junitxml=" + label + ".xml"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    pathlib.Path(label + ".log").write_text(result.stdout)
    print(result.stdout, flush=True)
    root = ET.parse(label + ".xml").getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    counts["returncode"] = result.returncode
    print(label, counts, flush=True)
    return counts

try:
    for path in sources:
        pathlib.Path(path).write_bytes(subprocess.check_output(["git", "show", base + ":" + path]))
    before = run("before", [test])
finally:
    for path, data in patched.items():
        pathlib.Path(path).write_bytes(data)
after = run("after", [test])
existing = run("existing", [
    "tests/test_big_modeling.py::BigModelingTester::test_cpu_offload",
    "tests/test_big_modeling.py::BigModelingTester::test_cpu_offload_with_unused_submodules",
    "tests/test_big_modeling.py::BigModelingTester::test_disk_offload",
    "tests/test_big_modeling.py::BigModelingTester::test_disk_offload_with_unused_submodules",
    "tests/test_big_modeling.py::BigModelingTester::test_dispatch_model_disk_only",
    "tests/test_big_modeling.py::BigModelingTester::test_dispatch_model_force_hooks",
    "tests/test_modeling_utils.py::ModelingUtilsTester::test_check_device_map",
    "tests/test_modeling_utils.py::ModelingUtilsTester::test_check_device_map_invalid_keys",
    "tests/test_hooks.py",
])
report = {"base": base, "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "before": before, "after": after, "existing": existing}
pathlib.Path("native-report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert before["failures"] == 19 and before["errors"] == 0 and before["tests"] == 22
assert after["returncode"] == 0 and after["tests"] == 22 and after["skipped"] == 0
assert existing["returncode"] == 0
