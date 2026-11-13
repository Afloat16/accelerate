import json
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

base = "3bd795f7cfb763b994b4ba8ba59e5a305a7c528a"
source = pathlib.Path("src/accelerate/accelerator.py")
patched = source.read_bytes()
test = "tests/test_join_cpu.py"

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
    source.write_bytes(subprocess.check_output(["git", "show", base + ":" + str(source)]))
    before = run("before", [test])
finally:
    source.write_bytes(patched)
after = run("after", [test])
existing = run("existing", ["tests/test_accelerator.py", "-k", "cpu_flag_prepare or can_be_reinstantiated or prepared_objects_are_referenced or free_memory_dereferences or mutable_states or accelerator_none or is_accelerator_prepared"])
report = {"base": base, "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "before": before, "after": after, "existing": existing}
pathlib.Path("native-report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert before["failures"] == 8 and before["errors"] == 0 and before["tests"] == 9
assert after["returncode"] == 0 and after["tests"] == 9 and after["skipped"] == 0
assert existing["returncode"] == 0
