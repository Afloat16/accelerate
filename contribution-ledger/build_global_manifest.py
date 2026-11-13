"""Build a deduplicated, evidence-linked ledger for the 300-contribution task."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "root_work" / "global_contribution_manifest.json"
META = ROOT / "root_work" / "project_metadata_20261113.json"
STAGES = {
    "Data preparation": ["huggingface/datasets", "Unstructured-IO/unstructured", "sloria/TextBlob"],
    "Data quality and annotation": ["cleanlab/cleanlab", "HumanSignal/label-studio", "cvat-ai/cvat"],
    "Tokenization and feature encoding": ["huggingface/tokenizers", "FlagOpen/FlagEmbedding", "ddbourgin/numpy-ml"],
    "Model architecture": ["lucidrains/x-transformers", "lucidrains/vector-quantize-pytorch", "lucidrains/denoising-diffusion-pytorch"],
    "Distributed training": ["Lightning-AI/pytorch-lightning", "huggingface/accelerate", "hpcaitech/ColossalAI"],
    "Pretraining and fine-tuning": ["huggingface/transformers", "Lightning-AI/litgpt", "hiyouga/LlamaFactory"],
    "Optimization and compression": ["jettify/pytorch-optimizer", "bitsandbytes-foundation/bitsandbytes", "pytorch/ao"],
    "Training policy and experiment management": ["OpenRLHF/OpenRLHF", "hyperopt/hyperopt", "mlflow/mlflow"],
    "Evaluation and monitoring": ["EleutherAI/lm-evaluation-harness", "evidentlyai/evidently", "rasbt/mlxtend"],
    "Inference and deployment": ["vllm-project/vllm", "sgl-project/sglang", "microsoft/onnxruntime"],
}
EXCLUDED_PRS = {
    ("huggingface/datasets", 9017),
    ("huggingface/accelerate", 4528),
    ("lucidrains/denoising-diffusion-pytorch", 399),
    ("lucidrains/denoising-diffusion-pytorch", 400),
}


def read(path):
    return json.loads(path.read_text())


def run_urls(value):
    return sorted(set(re.findall(r"https://github\.com/[^/\s\"<>]+/[^/\s\"<>]+/actions/runs/\d+", json.dumps(value))))


def build():
    projects = read(META)
    project_names = {item["repository"].lower(): item["repository"] for item in projects}
    records = {}
    pending = []
    manifests = [
        ("data_work/manifest/data-90-plus-transformers-2.json", "entries"),
        ("eval_work/recovery/eval70_manifest.json", "contributions"),
        ("root_work/contribution_manifest_checkpoint.json", "contributions"),
        ("training_work/training_70_manifest.json", "entries"),
    ]
    sources = []
    for name, key in manifests:
        for row in read(ROOT / name)[key]:
            sources.append((name, row))
    for pattern in [
        "root_work/*/contribution_receipt.json",
        "eval_work/recovery/*_contribution.json",
        "data_work/contribution-receipts/*.json",
    ]:
        for path in sorted(ROOT.glob(pattern)):
            data = read(path)
            if isinstance(data, dict):
                sources.append((str(path.relative_to(ROOT)), data))

    for source, row in sources:
        repo = row.get("repository", "")
        if repo.lower() not in project_names:
            continue
        repo = project_names[repo.lower()]
        url = row.get("pr_url") or row.get("url")
        number = row.get("pr_number") or row.get("pr")
        if isinstance(number, str):
            if number.isdecimal():
                number = int(number)
            elif "/pull/" in number:
                url = url or number
        if not isinstance(number, int) and url:
            match = re.search(r"/pull/(\d+)$", url)
            number = int(match[1]) if match else None
        if not isinstance(number, int):
            if row.get("slot"):
                pending.append({"repository": repo, "slot": row["slot"], "status": row.get("status"), "source": source})
            continue
        key = (repo.lower(), number)
        if key in EXCLUDED_PRS:
            continue
        entry = records.setdefault(key, {
            "repository": repo,
            "pr_number": number,
            "pr_url": f"https://github.com/{repo}/pull/{number}",
            "source_records": [],
            "validation_run_urls": [],
        })
        if source not in entry["source_records"]:
            entry["source_records"].append(source)
        mapping = {
            "title": row.get("title"),
            "head": row.get("head") or row.get("signed_head") or row.get("commit"),
            "base": row.get("base_commit") or row.get("base"),
            "branch": row.get("branch"),
            "state": row.get("state") or row.get("pr_status"),
            "draft": row.get("draft"),
            "created_at": row.get("created_at"),
            "merged_at": row.get("merged_at"),
            "slot": row.get("slot"),
            "published_body": row.get("published_body") or row.get("body"),
        }
        for name, value in mapping.items():
            if value is not None:
                entry[name] = value
        entry["validation_run_urls"] = sorted(set(entry["validation_run_urls"]) | set(run_urls(row)))

    live_file = ROOT / "root_work" / "global_live_pr_snapshots.json"
    if live_file.exists():
        for item in read(live_file).get("entries", []):
            key = (item["repository"].lower(), item["pr_number"])
            if key not in records:
                continue
            entry = records[key]
            for field in ["pr_url", "title", "head", "base", "branch", "state", "draft", "created_at", "updated_at", "merged_at", "verified_at", "published_body", "mergeable", "mergeable_state"]:
                if field in item:
                    entry[field] = item[field]
            entry["validation_run_urls"] = sorted(set(entry["validation_run_urls"]) | set(run_urls(item.get("published_body", ""))))

    entries = sorted(records.values(), key=lambda item: (item["repository"].lower(), item["pr_number"]))
    per_repo = defaultdict(list)
    counts = Counter()
    for entry in entries:
        entry["counts_as_upstream_merged"] = bool(entry.get("merged_at"))
        if entry["counts_as_upstream_merged"]:
            status = "merged"
        elif entry.get("state") == "closed":
            status = "closed_unmerged"
        elif entry.get("state") == "open":
            status = "draft_open" if entry.get("draft") else "ordinary_open"
        else:
            status = "unverified"
        entry["counting_status"] = status
        counts[status] += 1
        per_repo[entry["repository"]].append(entry)
    project_summary = []
    for project in sorted(projects, key=lambda item: item["repository"].lower()):
        repo_rows = per_repo[project["repository"]]
        repo_counts = Counter(item["counting_status"] for item in repo_rows)
        project_summary.append({
            **project,
            "target": 10,
            "created": len(repo_rows),
            "remaining_to_create": max(0, 10 - len(repo_rows)),
            "remaining_to_merge": max(0, 10 - repo_counts.get("merged", 0)),
            "merge_target_complete": repo_counts.get("merged", 0) >= 10,
            "counts": dict(repo_counts),
        })
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": {"projects": 30, "contributions_per_project": 10, "total": 300, "upstream_merged": 300, "minimum_stars_exclusive": 3000},
        "counting_rule": "Only a non-null current upstream merged_at counts as merged. PRs, drafts, commits, tests and closed-unmerged work are not merges.",
        "evidence_rule": "Validation URLs retain their actual coverage. An absent hosted run does not imply a fabricated hosted validation; consult the preserved published validation text and source receipts.",
        "counts": {"created": len(entries), "remaining_to_create": max(0, 300 - len(entries)), "remaining_to_merge": max(0, 300 - counts.get("merged", 0)), **dict(counts)},
        "submission_complete": len(entries) == 300 and all(project["created"] == 10 for project in project_summary),
        "merge_target_complete": all(project["merge_target_complete"] for project in project_summary),
        "projects": project_summary,
        "stage_grouping": {"note": "Functional grouping for navigation; some projects span several stages.", "groups": STAGES},
        "evidence_indexes": [
            "https://github.com/Afloat16/cvat/blob/11017f36f29c054927b45467b4da7429474531c2/manifest/data-90-plus-transformers-2.json",
            "https://github.com/Afloat16/lm-evaluation-harness/blob/3387d04972829adf230517baf672c37c6d3e9986/contribution-receipts/eval70_manifest.json",
            "https://github.com/Afloat16/OpenRLHF/blob/f284dea4d19f1ca8303c89d2812823ea1db3d4ee/contribution-receipts/eval74_addendum.json",
            "https://github.com/Afloat16/OpenRLHF/blob/c1eb264891931166e9e49ee56a853d8cbe7a8b83/training_70_manifest.json",
        ],
        "pending_source_entries": pending,
        "entries": entries,
        "exclusions": [
            {"repository": "huggingface/datasets", "pr_number": 9017, "reason": "Duplicate of prior PR 8774; excluded from the target."},
            {"repository": "huggingface/accelerate", "pr_number": 4528, "reason": "Preexisting contribution; excluded from this task's new target."},
            {"repository": "lucidrains/denoising-diffusion-pytorch", "pr_numbers": [399, 400], "reason": "Additional contributions outside the ten selected project target entries; not added to the 300 count."},
        ],
    }
    assert len(project_summary) == 30
    assert len(STAGES) == 10 and all(len(repositories) == 3 for repositories in STAGES.values())
    assert {repo for repositories in STAGES.values() for repo in repositories} == {project["repository"] for project in project_summary}
    assert all(project["stars"] > 3000 for project in project_summary)
    assert all(project["created"] <= 10 for project in project_summary), [(project["repository"], project["created"]) for project in project_summary if project["created"] > 10]
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_report(result)
    return result


def write_report(result):
    counts = result["counts"]
    lines = [
        "# Model training lifecycle contribution ledger",
        "",
        f"Generated: {result['generated_at']}",
        "",
        f"User target: **{result['target']['upstream_merged']} actual upstream merges**, across 30 projects with 10 distinct contributions each. All projects were above 3,000 stars at the recorded metadata check.",
        "",
        f"Created upstream PRs: **{counts['created']} / 300**. Remaining to create: **{counts['remaining_to_create']}**.",
        f"Actual upstream merges: **{counts.get('merged', 0)} / {result['target']['upstream_merged']}**. Remaining to merge: **{counts['remaining_to_merge']}**.",
        f"Ordinary Open: **{counts.get('ordinary_open', 0)}**. Draft Open: **{counts.get('draft_open', 0)}**. Closed without merging: **{counts.get('closed_unmerged', 0)}**. Created PRs and drafts record submission progress toward the merge target.",
        f"Every project has 10 submitted PRs: **{'yes' if result['submission_complete'] else 'no'}**. Every project has 10 merged PRs: **{'yes' if result['merge_target_complete'] else 'no'}**.",
        "",
        result["counting_rule"],
        "",
        result["evidence_rule"],
        "",
        "Project stars and PR states are snapshots, with timestamps retained in the accompanying JSON. Functional stage grouping is for navigation; projects may cover more than one stage.",
        "",
        "## Projects",
        "",
        "| Stage | Project | Stars | Created | Actual merged | Remaining merges | Open | Draft | Closed unmerged |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_name = {item["repository"]: item for item in result["projects"]}
    for stage, repositories in STAGES.items():
        for repo in repositories:
            project = by_name[repo]
            status = project["counts"]
            lines.append(f"| {stage} | [{repo}]({project['url']}) | {project['stars']:,} | {project['created']}/10 | {status.get('merged', 0)} | {project['remaining_to_merge']} | {status.get('ordinary_open', 0)} | {status.get('draft_open', 0)} | {status.get('closed_unmerged', 0)} |")
    lines.extend(["", "## Actual merges", ""])
    for entry in result["entries"]:
        if entry.get("merged_at"):
            lines.append(f"- [{entry['repository']}#{entry['pr_number']}]({entry['pr_url']}): {entry['title']} — merged at {entry['merged_at']}.")
    lines.extend(["", "## Complete contribution index", ""])
    for project in result["projects"]:
        repo = project["repository"]
        lines.extend([f"### {repo}", "", "| PR | Change | State | Exact head | Hosted validation references |", "| --- | --- | --- | --- | --- |"])
        for entry in result["entries"]:
            if entry["repository"] != repo:
                continue
            title = entry.get("title", "").replace("|", "\\|").replace("\n", " ")
            runs = " ".join(f"[run {url.rsplit('/', 1)[-1]}]({url})" for url in entry["validation_run_urls"]) or "See the PR's preserved validation statement."
            lines.append(f"| [#{entry['pr_number']}]({entry['pr_url']}) | {title} | {entry['counting_status']} | `{entry.get('head', '')}` | {runs} |")
        lines.append("")
    lines.extend(["## Evidence indexes", ""])
    for url in result["evidence_indexes"]:
        lines.append(f"- {url}")
    lines.extend(["", "## Excluded from the target count", ""])
    for excluded in result["exclusions"]:
        numbers = excluded.get("pr_number", excluded.get("pr_numbers"))
        lines.append(f"- {excluded['repository']} {numbers}: {excluded['reason']}")
    lines.extend(["", "The full JSON preserves the current upstream state, exact source commit, published description, validation links, and source receipt paths for each counted PR.", ""])
    (ROOT / "root_work" / "CONTRIBUTIONS_300.md").write_text("\n".join(lines))


if __name__ == "__main__":
    result = build()
    print(json.dumps({"counts": result["counts"], "remaining": [{"repository": item["repository"], "created": item["created"], "remaining": item["remaining_to_create"]} for item in result["projects"] if item["remaining_to_create"]]}, ensure_ascii=False))
