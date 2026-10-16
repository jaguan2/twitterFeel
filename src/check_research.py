"""Check real import → both models → results → findings in an isolated workspace."""
import json
from pathlib import Path
import tempfile
import time

from analysis_engine import AnalysisEngine
from desktop_api import DesktopAPI
from workbench_data import REPO_ROOT


def main():
    with tempfile.TemporaryDirectory(prefix="twitterfeel-research-") as temporary:
        root = Path(temporary)
        (root / "app_version.json").write_bytes((REPO_ROOT / "app_version.json").read_bytes())
        api = DesktopAPI(root)
        api._engine = AnalysisEngine(REPO_ROOT)

        def call(action, **payload):
            response = api.call(action, payload)
            if not response["ok"]:
                raise RuntimeError(action + ": " + response["error"])
            return response["data"]

        draft = call("preview_import", name="Synthetic workflow diagnostic", format="txt",
            content="I love this product, it is wonderful!\nThis is terrible and I am furious.\nThe meeting starts at three.\n\nI'm excited about moving, but scared to leave everyone behind.")
        dataset = call("import_dataset", token=draft["token"], name="Synthetic workflow diagnostic", text_column="text")
        job = call("start_analysis", dataset_id=dataset["id"], include_sentiment=True, name="Synthetic workflow diagnostic")
        deadline = time.monotonic() + 180
        while job["status"] == "running" and time.monotonic() < deadline:
            time.sleep(.2)
            job = call("analysis_status", job_id=job["id"])
        if job["status"] != "complete":
            raise RuntimeError(str(job))
        run = call("load_run", run_id=job["run_id"])
        assert len(run["posts"]) == 5
        assert run["posts"][3]["status"] == "invalid"
        assert [p["sentiment"]["label"] for p in run["posts"][:3]] == ["positive", "negative", "neutral"]
        call("save_finding", run_id=job["run_id"], author="Automated diagnostic", title="Execution check", note="Synthetic examples test execution, not model accuracy.")
        findings = call("findings", run_id=job["run_id"])
        exported = call("export_findings", run_id=job["run_id"])
        assert Path(exported["path"]).exists() and len(findings["notes"]) == 1
        report = {"success": True, "kind": "synthetic_execution_check", "app_version": run["config"]["app_version"],
                  "summary": run["summary"], "models": run["config"]["provenance"]["models"],
                  "examples": [{k: post[k] for k in ("text", "labels", "status", "sentiment")} for post in run["posts"]],
                  "checks": ["versioned import", "real emotion and sentiment inference", "blank input retained as invalid",
                             "persisted result integrity", "generated observations", "authored observations", "report export"],
                  "limitation": "Synthetic checks do not measure labeling accuracy."}
        path = REPO_ROOT / "runs/workbench/research-workflow-check.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"success": True, "report": str(path), "summary": run["summary"]}))


if __name__ == "__main__":
    main()
