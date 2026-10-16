"""Native WebView2 desktop entry: inline assets and a direct Python JS bridge."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import threading
import time

from desktop_api import DesktopAPI

ROOT = Path(__file__).resolve().parents[1]


def _process_tree():
    """Read only our native process family, including WebView2 renderers."""
    class Entry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                    ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                    ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                    ("flags", wintypes.DWORD), ("name", wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise OSError("Unable to inspect desktop process family")
    entries = []
    try:
        entry = Entry()
        entry.size = ctypes.sizeof(entry)
        valid = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while valid:
            entries.append((entry.pid, entry.parent))
            valid = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    family = {os.getpid()}
    while True:
        expanded = family | {pid for pid, parent in entries if parent in family}
        if expanded == family:
            return family
        family = expanded


def _listeners():
    family = _process_tree()
    result = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True,
                            text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    listeners = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[-2] == "LISTENING" and int(parts[-1]) in family:
            listeners.append({"address": parts[1], "pid": int(parts[-1])})
    return listeners


def _capture_preview(window, path):
    """Capture this app's rendered content via its own WebView2 SDK."""
    from System import Action
    from System.IO import FileAccess, FileMode, FileStream
    from System.Windows.Forms import FormWindowState
    from Microsoft.Web.WebView2.Core import CoreWebView2CapturePreviewImageFormat
    stream = FileStream(str(path), FileMode.Create, FileAccess.Write)
    holder = {}
    try:
        def restore_for_render():
            holder["state"] = window.native.WindowState
            window.native.WindowState = FormWindowState.Normal
        window.native.Invoke(Action(restore_for_render))
        # Let brief UI transitions finish before recording a design preview.
        time.sleep(.25)
        window.evaluate_js("window.__captureReady=false;requestAnimationFrame(()=>requestAnimationFrame(()=>window.__captureReady=true))")
        until = time.monotonic() + 5
        while not window.evaluate_js("window.__captureReady"):
            if time.monotonic() > until:
                raise RuntimeError("App did not paint a frame for render capture")
            time.sleep(.05)
        def capture():
            holder["task"] = window.native.browser.webview.CoreWebView2.CapturePreviewAsync(
                CoreWebView2CapturePreviewImageFormat.Png, stream)
        window.native.Invoke(Action(capture))
        if not holder["task"].Wait(15000):
            raise RuntimeError("App render capture timed out")
    finally:
        stream.Dispose()
        if "state" in holder:
            window.native.Invoke(Action(lambda: setattr(window.native, "WindowState", holder["state"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", type=Path)
    args = parser.parse_args()
    report = {"success": False, "project": str(ROOT), "mode": "native_webview2",
              "http_server": False, "ui_initialized": False, "bridge_ready": False,
              "pid": os.getpid()}
    smoke_done = threading.Event()
    try:
        import webview
        assets = ROOT / "src/desktop_ui"
        html = (assets / "index.html").read_text(encoding="utf-8")
        html = html.replace("__APP_CSS__", (assets / "app.css").read_text(encoding="utf-8"))
        html = html.replace("__APP_JS__", (assets / "app.js").read_text(encoding="utf-8"))
        if args.smoke_test:
            html = html.replace("<head>", "<head><script>window.__desktop_errors=[];window.addEventListener('error',e=>window.__desktop_errors.push(String(e.message)));window.addEventListener('unhandledrejection',e=>window.__desktop_errors.push(String(e.reason)));</script>")
        api = DesktopAPI(ROOT)
        smoke_workspace = None
        if args.smoke_test:
            # Exercise writes in disposable data, never the researcher's library.
            from analysis_engine import AnalysisEngine
            from workbench_data import discover_runs
            smoke_workspace = tempfile.TemporaryDirectory(prefix="twitterfeel-native-check-")
            fixture = Path(smoke_workspace.name)
            shutil.copyfile(ROOT / "app_version.json", fixture / "app_version.json")
            (fixture / "src").mkdir()
            shutil.copyfile(ROOT / "src/annotation_rubric.json", fixture / "src/annotation_rubric.json")
            source_run = discover_runs(ROOT)[0]["path"]
            target_run = fixture / "runs/fixture_analysis"
            target_run.mkdir(parents=True)
            for name in ("config.json", "summary.json", "predictions.jsonl"):
                shutil.copyfile(source_run / name, target_run / name)
            for public_report in (ROOT / "runs/public_benchmarks").glob("**/benchmark.json"):
                destination = fixture / public_report.parent.relative_to(ROOT)
                destination.mkdir(parents=True)
                for name in ("config.json", "summary.json", "predictions.jsonl", "benchmark.json"):
                    shutil.copyfile(public_report.parent / name, destination / name)
            for profile in api._calibration_profiles():
                source = ROOT / profile["id"]
                destination = fixture / source.parent.relative_to(ROOT)
                destination.mkdir(parents=True, exist_ok=True)
                for name in ("report.json", "calibration.json"):
                    shutil.copyfile(source.parent / name, destination / name)
            for comparison in (ROOT / "runs/model_comparisons").glob("*/comparison.json"):
                destination = fixture / comparison.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(comparison, destination)
            source_review = next(iter(api._files("review").values()))
            target_review = fixture / "runs/fixture_review/review.csv"
            target_review.parent.mkdir(parents=True)
            shutil.copyfile(source_review, target_review)
            model_path = api._model_path()
            api = DesktopAPI(fixture)
            api._model_path = lambda: model_path
            api._engine = AnalysisEngine(ROOT)
            api._candidate_engine = AnalysisEngine(ROOT, emotion_model="modernbert")
            report["isolated_workspace"] = True
        html = html.replace("__INITIAL_THEME__", json.dumps(api._preferences().get("theme", "system")))
        window = webview.create_window("TwitterFeel · Research Workbench " + api._app_version()["version"], html=html, js_api=api,
                                       width=1440, height=960, min_size=(1050, 720),
                                       hidden=False, minimized=bool(args.smoke_test),
                                       focus=not bool(args.smoke_test), background_color="#10151C")
        api._window = window

        def probe():
            try:
                def wait_js(expression, seconds=30):
                    until = time.monotonic() + seconds
                    while time.monotonic() < until:
                        value = window.evaluate_js(expression)
                        if value:
                            return value
                        time.sleep(.1)
                    raise RuntimeError("Desktop interaction timed out: " + expression)

                def nav(page, heading):
                    window.evaluate_js("document.querySelector('[data-nav=" + page + "]').click()")
                    wait_js("document.querySelector('h1')?.textContent === " + json.dumps(heading))

                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    ready = window.evaluate_js("Boolean(window.__desktop_ready)")
                    if ready:
                        break
                    time.sleep(.2)
                else:
                    raise RuntimeError("Desktop UI did not finish loading within 60 seconds")
                report["ui_initialized"] = True
                report["heading"] = window.evaluate_js("document.querySelector('h1')?.textContent || document.title")
                window.evaluate_js("window.pywebview.api.call('bootstrap',{}).then(r=>{window.__bridge_probe=r})")
                deadline = time.monotonic() + 15
                response = None
                while time.monotonic() < deadline:
                    response = window.evaluate_js("window.__bridge_probe || null")
                    if response:
                        break
                    time.sleep(.1)
                if not response or not response.get("ok"):
                    raise RuntimeError("Native JS-to-Python bootstrap bridge failed")
                report["bridge_ready"] = True
                report["run_count"] = len(response["data"]["runs"])
                report["model_available"] = response["data"]["model_available"]
                report["local_storage_available"] = bool(window.evaluate_js("(()=>{try{localStorage.setItem('__desktop_probe','1');localStorage.removeItem('__desktop_probe');return true}catch{return false}})()"))
                with api._store._connect() as connection:
                    revisions_before = connection.execute("SELECT count(*) FROM review_revisions").fetchone()[0]
                window.evaluate_js("""window.__desktop_calls=[];const original=window.pywebview.api.call;
                    window.pywebview.api.call=async function(action,payload){const result=await original(action,payload);
                    window.__desktop_calls.push({action,ok:result.ok});if(action==='infer')window.__smoke_inference=result;
                    if(action==='start_analysis'&&result.ok)window.__smoke_last_job=result.data;return result;};""")
                report["pages"] = []
                for page, heading in [("overview", "Overview"), ("explore", "Data explorer"),
                                      ("review", "Label review"), ("lab", "Model lab"),
                                      ("experiments", "Experiments"), ("datasets", "Datasets"),
                                      ("analysis", "Run analysis"), ("findings", "Findings & notes")]:
                    nav(page, heading)
                    report["pages"].append(heading)
                nav("experiments", "Experiments")
                window.evaluate_js("document.querySelector('[data-action=benchmark]').click()")
                wait_js("document.getElementById('benchmark-results')?.textContent.includes('No comparison is ready yet')")
                report["unreviewed_benchmark_has_no_scores"] = True
                window.evaluate_js("document.querySelector('[data-action=prepare-calibration]').click()")
                wait_js("document.querySelector('h1')?.textContent==='Label review' && document.querySelector('main').textContent.includes('Practice round')")
                report["blind_calibration"] = True
                report["label_definitions"] = bool(window.evaluate_js("document.getElementById('definition-search') && document.querySelector('main').textContent.includes('anticipation')"))
                nav("datasets", "Datasets")
                window.evaluate_js("""for(const [id,value] of Object.entries({'import-name':'Native preview diagnostic','import-format':'csv','import-content':'body,person\\nI feel hopeful,one\\nI feel worried,two\\n'})){const field=document.getElementById(id);field.value=value;field.dispatchEvent(new Event(id==='import-format'?'change':'input',{bubbles:true}));}document.querySelector('[data-action=preview-import]').click();""")
                wait_js("document.getElementById('import-text-column')?.options.length===3")
                report["dataset_preview"] = bool(window.evaluate_js("document.querySelector('#import-text-column option[value=body]') && document.querySelector('#import-user-column option[value=person]')"))
                window.evaluate_js("document.querySelector('[data-action=about]').click()")
                report["application_version"] = bool(window.evaluate_js("document.getElementById('dialog').textContent.includes('0.2.0')"))
                window.evaluate_js("document.querySelector('[data-action=close-dialog]').click()")
                nav("explore", "Data explorer")
                window.evaluate_js("const input=document.getElementById('post-search');input.value='__desktop_smoke_no_match__';input.dispatchEvent(new Event('input',{bubbles:true}));")
                report["search_filter"] = bool(wait_js("document.getElementById('post-results').textContent.includes('0 matching posts')"))
                window.evaluate_js("const input=document.getElementById('post-search');input.value='';input.dispatchEvent(new Event('input',{bubbles:true}));const filter=document.getElementById('emotion-filter');filter.value='joy';filter.dispatchEvent(new Event('change',{bubbles:true}));")
                report["emotion_filter"] = bool(window.evaluate_js("document.getElementById('emotion-filter').value==='joy' && Array.from(document.querySelectorAll('#post-results tbody tr[data-post]')).every(r=>r.textContent.includes('joy'))"))
                nav("review", "Label review")
                window.evaluate_js("document.querySelector('[data-emotion=joy]').click();document.querySelector('[data-emotion=fear]').click();")
                report["review_multilabel_selection"] = bool(window.evaluate_js("document.querySelector('[data-emotion=joy]').getAttribute('aria-pressed')==='true' && document.querySelector('[data-emotion=fear]').getAttribute('aria-pressed')==='true'"))
                window.evaluate_js("document.querySelector('[data-emotion=neutral]').click()")
                report["neutral_exclusive_selection"] = bool(window.evaluate_js("document.querySelectorAll('[data-emotion].active').length===1 && document.querySelector('[data-emotion=neutral]').getAttribute('aria-pressed')==='true'"))
                nav("lab", "Model lab")
                window.evaluate_js("const slider=document.getElementById('threshold');slider.value='0.8';slider.dispatchEvent(new Event('input',{bubbles:true}));")
                wait_js("window.__desktop_calls.some(x=>x.action==='sandbox' && x.ok)")
                report["threshold_update"] = bool(window.evaluate_js("document.getElementById('threshold-value').textContent==='0.80'"))
                window.evaluate_js("document.querySelector('[data-tab=\"lab:try\"]').click();document.querySelector('[data-action=infer]').click();")
                wait_js("window.__smoke_inference?.ok && document.querySelector('#trial-results .result-box')", 90)
                inference = window.evaluate_js("window.__smoke_inference.data")
                report["inference"] = {key: inference[key] for key in ("text", "labels", "status", "truncated")}
                nav("datasets", "Datasets")
                window.evaluate_js("const text=document.getElementById('import-text-column');text.value='body';text.dispatchEvent(new Event('change',{bubbles:true}));const user=document.getElementById('import-user-column');user.value='person';user.dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[data-action=import-dataset]').click();")
                wait_js("document.querySelector('[data-dataset]')")
                report["dataset_saved"] = True
                nav("analysis", "Run analysis")
                window.evaluate_js("document.querySelector('[data-action=start-analysis]').click()")
                wait_js("document.querySelector('#analysis-progress')?.textContent.includes('Analysis saved')", 90)
                report["batch_analysis"] = True
                profiles = api._calibration_profiles()
                if profiles:
                    window.evaluate_js("const select=document.getElementById('calibration-profile');select.selectedIndex=1;select.dispatchEvent(new Event('change',{bubbles:true}));")
                    selected_profile = window.evaluate_js("document.getElementById('calibration-profile').value")
                    report["profile_controls"] = bool(window.evaluate_js("document.getElementById('calibration-profile').value && !document.getElementById('analysis-threshold')"))
                    window.evaluate_js("document.querySelector('[data-action=start-analysis]').click()")
                    wait_js("window.__desktop_calls.filter(x=>x.action==='start_analysis'&&x.ok).length>=2 && document.querySelector('#analysis-progress')?.textContent.includes('Analysis saved')", 90)
                    job = window.evaluate_js("window.__smoke_last_job")
                    completed_job = api._jobs.status(job["id"])
                    calibrated_run = api._load_run({"run_id": completed_job["run_id"]})
                    report["research_profile_applied"] = calibrated_run["config"].get("calibration_profile", {}).get("id") == selected_profile
                    if not report["profile_controls"] or not report["research_profile_applied"]:
                        raise RuntimeError("Saved emotion profile did not reach the analysis model")
                if api._engine_for("modernbert").available_models()["emotion"]:
                    window.evaluate_js("const select=document.getElementById('emotion-model');select.value='modernbert';select.dispatchEvent(new Event('change',{bubbles:true}));")
                    report["model_change_resets_profile"] = bool(window.evaluate_js("document.getElementById('calibration-profile').value==='' && Boolean(document.getElementById('analysis-threshold'))"))
                    window.evaluate_js("const select=document.getElementById('calibration-profile');select.selectedIndex=1;select.dispatchEvent(new Event('change',{bubbles:true}));")
                    candidate_profile = window.evaluate_js("document.getElementById('calibration-profile').value")
                    window.evaluate_js("document.querySelector('[data-action=start-analysis]').click()")
                    wait_js("window.__desktop_calls.filter(x=>x.action==='start_analysis'&&x.ok).length>=3 && document.querySelector('#analysis-progress')?.textContent.includes('Analysis saved')", 90)
                    job = window.evaluate_js("window.__smoke_last_job")
                    candidate_run = api._load_run({"run_id": api._jobs.status(job["id"])["run_id"]})
                    report["alternative_model_applied"] = candidate_run["config"]["model"] == "cirimus/modernbert-base-go-emotions"
                    report["candidate_profile_applied"] = bool(candidate_profile) and candidate_run["config"].get("calibration_profile", {}).get("id") == candidate_profile
                    if not all(report[key] for key in ("model_change_resets_profile", "alternative_model_applied", "candidate_profile_applied")):
                        raise RuntimeError("The candidate model or its matching profile did not reach the analysis run")
                window.evaluate_js("document.querySelector('[data-action=job-findings]').click()")
                wait_js("document.querySelector('h1')?.textContent==='Findings & notes' && document.getElementById('finding-title')")
                window.evaluate_js("for(const[id,value]of Object.entries({'finding-title':'Synthetic native check','finding-note':'Execution check only; no accuracy claim.'})){const field=document.getElementById(id);field.value=value;field.dispatchEvent(new Event('input',{bubbles:true}));}document.querySelector('[data-action=save-finding]').click();")
                wait_js("document.querySelector('main').textContent.includes('Synthetic native check') && !document.getElementById('finding-title').value")
                report["authored_observation"] = True
                window.evaluate_js("document.querySelector('[data-action=export-findings]').click()")
                wait_js("window.__desktop_calls.some(x=>x.action==='export_findings' && x.ok)")
                report["findings_export"] = True
                report["page_previews"] = {}
                report["appearance_previews"] = {}
                nav("datasets", "Datasets")
                draft_before = window.evaluate_js("document.getElementById('import-content').value")
                window.evaluate_js("document.querySelector('[data-action=appearance]').click()")
                for mode in ("light", "dark", "system"):
                    window.evaluate_js("document.querySelector('[data-theme-choice=" + mode + "]').click()")
                    wait_js("document.querySelector('[data-theme-choice=" + mode + "]').getAttribute('aria-checked')==='true'")
                    if DesktopAPI(fixture)._preferences().get("theme") != mode:
                        raise RuntimeError("Appearance preference was not persisted")
                    expected = window.evaluate_js("matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'") if mode == "system" else mode
                    if window.evaluate_js("document.documentElement.dataset.theme") != expected or window.evaluate_js("document.body.dataset.theme") != expected:
                        raise RuntimeError("Appearance did not reach the app surfaces")
                    preview = args.smoke_test.with_name(args.smoke_test.stem + "-appearance-" + mode + ".png").resolve()
                    _capture_preview(window, preview)
                    report["appearance_previews"][mode] = str(preview)
                report["theme_preference_persisted"] = True
                report["theme_preserves_drafts"] = window.evaluate_js("document.getElementById('import-content').value") == draft_before
                if not report["theme_preserves_drafts"]:
                    raise RuntimeError("Appearance change discarded the import draft")
                window.evaluate_js("document.querySelector('[data-action=close-dialog]').click()")
                nav("overview", "Overview")
                for mode in ("light", "dark"):
                    window.evaluate_js("document.querySelector('[data-action=appearance]').click();document.querySelector('[data-theme-choice=" + mode + "]').click()")
                    wait_js("document.body.dataset.theme==='" + mode + "'")
                    window.evaluate_js("document.querySelector('[data-action=close-dialog]').click();document.getElementById('toast').className=''")
                    preview = args.smoke_test.with_name(args.smoke_test.stem + "-overview-" + mode + ".png").resolve()
                    _capture_preview(window, preview)
                    report["appearance_previews"]["overview_" + mode] = str(preview)
                for page, heading in (("datasets", "Datasets"), ("analysis", "Run analysis"), ("findings", "Findings & notes"),
                                      ("experiments", "Experiments"), ("review", "Label review")):
                    nav(page, heading)
                    preview = args.smoke_test.with_name(args.smoke_test.stem + "-" + page + ".png").resolve()
                    window.evaluate_js("window.scrollTo(0,0)")
                    _capture_preview(window, preview)
                    report["page_previews"][page] = str(preview)
                nav("overview", "Overview")
                try:
                    preview = args.smoke_test.with_suffix(".png").resolve()
                    _capture_preview(window, preview)
                    report["preview"] = str(preview)
                except Exception as exc:
                    report["preview_error"] = str(exc)
                report["javascript_errors"] = window.evaluate_js("window.__desktop_errors || []")
                report["error_toast"] = window.evaluate_js("document.querySelector('#toast.error')?.textContent || ''")
                with api._store._connect() as connection:
                    report["reviews_unchanged"] = connection.execute("SELECT count(*) FROM review_revisions").fetchone()[0] == revisions_before
                report["listening_tcp"] = _listeners()
                report["no_listening_tcp"] = not report["listening_tcp"]
                report["success"] = all(report[key] for key in ("no_listening_tcp", "reviews_unchanged", "search_filter", "emotion_filter", "review_multilabel_selection", "neutral_exclusive_selection", "threshold_update", "dataset_preview", "application_version", "dataset_saved", "batch_analysis", "authored_observation", "findings_export", "blind_calibration", "label_definitions", "unreviewed_benchmark_has_no_scores")) and not report["javascript_errors"] and not report["error_toast"]
            except Exception as exc:
                report["error"] = str(exc)
            finally:
                smoke_done.set()
                window.destroy()

        storage = ROOT / "runs/workbench/webview"
        storage.mkdir(parents=True, exist_ok=True)
        webview.start(probe if args.smoke_test else None, gui="edgechromium", http_server=False,
                      debug=False, storage_path=str(storage), private_mode=False,
                      icon=str(assets / "app.ico") if (assets / "app.ico").exists() else None)
        if smoke_workspace is not None:
            smoke_workspace.cleanup()
        if not args.smoke_test:
            report["success"] = True
    except Exception as exc:
        report["error"] = str(exc)
        if not args.smoke_test:
            ctypes.windll.user32.MessageBoxW(0, "Could not start TwitterFeel: " + str(exc) + "\nSee README.md for desktop setup.", "TwitterFeel", 16)
    finally:
        if args.smoke_test:
            args.smoke_test.parent.mkdir(parents=True, exist_ok=True)
            args.smoke_test.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logs = ROOT / "runs/workbench/logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / ("native-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
