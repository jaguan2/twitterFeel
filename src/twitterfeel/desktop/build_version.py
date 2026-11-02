"""Generate Windows executable metadata from the application's version manifest."""
import json
from pathlib import Path
import re
from twitterfeel.common.config import workspace_root



def main():
    root = workspace_root()
    version = json.loads((root / "app_version.json").read_text(encoding="utf-8"))["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Executable versions must use major.minor.patch")
    target = root / "runs/workbench/build-version.cs"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('using System.Reflection;\n[assembly: AssemblyTitle("TwitterFeel Research Workbench")]\n'
                      f'[assembly: AssemblyVersion("{version}.0")]\n'
                      f'[assembly: AssemblyFileVersion("{version}.0")]\n'
                      f'[assembly: AssemblyInformationalVersion("{version}")]\n', encoding="utf-8")


if __name__ == "__main__":
    main()
