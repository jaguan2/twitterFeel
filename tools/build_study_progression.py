"""Build the dated research record from its editable Markdown master.

Install reporting-only dependency: python -m pip install reportlab
Then run from any directory: python tools/build_study_progression.py
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/STUDY_PROGRESSION.md"
NAVY = colors.HexColor("#153044")
TEAL = colors.HexColor("#167B82")
GRAY = colors.HexColor("#52616C")
PALE = colors.HexColor("#EFF5F6")


def inline(text):
    """Escape source text before applying the two supported inline styles."""
    parts = re.split(r"(`[^`]+`)", text)
    rendered = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            rendered.append('<font name="Courier" size="8">' + html.escape(part[1:-1]) + '</font>')
        else:
            rendered.append(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(part)))
    return "".join(rendered)


def make_styles():
    regular = Path("C:/Windows/Fonts/segoeui.ttf")
    bold = Path("C:/Windows/Fonts/segoeuib.ttf")
    font, strong = "Helvetica", "Helvetica-Bold"
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("StudySans", str(regular)))
        pdfmetrics.registerFont(TTFont("StudySans-Bold", str(bold)))
        pdfmetrics.registerFontFamily("StudySans", normal="StudySans", bold="StudySans-Bold")
        font, strong = "StudySans", "StudySans-Bold"
    base = dict(fontName=font, textColor=NAVY, alignment=TA_LEFT)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="StudyTitle", fontName=strong, fontSize=34,
                              leading=39, spaceBefore=12, spaceAfter=12, textColor=NAVY))
    styles.add(ParagraphStyle(name="StudyDate", fontName=strong, fontSize=19,
                              leading=24, spaceBefore=3, spaceAfter=15, textColor=NAVY,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="StudySub", fontName=strong, fontSize=11,
                              leading=15, spaceBefore=12, spaceAfter=6, textColor=TEAL,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="StudyBody", fontSize=9.7, leading=14.3,
                              spaceAfter=8, **base))
    styles.add(ParagraphStyle(name="StudyBullet", fontSize=9.4, leading=13.5,
                              leftIndent=11, firstLineIndent=0, bulletIndent=0,
                              spaceAfter=6, **base))
    styles.add(ParagraphStyle(name="StudyCell", fontSize=8.2, leading=11.3,
                              spaceAfter=0, **base))
    styles.add(ParagraphStyle(name="StudyHeadCell", fontName=strong, fontSize=8.1,
                              leading=11, textColor=colors.white))
    styles.add(ParagraphStyle(name="StudySource", fontSize=8.6, leading=12.4,
                              leftIndent=10, bulletIndent=0, spaceAfter=5, **base))
    return styles, font


def make_table(lines, styles, width):
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Table rows must have matching column counts")
    rows.pop(1)  # Markdown alignment/separator row.
    count = len(rows[0])
    if count == 2:
        weights = [0.32, 0.68]
    elif count == 4 and rows[0][0] == "Experiment":
        weights = [0.13, 0.53, 0.17, 0.17]
    elif count == 3:
        weights = [0.36, 0.22, 0.42]
    elif count in (4, 5, 6):
        first = 0.34 if count < 6 else 0.23
        if rows[0][0] == "Task":
            first = 0.17
        if "Weighting" in rows[0]:
            weights = [0.22, 0.23, 0.18, 0.18, 0.19]
        else:
            weights = [first] + [(1-first)/(count-1)] * (count-1)
    else:
        weights = [1/count] * count
    cells = [[Paragraph(inline(cell), styles["StudyHeadCell" if i == 0 else "StudyCell"])
              for cell in row] for i, row in enumerate(rows)]
    table = Table(cells, colWidths=[width*w for w in weights], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), .4, colors.HexColor("#D5E2E5")),
    ]))
    return KeepTogether([Spacer(1, 4), table, Spacer(1, 12)])


def parse_markdown(source, styles, width):
    lines = source.splitlines()
    story = []
    index = 0
    in_sources = False
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            index += 1
            continue
        if line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            story.append(make_table(block, styles, width))
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line[level:].strip()
            in_sources = in_sources or title == "Source register"
            style = "StudyTitle" if level == 1 else "StudyDate" if level == 2 else "StudySub"
            story.append(Paragraph(inline(title), styles[style]))
            if level == 2:
                story.extend([HRFlowable(width=width, thickness=1.3, color=TEAL), Spacer(1, 9)])
            index += 1
            continue
        bullet = line.startswith("- ")
        block = [line[2:] if bullet else line]
        index += 1
        while index < len(lines):
            following = lines[index].strip()
            if not following or following.startswith(("#", "|", "- ", "<!--")):
                break
            block.append(following)
            index += 1
        style = "StudySource" if bullet and in_sources else "StudyBullet" if bullet else "StudyBody"
        story.append(Paragraph(inline(" ".join(block)), styles[style], bulletText="-" if bullet else None))
    return story


def evidence_manifest(source_path, output_path, edition_date):
    """Record actual source bytes, without embedding private research data."""
    paths = sorted(set(
        list((ROOT / "docs").glob("*.md"))
        + list((ROOT / "src/twitterfeel").rglob("*.py"))
        + list((ROOT / "src/twitterfeel/desktop/assets").glob("*"))
        + [ROOT / "src/twitterfeel/labeling/annotation_rubric.json"]
        + list((ROOT / "metrics").glob("*.json"))
        + [ROOT / name for name in [
            "HANDOFF.md", "CHANGELOG.md", "app_version.json", "docs/CAP4773_fin_1.ipynb",
            "src/preprocess.py", "src/analysis_engine.py", "src/analysis_jobs.py",
            "src/train.py", "src/train_binary.py", "src/train_leaky_binary.py",
            "data/interim/dataset_godknowswhat.csv",
            "src/prepare_reddit.py", "twitter_feel_plus/README.md",
            "twitter_feel_plus/sadness_flag.py", "twitter_feel_plus/results/REPORT.md",
            "twitter_feel_plus/results/params.json", "twitter_feel_plus/results/aggregate.json",
            "tools/build_study_progression.py",
        ]]
    ))
    external = [Path.home() / "Downloads" / name for name in
                ["Project Proposal Presentation.pptx", "Social Media Project Report.pdf"]]
    sources = []
    for path in paths + external:
        identifier = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else "external/" + path.name
        sources.append({"source": identifier, "available": path.is_file(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    return {"edition_date": edition_date, "timezone": "America/New_York",
            "repository_head": commit.stdout.strip() if commit.returncode == 0 else None,
            "note": "Working-tree evidence snapshot; uncommitted documentation may be included. Hashes identify bytes, not scientific validity.",
            "master": source_path.relative_to(ROOT).as_posix(),
            "pdf_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(), "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/twitterfeel-study-progression.pdf")
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    match = re.search(r"Compiled (\d{4}-\d{2}-\d{2})", source)
    if match is None:
        raise ValueError("The Markdown master must include its compiled date")
    edition_date = match.group(1)
    edition = re.search(r"Edition (\d+)", source)
    if edition is None:
        raise ValueError("The Markdown master must include its edition number")
    styles, font = make_styles()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=44, leftMargin=44,
                            topMargin=51, bottomMargin=45, title="TwitterFeel: Study progression and findings",
                            author="TwitterFeel project", subject="Dated research findings, limitations and evidence")

    def page_chrome(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setFont(font, 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(44, height-28, "TWITTERFEEL / STUDY RECORD")
        canvas.drawRightString(width-44, height-28, "EDITION " + edition.group(1))
        canvas.setStrokeColor(colors.HexColor("#D5E2E5"))
        canvas.line(44, 34, width-44, 34)
        canvas.setFont(font, 7.5)
        canvas.drawString(44, 22, f"Compiled {edition_date} | Retrospective research record")
        canvas.drawRightString(width-44, 22, str(document.page))
        canvas.restoreState()

    # Platypus frames reserve six points of padding on each side.
    story = parse_markdown(source, styles, doc.width - 12)
    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    manifest_path = output.with_suffix(".sources.json")
    manifest_path.write_text(json.dumps(evidence_manifest(SOURCE, output, edition_date), indent=2) + "\n", encoding="utf-8")
    print(output)
    print(manifest_path)


if __name__ == "__main__":
    main()
