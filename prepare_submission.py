"""After your OWN lab run, produce README with real results and one PDF.

Before running, edit submission_details.json with your OWN findings and URLs,
and save one genuine screenshot from BITS Virtual Lab at evidence/bits_virtual_lab.png.

    python prepare_submission.py

This generator never fabricates scores, observations, URLs, or screenshots.
"""

import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from common import METRICS_FILE, MODEL_DIR, MODEL_FILES, ROOT, TEST_FILE

DETAILS_PATH = ROOT / "submission_details.json"
README_PATH = ROOT / "README.md"
REPORT_PATH = ROOT / "submission.pdf"
MODEL_HEADING = "## d. Models used - Comparison table"
NEXT_HEADING = "## Training and deployment methodology"


def must_have_details(config):
    for key in ["student_name", "student_id", "github_repository_url", "streamlit_app_url", "overall_winner"]:
        value = str(config.get(key, "")).strip()
        if not value or "REPLACE" in value or "YOUR-" in value or "PENDING" in value:
            raise ValueError(f"Complete your own '{key}' in submission_details.json")
    for key in ["github_repository_url", "streamlit_app_url"]:
        if not config[key].startswith("https://"):
            raise ValueError(f"{key} must be a real https link")
    for model_label in MODEL_FILES:
        note = config.get("model_observations", {}).get(model_label, "")
        if len(note.strip()) < 25 or "REPLACE" in note or "PENDING" in note:
            raise ValueError(f"Write your own observation for {model_label}")
    screenshot = ROOT / config.get("virtual_lab_screenshot_path", "")
    if screenshot.is_dir() or not screenshot.is_file():
        raise ValueError("Add one real BITS lab screenshot at the configured path")
    return screenshot


def build_tables(config, comparison):
    lines = [
        MODEL_HEADING, "",
        "The following results were calculated on the same untouched labelled held-out test split.",
        "",
        "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in MODEL_FILES:
        row = comparison.loc[label]
        result_values = " | ".join(f"{row[col]:.4f}" for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])
        lines.append(f"| {label} | {result_values} |")
    lines += [
        "", "## Model performance observations", "",
        "| ML Model Name | Observation about model performance |",
        "|---|---|",
    ]
    for label in MODEL_FILES:
        safe_note = config["model_observations"][label].replace("|", "/").replace("\n", " ")
        lines.append(f"| {label} | {safe_note} |")
    safe_winner = config["overall_winner"].replace("|", "/").replace("\n", " ")
    lines.append(f"| Overall winner for your dataset? | {safe_winner} |")
    lines.append("")
    return "\n".join(lines)


def make_readme(config, comparison):
    existing = README_PATH.read_text(encoding="utf-8")
    before, _, rest = existing.partition(MODEL_HEADING)
    if not rest or NEXT_HEADING not in rest:
        raise ValueError("README.md is missing the expected model section headings")
    _, _, after = rest.partition(NEXT_HEADING)
    before = re.sub(
        r"(## c\. GitHub Repository Link\n\n)[^\n]+",
        lambda m: m.group(1) + config["github_repository_url"],
        before,
    )
    before = before.rstrip() + "\n\n"
    markdown = before + build_tables(config, comparison) + "\n" + NEXT_HEADING + after
    README_PATH.write_text(markdown, encoding="utf-8")
    return markdown


def render_readme_markdown(readme, story, styles):
    """Render the complete current README, preserving every content line."""
    lines = readme.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i]); i += 1
            rows = []
            for item in table_lines:
                cols = [x.strip() for x in item.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", x or "") for x in cols):
                    continue
                rows.append(cols)
            if rows:
                count = len(rows[0])
                widths = [132] + [62] * 6 if count == 7 else [135, 365] if count == 2 else None
                table_data = [[Paragraph(escape(col), styles["Cell"]) for col in row] for row in rows]
                table = Table(table_data, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9F1FB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor("#355C8C")),
                    ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#D2DDEB")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(table)
                story.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), styles["H1Custom"]))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), styles["H2Custom"]))
        elif not line.strip():
            story.append(Spacer(1, 4))
        else:
            story.append(Paragraph(escape(line), styles["BodyCustom"]))
        i += 1


def build_pdf(config, screenshot, markdown):
    doc = SimpleDocTemplate(
        str(REPORT_PATH), pagesize=A4, leftMargin=42, rightMargin=42,
        topMargin=42, bottomMargin=47,
        title="Machine Learning Assignment 2 - BITS Pilani",
        author=config["student_name"],
    )
    base = getSampleStyleSheet()
    styles = {
        "H1Custom": ParagraphStyle("H1Custom", parent=base["Heading1"],
            fontName="Helvetica-Bold", fontSize=17, textColor=colors.HexColor("#16375C"),
            leading=22, spaceAfter=10),
        "H2Custom": ParagraphStyle("H2Custom", parent=base["Heading2"],
            fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor("#16375C"),
            leading=14, spaceBefore=10, spaceAfter=5),
        "BodyCustom": ParagraphStyle("BodyCustom", parent=base["BodyText"],
            fontName="Helvetica", fontSize=8.7, leading=13, spaceAfter=3,
            splitLongWords=True, wordWrap="CJK"),
        "Cell": ParagraphStyle("Cell", parent=base["BodyText"],
            fontName="Helvetica", fontSize=7.2, leading=10, splitLongWords=True,
            wordWrap="CJK"),
        "Caption": ParagraphStyle("CaptionCustom", parent=base["BodyText"],
            fontSize=9, textColor=colors.HexColor("#52667B"), leading=13),
    }
    story = [
        Paragraph("BITS Pilani - M.Tech (AI/ML)", styles["Caption"]),
        Paragraph("Machine Learning | Assignment 2", styles["H1Custom"]),
        Paragraph(
            f"Student: {escape(config['student_name'])} | Student ID: {escape(config['student_id'])}",
            styles["BodyCustom"],
        ),
        Spacer(1, 14),
        Paragraph("1. GitHub repository link", styles["H2Custom"]),
        Paragraph(
            f"<link href=\"{escape(config['github_repository_url'], {'\"': '&quot;'})}\" color=\"#1652A8\">"
            f"{escape(config['github_repository_url'])}</link>",
            styles["BodyCustom"],
        ),
        Paragraph("2. Live Streamlit application link", styles["H2Custom"]),
        Paragraph(
            f"<link href=\"{escape(config['streamlit_app_url'], {'\"': '&quot;'})}\" color=\"#1652A8\">"
            f"{escape(config['streamlit_app_url'])}</link>",
            styles["BodyCustom"],
        ),
        Paragraph("3. Evidence: BITS Virtual Lab execution screenshot", styles["H2Custom"]),
        Spacer(1, 7),
    ]
    with PILImage.open(screenshot) as im:
        w, h = im.size
    maximum_w, maximum_h = 495, 340
    factor = min(maximum_w / w, maximum_h / h)
    story.append(Image(str(screenshot), width=w * factor, height=h * factor))
    story.append(Paragraph("Captured by the student while executing the assignment in BITS Virtual Lab.", styles["Caption"]))
    story.append(PageBreak())
    story.append(Paragraph("4. Complete GitHub README.md content", styles["H1Custom"]))
    render_readme_markdown(markdown, story, styles)

    def page_number(canvas, document):
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#66798B"))
        canvas.drawRightString(A4[0] - 42, 22, f"Page {document.page}")

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


def main():
    config = json.loads(DETAILS_PATH.read_text(encoding="utf-8"))
    screenshot = must_have_details(config)
    for label, filename in MODEL_FILES.items():
        if not (MODEL_DIR / filename).is_file():
            raise ValueError(f"Saved model missing for {label}: {filename}")
    if not TEST_FILE.is_file():
        raise ValueError("test_data.csv missing; run train_models.py first")
    if not METRICS_FILE.is_file():
        raise ValueError("model_comparison.csv missing; run train_models.py first")
    comparison = pd.read_csv(METRICS_FILE).set_index("ML Model Name")
    if set(MODEL_FILES) != set(comparison.index):
        raise ValueError("Comparison CSV must contain exactly the five models")
    markdown = make_readme(config, comparison)
    build_pdf(config, screenshot, markdown)
    print("Updated:", README_PATH)
    print("PDF created:", REPORT_PATH)
    print("Review the PDF and verify links, screenshot, metrics and YOUR observations.")


if __name__ == "__main__":
    main()
