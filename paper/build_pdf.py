#!/usr/bin/env python3
"""Render PAPER.md to a formatted PAPER.pdf using reportlab.

Reads the markdown source directly so the PDF always matches PAPER.md. Handles the
template structure (title, author block, abstract, numbered sections), the results
table, an embedded figure after the Results heading, and the reference list.
"""
import re
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable,
)

ROOT = Path(__file__).resolve().parent
MD = ROOT / "PAPER.md"
FIG = ROOT / ".worktrees/plus-factor-forensics/artifacts/kaggle/out/results/figures/fig_naive_vs_auditor_q3.png"
OUT = ROOT / "PAPER.pdf"

s = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=s["Title"], fontSize=16, leading=19, spaceAfter=6)
AUTH = ParagraphStyle("A", parent=s["BodyText"], fontSize=10, leading=13, alignment=1, spaceAfter=1)
H1 = ParagraphStyle("H1", parent=s["Heading1"], fontSize=12.5, spaceBefore=11, spaceAfter=3)
H2 = ParagraphStyle("H2", parent=s["Heading2"], fontSize=10.5, spaceBefore=7, spaceAfter=2)
BODY = ParagraphStyle("B", parent=s["BodyText"], fontSize=9.5, leading=13, spaceAfter=6, alignment=4)
ABS = ParagraphStyle("Abs", parent=BODY, fontSize=9, leading=12, leftIndent=14, rightIndent=14)
CAP = ParagraphStyle("C", parent=s["BodyText"], fontSize=8.5, leading=11, alignment=1, textColor=colors.grey, spaceAfter=8, spaceBefore=2)
REF = ParagraphStyle("R", parent=s["BodyText"], fontSize=8.2, leading=10.5, spaceAfter=3, leftIndent=12, firstLineIndent=-12)


def md_inline(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    return t


def results_table():
    data = [
        ["Model", "Type", "Naive FPR", "Auditor FPR", "Auditor recall"],
        ["Qwen2.5-7B", "open", "0.0 (uninform.)", "0.0", "0.5"],
        ["Llama-3.1-8B", "open", "0.0 (uninform.)", "0.0", "0.0"],
        ["GPT-5.5", "frontier", "0.0", "0.0", "0.5"],
        ["Gemini-3.8-flash", "frontier", "0.0", "0.0", "0.5"],
        ["Gemini-2.5-pro", "frontier", "1.0 (flags all)", "0.0", "0.0"],
        ["Claude Sonnet 4.6", "frontier", "1.0 (flags all)", "0.0", "0.0"],
    ]
    t = Table(data, colWidths=[1.5*inch, 0.8*inch, 1.3*inch, 1.0*inch, 1.1*inch])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("BACKGROUND", (2, 5), (2, 6), colors.HexColor("#ffe0e0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build():
    lines = MD.read_text().splitlines()
    e = []
    i = 0
    # title
    e.append(Paragraph(lines[0].lstrip("# ").strip(), TITLE))
    i = 1
    in_refs = False
    in_table = False
    emitted_fig = False
    buf = []

    def flush():
        nonlocal buf
        if buf:
            e.append(Paragraph(md_inline(" ".join(buf)), BODY))
            buf = []

    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("# "):
            i += 1; continue
        if ln.startswith("## "):
            flush()
            title = ln[3:].strip()
            in_refs = title.lower().startswith("references")
            e.append(Paragraph(title, H1))
            i += 1; continue
        if ln.startswith("### "):
            flush(); e.append(Paragraph(ln[4:].strip(), H2)); i += 1; continue
        if ln.startswith("|"):
            # table region: emit our styled table once, then the figure, skip raw rows
            if not in_table:
                flush(); e.append(results_table()); in_table = True
                if not emitted_fig and FIG.exists():
                    img = Image(str(FIG))
                    maxw = 4.0*inch
                    img.drawWidth = maxw
                    img.drawHeight = maxw * (img.imageHeight / float(img.imageWidth))
                    img.hAlign = "CENTER"
                    e.append(Spacer(1, 6)); e.append(img)
                    e.append(Paragraph(
                        "Figure 1. Naive detector versus plus-factor auditor on the Q3 "
                        "innocent-parallelism control. The naive approach treats behavioral "
                        "similarity as collusion; the auditor requires evidence of agreement "
                        "and stays quiet.", CAP))
                    emitted_fig = True
            i += 1; continue
        else:
            in_table = False
        if not ln.strip():
            flush()
            i += 1; continue
        # author block and affiliation lines near the top
        if i <= 6 and ("Affiliation" in ln or ln.startswith("With ")):
            flush()
            e.append(Paragraph(md_inline(ln), AUTH))
            if ln.startswith("With "):
                e.append(Spacer(1, 4)); e.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
            i += 1; continue
        if ln.startswith("Table 1."):
            flush(); e.append(Paragraph(md_inline(ln), CAP)); i += 1; continue
        if in_refs:
            flush(); e.append(Paragraph(md_inline(ln), REF)); i += 1; continue
        buf.append(ln)
        i += 1
    flush()

    doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                            leftMargin=0.8*inch, rightMargin=0.8*inch,
                            topMargin=0.7*inch, bottomMargin=0.7*inch)
    doc.build(e)
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    build()
