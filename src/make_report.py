"""Builds the typed technical report (DOCX) from the generated result files; convert to PDF with LibreOffice."""
import os, json, glob
import numpy as np, pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from config import *

RD = os.path.join(ROOT, "report"); os.makedirs(RD, exist_ok=True)
J = lambda *p: json.load(open(os.path.join(OUT, *p)))
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
for s in doc.sections: s.left_margin = s.right_margin = Cm(2.0); s.top_margin = s.bottom_margin = Cm(2.0)
for lvl, sz in ((1, 15), (2, 12.5), (3, 11)):
    h = doc.styles[f"Heading {lvl}"]; h.font.name = "Calibri"; h.font.size = Pt(sz); h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

def LINK(par, url):
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    rid = par.part.relate_to(url, RT.HYPERLINK, is_external=True)
    h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), rid)
    r = OxmlElement("w:r"); rPr = OxmlElement("w:rPr"); u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); c = OxmlElement("w:color"); c.set(qn("w:val"), "0563C1")
    rPr.append(u); rPr.append(c); r.append(rPr); t = OxmlElement("w:t"); t.text = url; r.append(t); h.append(r); par._p.append(h)
def H(t, l=1): doc.add_heading(t, l)
def Pp(t, bold=False, italic=False, size=None):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(4); return p
def B(items):
    for t in items: p = doc.add_paragraph(t, style="List Bullet"); p.paragraph_format.space_after = Pt(1)
def fmt(x, d=2):
    if isinstance(x, (int, np.integer)): return f"{x:,}"
    if isinstance(x, (float, np.floating)):
        if x != x: return ""
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5): return f"{x:.2e}"
        return f"{x:.{d}f}"
    return str(x)
def T(df, widths=None, size=8, cap=None):
    if cap: Pp(cap, bold=True, size=9).paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]; cell.text = ""; r = cell.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(size)
    for row in df.itertuples(index=False):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(fmt(v)); r.font.size = Pt(size)
    if len(df) <= 14:   # keep short tables together with their caption and avoid orphaned header rows
        for row_ in t.rows[:-1]:
            for c_ in row_.cells:
                for p_ in c_.paragraphs: p_.paragraph_format.keep_with_next = True
    for row_ in t.rows:
        trPr = row_._tr.get_or_add_trPr(); cs_ = OxmlElement("w:cantSplit"); trPr.append(cs_)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
def IMG(path, w=15.5, cap=None):
    if os.path.exists(path):
        doc.add_picture(path, width=Cm(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if cap: p = Pp(cap, italic=True, size=9); p.alignment = WD_ALIGN_PARAGRAPH.CENTER

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_text.py")).read())
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
def _page_number_footer(section):
    p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("ML-Driven PVT-Variation-Aware Leakage Estimation  |  page "); r.font.size = Pt(8)
    r2 = p.add_run(); r2.font.size = Pt(8)
    for typ, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if typ: e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), typ)
        else: e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
        r2._r.append(e)
for sec in doc.sections: _page_number_footer(sec)
cp = doc.core_properties; cp.title = "ML-Driven PVT-Variation-Aware Leakage Estimation"; cp.subject = "Digital VLSI Design Course Project"
cp.keywords = "45nm LP, CMOS Leakage, PVT Variation, Machine Learning, SPICE"; cp.author = ""; cp.last_modified_by = ""; cp.comments = ""; cp.category = ""; cp.language = "en-US"
import datetime; cp.created = cp.modified = datetime.datetime(2026, 11, 13)
doc.save(os.path.join(RD, "Project1_Report_45nmLP_1V_ParityChecker.docx")); print("report written")
