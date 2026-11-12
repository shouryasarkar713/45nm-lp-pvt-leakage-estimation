"""Stage 17/18: checks the rendered PDF against the result files (numbers, page content)."""
from docx import Document as _D
import os, re, json, subprocess, sys, numpy as np, pandas as pd
from config import *
RP = os.path.join(ROOT, "report", "Project1_Report_45nmLP_1V_ParityChecker.pdf")
def text(pg=None):
    a = ["pdftotext", "-layout"] + (["-f", str(pg), "-l", str(pg)] if pg else []) + [RP, "-"]; return subprocess.run(a, capture_output=True, text=True).stdout
res = []
def chk(n, ok, d=""): res.append(ok); print(("PASS " if ok else "FAIL ") + n + (" -- " + d if d else ""))
npg = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", RP], capture_output=True, text=True).stdout).group(1)); print("pages:", npg)
def has_img(p): return len(subprocess.run(["pdfimages", "-list", "-f", str(p), "-l", str(p), RP], capture_output=True, text=True).stdout.strip().splitlines()) > 2
chk("no empty pages (every page has text > 200 chars, or an embedded figure plus a caption)", all(len(text(p).strip()) > 200 or (has_img(p) and len(text(p).strip()) > 40) for p in range(1, npg + 1)))
T = text(); flat = re.sub(r"\s+", " ", T)
chk("no 'nan' / 'None' tokens in report text", not re.search(r"\b(nan|None)\b", flat))
g = json.load(open(os.path.join(OUT, "gate", "gate_overall_metrics.json"))); c = json.load(open(os.path.join(OUT, "circuit", "circuit_overall_metrics.json")))
for name, v in (("gate T", g["T"]["mean_accuracy_pct"]), ("gate S", g["S"]["mean_accuracy_pct"]), ("circuit T", c["T"]["mean_accuracy_pct"]), ("circuit S plain", c["S"]["mean_accuracy_pct"]), ("circuit S load-corrected", c["S_load_corrected"]["mean_accuracy_pct"])):
    chk(f"{name} mean accuracy {v:.2f} % appears in report", f"{v:.2f}" in flat)
gt = pd.read_csv(os.path.join(OUT, "gate", "gate_results_table_T.csv")); miss = [f"{x:.2f}" for x in gt.mean_accuracy_pct if f"{x:.2f}" not in flat]
chk("all 18 gate-table accuracies appear", not miss, str(miss))
sw = pd.read_csv(os.path.join(OUT, "audit", "solver_warning_summary.csv")); chk("solver audit classes and counts appear", all(r.classification in flat.replace(" ", "") or r.classification.replace("_", " ") in flat or r.classification in flat for r in sw.itertuples()))
sw_ = pd.read_csv(os.path.join(OUT, "audit", "solver_warning_audit.csv")); nv_ = sw_.n_valid_resolves
chk("solver section states the recomputed valid-solution range and median", f"{int(nv_.min())}-{int(nv_.max())} KCL-valid solutions per point, with a median of {int(nv_.median())}" in flat)
pvT_ = pd.read_csv(os.path.join(OUT, "circuit", "circuit_results_per_vector_T.csv")); chk("per-vector subsection present with recomputed extreme vectors", "Per-input-vector circuit leakage estimation results" in flat and f"{pvT_.median_error_pct.min():.2f}" in flat and f"{pvT_.median_error_pct.max():.2f}" in flat)
dcs_ = pd.read_csv(os.path.join(OUT, "audit", "dc_branch_probe.csv")); chk("DC-branch section reports the probes without a valid solution", f"{int((dcs_.n_valid==0).sum())} probes (" in flat and "does not establish" in flat)
chk("accuracy equation present", "Accuracy_Q" in flat)
chk("conclusion present on the last pages", "19. Conclusion" in flat)
chk("all figures embedded (>= 12 images)", len(subprocess.run(["pdfimages", "-list", RP], capture_output=True, text=True).stdout.strip().splitlines()) - 2 >= 12)
bad_terms = re.findall(r"(?i)(previous (?:implementation|solution|team|version|project)|earlier (?:agent|version|implementation|revision)|\bGLM\b|\bagent\b|\brevision\b|\biteration\b|corrected version|final pass|new model|updated model|\b(?:work|dist)[0-9]*/)", flat)
chk("no development-history wording in the report text", not bad_terms, str(bad_terms[:5]))
_d = _D(os.path.join(ROOT, "report", "Project1_Report_45nmLP_1V_ParityChecker.docx"))
_txt = " ".join(p.text for p in _d.paragraphs) + " " + " ".join(c_.text for t_ in _d.tables for r_ in t_.rows for c_ in r_.cells)
_nd = set(re.findall(r"\d+\.\d+", _txt)); _np = set(re.findall(r"\d+\.\d+", flat)); chk("DOCX and PDF contain the same decimal numbers", _nd <= _np and len(_nd) > 300, f"{len(_nd)} numbers; missing in PDF: {sorted(_nd - _np)[:5]}")
_U = "https://github.com/shouryasarkar713/45nm-lp-pvt-leakage-estimation"
_rd = open(os.path.join(ROOT, "README.md")).read()
_hl = subprocess.run(["unzip", "-p", os.path.join(ROOT, "report", "Project1_Report_45nmLP_1V_ParityChecker.docx"), "word/_rels/document.xml.rels"], capture_output=True, text=True).stdout
chk("repository URL visible in the PDF, present as a DOCX hyperlink and in the README", _U in flat.replace(" ", "") and _U in _hl and _U in _rd)
chk("no local machine paths in the report text", "/home/" not in flat and "/tmp/" not in flat)
meta = subprocess.run(["pdfinfo", RP], capture_output=True, text=True).stdout
chk("PDF metadata: neutral title/subject, no author", "Title:           ML-Driven PVT-Variation-Aware Leakage Estimation" in meta and "Author" not in meta and "Subject:         Digital VLSI Design Course Project" in meta)
_cp = _D(os.path.join(ROOT, "report", "Project1_Report_45nmLP_1V_ParityChecker.docx")).core_properties
chk("DOCX metadata: neutral title/subject/keywords, no author, no comments", _cp.title.startswith("ML-Driven") and _cp.author == "" and _cp.comments == "" and _cp.last_modified_by == "")
json.dump(dict(pages=npg, ok=bool(all(res))), open(os.path.join(OUT, "audit", "report_check.json"), "w"))
sys.exit(0 if all(res) else 1)
