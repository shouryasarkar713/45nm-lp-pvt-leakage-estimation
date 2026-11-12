"""Stage 20: build Final_Submission_Package and Full_Research_Package (+ package-specific manifests), then verify both from fresh extractions."""
import os, sys, shutil, subprocess, zipfile, tempfile
from config import *
DIST = os.path.join(ROOT, "dist")
CORE = ("config spice corners circuit circuit_eval circuit_run gen_transistor gen_twin gen_gate gen_circuit gate_model gate_eval gate_nodes gate_run split ml_data ml_baseline ml_struct ml_infer train_models "
        "data_check make_netlists make_train_manifest make_packages make_report report_text verify_report check_env validate_all oracle_check dev_ablation ladder_consistency dc_branch_probe solver_warning_audit "
        "verify_definitions test_stats_extra solver_repeatability circuit_per_vector compare_rerun repro_test").split()
def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst)
def tree(src, dst, skip=lambda p: False):
    for dp, dn, fn in os.walk(src):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            p = os.path.join(dp, f)
            if not skip(os.path.relpath(p, src)): copy(p, os.path.join(dst, os.path.relpath(p, src)))
def manifest(pkg, name):
    L = []
    for dp, dn, fn in sorted(os.walk(pkg)):
        for f in sorted(fn):
            p = os.path.join(dp, f)
            if os.path.relpath(p, pkg) in (name,): continue            # the manifest never lists itself
            L.append(f"{sha256(p)}  {os.path.relpath(p, pkg)}")
    open(os.path.join(pkg, name), "w").write("\n".join(L) + "\n"); return len(L)
def verify_manifest(pkg, name):
    bad = []; listed = set()
    for l in open(os.path.join(pkg, name)).read().splitlines():
        h, p = l.split("  ", 1); listed.add(p)
        fp = os.path.join(pkg, p)
        if not os.path.exists(fp) or sha256(fp) != h: bad.append(p)
    actual = {os.path.relpath(os.path.join(dp, f), pkg) for dp, _, fn in os.walk(pkg) for f in fn} - {name}
    return bad, sorted(actual - listed), len(listed)
def build():
    shutil.rmtree(DIST, ignore_errors=True); S = os.path.join(DIST, "Final_Submission_Package"); R = os.path.join(DIST, "Full_Research_Package")
    sub_top = ["README.md", "requirements.txt", "run_all.sh", ".gitignore"]
    for f in sub_top: copy(os.path.join(ROOT, f), os.path.join(S, f))
    for f in CORE: copy(os.path.join(ROOT, "src", f + ".py"), os.path.join(S, "src", f + ".py"))
    tree(os.path.join(ROOT, "model"), os.path.join(S, "model")); tree(os.path.join(ROOT, "netlists"), os.path.join(S, "netlists")); tree(os.path.join(ROOT, "report"), os.path.join(S, "report"))
    for c in CASES: copy(os.path.join(DATA, "base", c + ".csv"), os.path.join(S, "data", "base", c + ".csv"))
    for f in ("gate_spice_official.csv", "gate_spice_dev.csv", "corners_official.csv", "corners_dev.csv"): copy(os.path.join(DATA, f), os.path.join(S, "data", f))
    tree(os.path.join(ROOT, "work", "circuit_spice"), os.path.join(S, "work", "circuit_spice"))
    tree(OUT, os.path.join(S, "out"), skip=lambda p: p.startswith("models/struct_") or p.startswith("audit/solver_logs") or p.startswith("MANIFEST"))
    # research package = final package + everything else
    tree(S, R); tree(ROOT, R, skip=lambda p: p.startswith(("dist", "work2")) or "__pycache__" in p or p.endswith(".zip") or p.startswith("MANIFEST") or p.endswith("MANIFEST_sha256.txt") or p.endswith("MANIFEST_research_sha256.txt"))
    n1 = manifest(S, "MANIFEST_sha256.txt")
    copy(os.path.join(S, "MANIFEST_sha256.txt"), os.path.join(R, "MANIFEST_sha256.txt"))      # research package contains the submission manifest too (verifies its subset)
    print("final manifest entries:", n1, " research manifest entries:", manifest(R, "MANIFEST_research_sha256.txt"))
    for pkg in (S, R):
        z = pkg + ".zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for dp, dn, fn in os.walk(pkg):
                for f in fn: zf.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), pkg))
        print(z, os.path.getsize(z) // 2 ** 20, "MB")
def fresh_check():
    ok = True
    for name, man in (("Final_Submission_Package", "MANIFEST_sha256.txt"), ("Full_Research_Package", "MANIFEST_research_sha256.txt")):
        d = tempfile.mkdtemp(prefix="dvd_fresh_"); zipfile.ZipFile(os.path.join(DIST, name + ".zip")).extractall(d)
        bad, extra, n = verify_manifest(d, man); print(name, "manifest entries", n, "mismatched/missing:", bad[:5], len(bad), "unlisted files:", extra[:5], len(extra)); ok &= (not bad and not extra)
        if name == "Full_Research_Package":
            bad2, _, n2 = verify_manifest(d, "MANIFEST_sha256.txt"); print("  submission manifest inside research package:", n2, "entries, mismatches", len(bad2)); ok &= not bad2
        r = subprocess.run([sys.executable, os.path.join(d, "src", "validate_all.py")], capture_output=True, text=True, cwd=os.path.join(d, "src"))
        print(name, "validate_all exit", r.returncode, r.stdout.strip().splitlines()[-2:]); ok &= (r.returncode == 0)
        shutil.rmtree(d, ignore_errors=True)
    return ok
if __name__ == "__main__":
    if "build" in sys.argv: build()
    if "check" in sys.argv: sys.exit(0 if fresh_check() else 1)
