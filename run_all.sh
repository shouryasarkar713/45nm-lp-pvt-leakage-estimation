#!/usr/bin/env bash
# Full regeneration of every deliverable.  Usage:  bash run_all.sh [--reuse] [--skip "stageA stageB"] [--quick-validate]
#   --reuse   : a stage whose completion marker (logs/stages/<name>.done) exists is reported as REUSED and not re-executed
#   --skip    : stages listed are reported as SKIPPED (their outputs must already exist)
# Requires python>=3.10 + requirements.txt, ngspice-47 on PATH (or $NGSPICE, $NGSPICE_LD_LIBRARY_PATH).  Any failing stage stops the run (exit code kept).
# Approximate cost on 16 cores: data generation ~15 min, training ~1-2 h, circuit SPICE ~40-60 min.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; export DVD_ROOT="$ROOT"; cd "$ROOT/src"; PY=${PY:-python3}
REUSE=0; SKIP=""; while [ $# -gt 0 ]; do case "$1" in --reuse) REUSE=1;; --skip) SKIP="$2"; shift;; esac; shift; done
LOG="$ROOT/logs/run_all"; mkdir -p "$LOG" "$ROOT/logs/stages"; SUM="$LOG/stage_summary.tsv"; : > "$SUM"; printf "stage\tstatus\tseconds\texit_code\n" >> "$SUM"
stage() {  # stage <name> <command...>
  local name="$1"; shift
  if [[ " $SKIP " == *" $name "* ]]; then echo "[SKIPPED] $name"; printf "%s\tSKIPPED\t0\t0\n" "$name" >> "$SUM"; return 0; fi
  if [ "$REUSE" = 1 ] && [ -f "$ROOT/logs/stages/$name.done" ]; then echo "[REUSED ] $name"; printf "%s\tREUSED\t0\t0\n" "$name" >> "$SUM"; return 0; fi
  echo "[RUNNING] $name"; local t0=$(date +%s); "$@" > "$LOG/$name.log" 2>&1; local rc=$?; local dt=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ]; then printf "%s\tFAILED\t%s\t%s\n" "$name" "$dt" "$rc" >> "$SUM"; echo "[FAILED ] $name (exit $rc) - see $LOG/$name.log"; exit $rc; fi
  touch "$ROOT/logs/stages/$name.done"; printf "%s\tEXECUTED\t%s\t0\n" "$name" "$dt" >> "$SUM"; echo "[OK     ] $name (${dt}s)"
}
stage 01_environment      $PY check_env.py
stage 02_model_card       $PY -c "from config import *; import sys; h=sha256(MODEL_CARD); print(h); sys.exit(0 if h=='e2423b27f22e994cdf8aeaa116c25114ee5166c19e35e7bae4a3a03214d512be' else 1)"
stage 03_corners          $PY corners.py
stage 04_base_data        $PY gen_transistor.py base
stage 04b_supplementary   bash -c "$PY gen_transistor.py extra uniform 100000 1 && $PY gen_transistor.py extra gatecorner 100000 2 && $PY gen_transistor.py extra nearzero 60000 3 && $PY gen_twin.py ../data/base/NMOS_ON.csv ../data/base/NMOS_OFF.csv ../data/base/PMOS_ON.csv ../data/base/PMOS_OFF.csv ../data/extra/*_?.csv"
stage 05_data_check       $PY data_check.py
stage 06_train            $PY train_models.py fit
stage 07_train_manifest   $PY make_train_manifest.py
stage 08_test_eval        bash -c "$PY train_models.py test && $PY test_stats_extra.py"
stage 09_gate_spice       bash -c "$PY gen_gate.py official && $PY gen_gate.py dev"
stage 10_gate_ml_eval     $PY gate_run.py
stage 11_circuit_spice    $PY gen_circuit.py
stage 12_circuit_ml_eval  $PY circuit_run.py
stage 13_error_decomposition bash -c "$PY verify_definitions.py && $PY oracle_check.py && $PY dev_ablation.py"
stage 14_solver_audit     bash -c "$PY solver_warning_audit.py && $PY ladder_consistency.py && $PY dc_branch_probe.py"
stage 15_netlists         $PY make_netlists.py
stage 16_report_docx      $PY make_report.py
stage 17_report_pdf       bash -c "cd ../report && soffice --headless --convert-to pdf Project1_Report_45nmLP_1V_ParityChecker.docx && pdfinfo Project1_Report_45nmLP_1V_ParityChecker.pdf | grep Pages"
stage 18_report_numbers   $PY verify_report.py
stage 19_package_build    $PY make_packages.py build      # both archives + package-specific manifests (computed from the packaged files)
stage 20_fresh_validation $PY make_packages.py check      # extracts both archives to temp dirs, verifies every manifest entry, runs validate_all.py inside each
echo "== stage summary =="; cat "$SUM"
