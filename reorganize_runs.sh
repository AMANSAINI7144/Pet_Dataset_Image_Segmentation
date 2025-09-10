#!/usr/bin/env bash
# reorganize_runs.sh
# Safe reorganizer: finds run_* tokens and converts loose files into structured runs/<run>/ {checkpoints,results,logs,models,other}
# Usage:
#   ./reorganize_runs.sh            # dry-run (shows what would be moved)
#   ./reorganize_runs.sh --apply    # actually move files
#   ./reorganize_runs.sh --root /path/to/project --apply

set -euo pipefail

# Defaults
ROOT="$(pwd)"
DRY_RUN=1
APPLY=0
MAXDEPTH=6

function usage() {
    cat <<EOF
Usage: $0 [--root PATH] [--apply] [--help]
  --root PATH   project root to scan (default: current directory)
  --apply       actually move files; without this flag the script runs in DRY-RUN mode
  --help        show this message
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        --apply) APPLY=1; DRY_RUN=0; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

echo "Project root: $ROOT"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "MODE: DRY-RUN (no files will be moved). Add --apply to perform the moves."
else
    echo "MODE: APPLY (will move files)."
fi
echo

# find tokens from existing run directories (runs/run_*) first
declare -A TOKENS
if [ -d "$ROOT/runs" ]; then
    for d in "$ROOT"/runs/run_*; do
        if [ -d "$d" ]; then
            token=$(basename "$d")
            TOKENS["$token"]=1
        fi
    done
fi

# Find any file/dir that contains 'run_' in name and extract token run_YYYYMMDD(-HHMMSS)
# We'll search up to MAXDEPTH to avoid scanning too deep
while IFS= read -r path; do
    name=$(basename "$path")
    # extract token like run_20250902 or run_20250902-123456
    token=$(echo "$name" | grep -oE 'run_[0-9]{8}(-[0-9]{6})?' || true)
    if [ -n "$token" ]; then
        TOKENS["$token"]=1
    fi
done < <(find "$ROOT" -maxdepth $MAXDEPTH -iname "*run_*" -print 2>/dev/null || true)

# If no tokens found, exit
if [ ${#TOKENS[@]} -eq 0 ]; then
    echo "No run_* tokens found under $ROOT (searched depth $MAXDEPTH). Nothing to do."
    exit 0
fi

echo "Found runs:"
for k in "${!TOKENS[@]}"; do
    echo "  - $k"
done
echo

# For each token, create structure and move related files
for token in "${!TOKENS[@]}"; do
    echo "Processing token: $token"
    TARGET="$ROOT/runs/$token"
    CHECKPOINTS_DIR="$TARGET/checkpoints"
    RESULTS_DIR="$TARGET/results"
    LOGS_DIR="$TARGET/logs"
    MODELS_DIR="$TARGET/models"
    OTHER_DIR="$TARGET/other"

    # create directories (dry-run shows what would be created)
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "DRY-RUN: mkdir -p \"$CHECKPOINTS_DIR\" \"$RESULTS_DIR\" \"$LOGS_DIR\" \"$MODELS_DIR\" \"$OTHER_DIR\""
    else
        mkdir -p "$CHECKPOINTS_DIR" "$RESULTS_DIR" "$LOGS_DIR" "$MODELS_DIR" "$OTHER_DIR"
    fi

    # find files/dirs that include the token
    # limit to MAXDEPTH to keep it reasonably quick
    while IFS= read -r f; do
        # Skip the already-organized target path
        if [[ "$f" == "$TARGET"* ]]; then
            continue
        fi

        bname=$(basename "$f")
        dest="$OTHER_DIR"

        # Decide destination
        if [ -d "$f" ]; then
            # directories: common names -> put in checkpoints or logs
            if echo "$bname" | grep -qi "checkpoint"; then dest="$CHECKPOINTS_DIR"
            elif echo "$bname" | grep -qi "checkpoints"; then dest="$CHECKPOINTS_DIR"
            elif echo "$bname" | grep -qi "result"; then dest="$RESULTS_DIR"
            elif echo "$bname" | grep -qi "log"; then dest="$LOGS_DIR"
            else dest="$OTHER_DIR"
            fi
        else
            # files: choose by extension / name
            ext="${bname##*.}"
            if echo "$bname" | grep -qi "events.out"; then
                dest="$LOGS_DIR"
            elif [[ "$ext" == "pth" || "$ext" == "pt" ]] || echo "$bname" | grep -qi "checkpoint"; then
                dest="$CHECKPOINTS_DIR"
            elif [[ "$ext" =~ ^(png|jpg|jpeg)$ ]]; then
                dest="$RESULTS_DIR"
            elif [[ "$ext" == "log" || "$ext" == "csv" ]] || echo "$bname" | grep -qi "training.log"; then
                dest="$LOGS_DIR"
            elif echo "$bname" | grep -qi "model"; then
                dest="$MODELS_DIR"
            else
                dest="$OTHER_DIR"
            fi
        fi

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "DRY-RUN: mv -n \"$f\" \"$dest/\""
        else
            # create destination folder (in case)
            mkdir -p "$dest"
            # attempt to move, do not overwrite existing files (mv -n)
            if mv -n "$f" "$dest/"; then
                echo "Moved: $f -> $dest/"
            else
                echo "Skipped (exists?): $f -> $dest/"
            fi
        fi

    done < <(find "$ROOT" -maxdepth $MAXDEPTH -iname "*${token}*" -print 2>/dev/null || true)

    echo "Finished token: $token"
    echo
done

echo "All tokens processed. If you ran without --apply everything above was a DRY-RUN (no changes)."
if [[ $DRY_RUN -eq 1 ]]; then
    echo "To actually perform the reorganization, run this script with --apply"
fi
