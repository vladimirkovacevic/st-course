#!/bin/bash
###############################################################################
# docker_run.sh
#
# Run the BANKSY spatial transcriptomics pipeline inside a Docker container.
#
# USAGE:
#   ./docker_run.sh <input.h5ad> <output_dir> <run_banksy.R>
#
# ARGUMENTS:
#   1) <input.h5ad>
#      Path to the input AnnData (.h5ad) file containing spatial transcriptomics
#      data. This file is mounted read-only into the container.
#
#   2) <output_dir>
#      Path to an existing or new directory on the host where BANKSY results
#      will be written. This directory is mounted read–write.
#
#   3) <run_banksy.R>
#      Path to the R script that runs the BANKSY workflow. This script is mounted
#      read-only and executed inside the container via Rscript.
#
# EXAMPLE:
#   ./docker_run.sh \
#     ./data/visium_sample.h5ad \
#     ./results/banksy \
#     ./scripts/run_banksy.R
#
# NOTES:
# - The container is forced to run as linux/amd64 to ensure compatibility on
#   macOS (Apple Silicon) and other non-amd64 hosts.
# - The container is removed automatically after execution (--rm).
###############################################################################

# set -euo pipefail: exit on any error, forbid use of unset variables, 
# and fail pipelines if any command fails (prevents silent failures)
set -euo pipefail

# -----------------------------
# Argument validation
# -----------------------------
if [ "$#" -ne 3 ]; then
    echo "ERROR: Invalid number of arguments."
    echo "USAGE: $0 <input.h5ad> <output_dir> <run_banksy.R>"
    exit 1
fi

INPUT_H5AD="$1"
OUTPUT_DIR="$2"
BANKSY_SCRIPT="$3"

# -----------------------------
# Sanity checks
# -----------------------------
if [ ! -f "$INPUT_H5AD" ]; then
    echo "ERROR: Input file not found: $INPUT_H5AD"
    exit 1
fi

if [ ! -f "$BANKSY_SCRIPT" ]; then
    echo "ERROR: R script not found: $BANKSY_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# -----------------------------
# Run Docker container
# -----------------------------
docker run --rm \
  --platform linux/amd64 \
  -v "$INPUT_H5AD:/data/input.h5ad:ro" \
  -v "$OUTPUT_DIR:/output" \
  -v "$BANKSY_SCRIPT:/scripts/run_banksy.R:ro" \
  almahmoud/bioc2024-banksy:manual \
  Rscript /scripts/run_banksy.R

# -----------------------------
# Completion message
# -----------------------------
echo "BANKSY run completed successfully."
echo "Results written to: $OUTPUT_DIR"
