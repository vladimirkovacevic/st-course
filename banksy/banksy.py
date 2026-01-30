#!/usr/bin/env python3
"""
BANKSY Spatial Transcriptomics Clustering Pipeline
===================================================

This script runs BANKSY (Building Aggregates with a Neighborhood Kernel and
Spatial Yardstick) analysis on spatial transcriptomics data using the
almahmoud/bioc2024-banksy Docker container.

BANKSY unifies cell typing and tissue domain segmentation by incorporating
spatial neighborhood information into the clustering process.

Author: Hospital Bioinformatics Team
Version: 1.0.0
License: MIT

Usage:
    python banksy.py --input <path_to_h5ad> [options]

Requirements:
    - Docker installed and running
    - almahmoud/bioc2024-banksy:manual Docker image
    - Input data in h5ad format with spatial coordinates

References:
    - Singhal et al. (2024) Nature Genetics. BANKSY unifies cell typing and
      tissue domain segmentation for scalable spatial omics data analysis.
    - https://prabhakarlab.github.io/Banksy/
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Docker image configuration
DOCKER_IMAGE = "almahmoud/bioc2024-banksy:manual"
DOCKER_PLATFORM = "linux/amd64"

# Default BANKSY parameters
DEFAULT_LAMBDA = [0.0, 0.2]  # 0=non-spatial, 0.2=cell-typing, 0.8=domain
DEFAULT_K_GEOM = [15, 30]    # Neighbors for mean and Gabor filter
DEFAULT_RESOLUTION = 0.8    # Leiden clustering resolution
DEFAULT_N_PCS = 20          # Number of principal components

# Logging configuration
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_file: Optional[Path] = None, verbose: bool = False) -> logging.Logger:
    """
    Configure logging with both console and file handlers.

    Args:
        log_file: Optional path to log file. If None, logs only to console.
        verbose: If True, set log level to DEBUG; otherwise INFO.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("BANKSY")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_docker_available(logger: logging.Logger) -> bool:
    """
    Check if Docker is installed and running.

    Args:
        logger: Logger instance for output.

    Returns:
        True if Docker is available, False otherwise.
    """
    logger.info("Validating Docker availability...")

    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            logger.error("Docker command failed: %s", result.stderr)
            return False

        docker_version = result.stdout.strip()
        logger.info("Docker found: %s", docker_version)

        # Check if Docker daemon is running
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            logger.error("Docker daemon not running. Please start Docker.")
            return False

        logger.info("Docker daemon is running")
        return True

    except FileNotFoundError:
        logger.error("Docker not found. Please install Docker.")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Docker command timed out")
        return False
    except Exception as e:
        logger.error("Unexpected error checking Docker: %s", str(e))
        return False


def validate_docker_image(logger: logging.Logger) -> bool:
    """
    Check if the BANKSY Docker image is available locally.

    Args:
        logger: Logger instance for output.

    Returns:
        True if image is available, False otherwise.
    """
    logger.info("Checking for BANKSY Docker image: %s", DOCKER_IMAGE)

    try:
        result = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            logger.error("Failed to query Docker images: %s", result.stderr)
            return False

        if not result.stdout.strip():
            logger.warning("Docker image not found locally. Attempting to pull...")
            return pull_docker_image(logger)

        logger.info("Docker image found: %s", DOCKER_IMAGE)
        return True

    except Exception as e:
        logger.error("Error checking Docker image: %s", str(e))
        return False


def pull_docker_image(logger: logging.Logger) -> bool:
    """
    Pull the BANKSY Docker image from Docker Hub.

    Args:
        logger: Logger instance for output.

    Returns:
        True if pull successful, False otherwise.
    """
    logger.info("Pulling Docker image %s (this may take several minutes)...", DOCKER_IMAGE)

    try:
        result = subprocess.run(
            ["docker", "pull", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout for large images
        )

        if result.returncode != 0:
            logger.error("Failed to pull Docker image: %s", result.stderr)
            return False

        logger.info("Docker image pulled successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Docker pull timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error("Error pulling Docker image: %s", str(e))
        return False


def validate_input_file(input_path: Path, logger: logging.Logger) -> bool:
    """
    Validate the input h5ad file exists and has correct extension.

    Args:
        input_path: Path to input h5ad file.
        logger: Logger instance for output.

    Returns:
        True if file is valid, False otherwise.
    """
    logger.info("Validating input file: %s", input_path)

    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return False

    if not input_path.is_file():
        logger.error("Input path is not a file: %s", input_path)
        return False

    if input_path.suffix.lower() != ".h5ad":
        logger.warning("Input file does not have .h5ad extension: %s", input_path.suffix)

    # Check file size
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    logger.info("Input file size: %.2f MB", file_size_mb)

    if file_size_mb < 0.001:
        logger.error("Input file appears to be empty")
        return False

    logger.info("Input file validation passed")
    return True


def validate_output_directory(output_dir: Path, logger: logging.Logger) -> bool:
    """
    Validate and create output directory if needed.

    Args:
        output_dir: Path to output directory.
        logger: Logger instance for output.

    Returns:
        True if directory is ready, False otherwise.
    """
    logger.info("Preparing output directory: %s", output_dir)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test write permissions
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()

        logger.info("Output directory ready")
        return True

    except PermissionError:
        logger.error("No write permission for output directory: %s", output_dir)
        return False
    except Exception as e:
        logger.error("Error preparing output directory: %s", str(e))
        return False

# =============================================================================
# R SCRIPT GENERATION
# =============================================================================

def generate_r_script(
    input_file: str,
    output_dir: str,
    lambda_values: list,
    k_geom: list,
    resolution: float,
    n_pcs: int,
    sample_name: str,
    logger: logging.Logger
) -> str:
    """
    Generate the R script for BANKSY analysis.

    Args:
        input_file: Path to input h5ad file (inside container).
        output_dir: Path to output directory (inside container).
        lambda_values: Lambda parameters for BANKSY.
        k_geom: k values for neighborhood computation.
        resolution: Leiden clustering resolution.
        n_pcs: Number of principal components.
        sample_name: Sample name for output files.
        logger: Logger instance for output.

    Returns:
        R script as a string.
    """
    logger.info("Generating BANKSY R analysis script...")

    # Format lambda and k_geom for R
    lambda_r = "c(" + ", ".join(str(l) for l in lambda_values) + ")"
    k_geom_r = "c(" + ", ".join(str(k) for k in k_geom) + ")"

    r_script = f'''#!/usr/bin/env Rscript
# =============================================================================
# BANKSY Spatial Transcriptomics Analysis
# =============================================================================
# Automatically generated script for running BANKSY analysis
# Generated: {datetime.datetime.now().isoformat()}
# Sample: {sample_name}
# =============================================================================

# -----------------------------------------------------------------------------
# 1. SETUP AND CONFIGURATION
# -----------------------------------------------------------------------------

cat("\\n========================================\\n")
cat("BANKSY SPATIAL CLUSTERING ANALYSIS\\n")
cat("========================================\\n\\n")

start_time <- Sys.time()
cat("Analysis started:", format(start_time, "%Y-%m-%d %H:%M:%S"), "\\n\\n")

# Suppress warnings for cleaner output
options(warn = -1)

# Load required libraries
cat("Loading required packages...\\n")

suppressPackageStartupMessages({{
    library(Banksy)
    library(SpatialExperiment)
    library(SummarizedExperiment)
    library(SingleCellExperiment)
    library(scuttle)
    library(scater)
    library(rhdf5)
    library(Matrix)
    library(ggplot2)
    library(patchwork)
    library(data.table)
}})

cat("Packages loaded successfully\\n\\n")

# Configuration parameters
config <- list(
    input_file = "{input_file}",
    output_dir = "{output_dir}",
    sample_name = "{sample_name}",
    lambda = {lambda_r},
    k_geom = {k_geom_r},
    resolution = {resolution},
    n_pcs = {n_pcs}
)

cat("Configuration:\\n")
cat("  Input file:", config$input_file, "\\n")
cat("  Output directory:", config$output_dir, "\\n")
cat("  Lambda values:", paste(config$lambda, collapse=", "), "\\n")
cat("  k_geom values:", paste(config$k_geom, collapse=", "), "\\n")
cat("  Clustering resolution:", config$resolution, "\\n")
cat("  Number of PCs:", config$n_pcs, "\\n\\n")

# Create output directory
dir.create(config$output_dir, showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 2. DATA LOADING (using rhdf5 directly for speed)
# -----------------------------------------------------------------------------

cat("Loading h5ad data using rhdf5...\\n")

tryCatch({{
    # Read h5ad file structure
    h5_contents <- h5ls(config$input_file)
    cat("  H5AD file structure detected\\n")

    # Read expression matrix (X)
    # Check if X is stored as sparse or dense
    x_group <- h5_contents[h5_contents$name == "X" | h5_contents$group == "/X", ]

    if (any(h5_contents$name == "data" & h5_contents$group == "/X")) {{
        # Sparse matrix format (CSR/CSC)
        cat("  Reading sparse matrix...\\n")
        data_vec <- h5read(config$input_file, "/X/data")
        indices <- h5read(config$input_file, "/X/indices")
        indptr <- h5read(config$input_file, "/X/indptr")

        # Get dimensions from obs and var
        obs_names <- h5read(config$input_file, "/obs/_index")
        # Try different possible gene name columns
        var_names <- tryCatch({{
            h5read(config$input_file, "/var/_index")
        }}, error = function(e) {{
            tryCatch({{
                h5read(config$input_file, "/var/Symbol")
            }}, error = function(e2) {{
                h5read(config$input_file, "/var/gene_ids")
            }})
        }})
        n_obs <- length(obs_names)
        n_var <- length(var_names)
        cat("  n_obs:", n_obs, "n_var:", n_var, "\\n")

        # Create sparse matrix (CSR format -> need to transpose for R)
        # Ensure data is a proper numeric vector
        data_vec <- as.numeric(data_vec)
        indices <- as.integer(indices)
        indptr <- as.integer(indptr)

        expr_matrix <- sparseMatrix(
            i = indices + 1L,  # 0-indexed to 1-indexed
            p = indptr,
            x = data_vec,
            dims = c(n_var, n_obs)  # genes x cells after transpose
        )
    }} else {{
        # Dense matrix
        cat("  Reading dense matrix...\\n")
        expr_matrix <- t(h5read(config$input_file, "/X"))
    }}

    cat("  Matrix dimensions:", nrow(expr_matrix), "genes x", ncol(expr_matrix), "cells\\n")

    # Set row and column names (already read above)
    colnames(expr_matrix) <- obs_names
    rownames(expr_matrix) <- var_names

    cat("  Cells/spots loaded:", ncol(expr_matrix), "\\n")
    cat("  Genes loaded:", nrow(expr_matrix), "\\n")

    # Read spatial coordinates from obsm/spatial
    spatial_coords <- NULL

    if (any(h5_contents$group == "/obsm" & h5_contents$name == "spatial")) {{
        spatial_coords <- h5read(config$input_file, "/obsm/spatial")
        cat("  Raw spatial coords dim:", paste(dim(spatial_coords), collapse=" x "), "\\n")
        if (is.matrix(spatial_coords)) {{
            # Data is stored as features x cells, transpose to cells x features
            if (nrow(spatial_coords) < ncol(spatial_coords)) {{
                spatial_coords <- t(spatial_coords)
            }}
            # Take only first 2 columns (x, y)
            if (ncol(spatial_coords) > 2) {{
                spatial_coords <- spatial_coords[, 1:2]
            }}
        }}
        cat("  Spatial coordinates found in obsm/spatial\\n")
    }} else if (any(h5_contents$group == "/obsm" & h5_contents$name == "X_spatial")) {{
        spatial_coords <- h5read(config$input_file, "/obsm/X_spatial")
        if (is.matrix(spatial_coords)) {{
            if (nrow(spatial_coords) < ncol(spatial_coords)) {{
                spatial_coords <- t(spatial_coords)
            }}
            if (ncol(spatial_coords) > 2) {{
                spatial_coords <- spatial_coords[, 1:2]
            }}
        }}
        cat("  Spatial coordinates found in obsm/X_spatial\\n")
    }} else {{
        # Try to find coordinates in obs columns
        obs_cols <- h5_contents[h5_contents$group == "/obs", "name"]
        x_candidates <- c("x", "X", "x_coord", "spatial_x", "array_col", "imagecol", "x_centroid")
        y_candidates <- c("y", "Y", "y_coord", "spatial_y", "array_row", "imagerow", "y_centroid")

        x_col <- NULL
        y_col <- NULL

        for (xname in x_candidates) {{
            if (xname %in% obs_cols) {{
                x_col <- xname
                break
            }}
        }}

        for (yname in y_candidates) {{
            if (yname %in% obs_cols) {{
                y_col <- yname
                break
            }}
        }}

        if (!is.null(x_col) && !is.null(y_col)) {{
            x_vals <- h5read(config$input_file, paste0("/obs/", x_col))
            y_vals <- h5read(config$input_file, paste0("/obs/", y_col))
            spatial_coords <- cbind(x_vals, y_vals)
            cat("  Spatial coordinates found in obs:", x_col, "/", y_col, "\\n")
        }}
    }}

    if (is.null(spatial_coords)) {{
        stop("No spatial coordinates found in h5ad file. Checked obsm/spatial, obsm/X_spatial, and obs columns.")
    }}

    # Ensure spatial_coords has correct dimensions
    if (nrow(spatial_coords) != ncol(expr_matrix)) {{
        if (ncol(spatial_coords) == ncol(expr_matrix)) {{
            spatial_coords <- t(spatial_coords)
        }}
    }}

    cat("  Spatial coordinates shape:", nrow(spatial_coords), "x", ncol(spatial_coords), "\\n")

    # Create SpatialExperiment object
    cat("  Creating SpatialExperiment object...\\n")

    spe <- SpatialExperiment(
        assays = list(counts = expr_matrix),
        spatialCoords = spatial_coords
    )

    cat("  SpatialExperiment created successfully\\n\\n")

    # Close h5 file connections
    h5closeAll()

}}, error = function(e) {{
    h5closeAll()
    cat("ERROR loading data:", conditionMessage(e), "\\n")
    cat("Stack trace:\\n")
    traceback()
    quit(status = 1)
}})

# -----------------------------------------------------------------------------
# 3. QUALITY CONTROL
# -----------------------------------------------------------------------------

cat("Performing quality control...\\n")

# Calculate QC metrics
qc_metrics <- perCellQCMetrics(spe)
colData(spe) <- cbind(colData(spe), qc_metrics)

cat("  Total UMI range:", range(qc_metrics$sum), "\\n")
cat("  Genes detected range:", range(qc_metrics$detected), "\\n")

# Filter cells based on QC metrics
# Use adaptive thresholds
sum_threshold <- quantile(qc_metrics$sum, c(0.01, 0.99))
detected_threshold <- quantile(qc_metrics$detected, c(0.01, 0.99))

keep_cells <- (qc_metrics$sum >= sum_threshold[1]) &
              (qc_metrics$sum <= sum_threshold[2]) &
              (qc_metrics$detected >= detected_threshold[1]) &
              (qc_metrics$detected <= detected_threshold[2])

n_removed <- sum(!keep_cells)
cat("  Cells removed by QC:", n_removed, "(", round(100 * n_removed / ncol(spe), 1), "% )\\n")

spe <- spe[, keep_cells]
cat("  Cells after QC:", ncol(spe), "\\n\\n")

# Filter genes - keep genes expressed in at least 1% of cells
min_cells <- max(10, floor(ncol(spe) * 0.01))
gene_counts <- rowSums(assay(spe, "counts") > 0)
keep_genes <- gene_counts >= min_cells

n_genes_removed <- sum(!keep_genes)
cat("  Genes removed (detected in <", min_cells, "cells):", n_genes_removed, "\\n")

spe <- spe[keep_genes, ]
cat("  Genes after filtering:", nrow(spe), "\\n\\n")

# -----------------------------------------------------------------------------
# 4. NORMALIZATION
# -----------------------------------------------------------------------------

cat("Normalizing data...\\n")

# Library size normalization
spe <- computeLibraryFactors(spe)
assay(spe, "normcounts") <- normalizeCounts(spe, log = FALSE)
assay(spe, "lognormcounts") <- log1p(assay(spe, "normcounts"))

cat("  Normalization complete\\n\\n")

# -----------------------------------------------------------------------------
# 5. BANKSY ANALYSIS
# -----------------------------------------------------------------------------

cat("Running BANKSY analysis...\\n")
cat("  Computing neighborhood features...\\n")

# Compute BANKSY neighborhood features
spe <- computeBanksy(
    spe,
    assay_name = "normcounts",
    compute_agf = TRUE,
    k_geom = config$k_geom
)

cat("  Running BANKSY PCA...\\n")

# Run BANKSY PCA
set.seed(42)  # For reproducibility
spe <- runBanksyPCA(
    spe,
    use_agf = TRUE,
    lambda = config$lambda,
    npcs = config$n_pcs
)

cat("  Running BANKSY UMAP...\\n")

# Run BANKSY UMAP
spe <- runBanksyUMAP(
    spe,
    use_agf = TRUE,
    lambda = config$lambda
)

cat("  Clustering with BANKSY...\\n")

# Cluster using BANKSY
spe <- clusterBanksy(
    spe,
    use_agf = TRUE,
    lambda = config$lambda,
    resolution = config$resolution
)

# Connect clusters across lambda values (only if multiple lambdas)
if (length(config$lambda) > 1) {{
    spe <- connectClusters(spe)
    cat("  Clusters connected across lambda values\\n")
}} else {{
    cat("  Single lambda value - skipping connectClusters\\n")
}}

cat("  BANKSY analysis complete\\n\\n")

# -----------------------------------------------------------------------------
# 6. EXTRACT AND SUMMARIZE RESULTS
# -----------------------------------------------------------------------------

cat("Extracting results...\\n")

# Get cluster assignments for each lambda value
cluster_cols <- grep("^clust_", colnames(colData(spe)), value = TRUE)
cat("  Cluster columns found:", paste(cluster_cols, collapse=", "), "\\n")

# Summarize clusters
for (cc in cluster_cols) {{
    n_clusters <- length(unique(colData(spe)[[cc]]))
    cat("  ", cc, ": ", n_clusters, " clusters\\n", sep = "")
}}

# Create summary table
results_summary <- data.frame(
    sample = config$sample_name,
    n_cells = ncol(spe),
    n_genes = nrow(spe),
    lambda_values = paste(config$lambda, collapse = ","),
    k_geom_values = paste(config$k_geom, collapse = ","),
    resolution = config$resolution,
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)

# Add cluster counts
for (cc in cluster_cols) {{
    results_summary[[paste0("n_", cc)]] <- length(unique(colData(spe)[[cc]]))
}}

cat("\\n")

# -----------------------------------------------------------------------------
# 7. SAVE OUTPUTS
# -----------------------------------------------------------------------------

cat("Saving outputs...\\n")

# Save processed SpatialExperiment object as RDS
rds_file <- file.path(config$output_dir, paste0(config$sample_name, "_banksy_results.rds"))
saveRDS(spe, rds_file)
cat("  RDS object saved:", rds_file, "\\n")

# Save cluster assignments as CSV
cluster_data <- as.data.frame(colData(spe)[, c(cluster_cols, "sum", "detected")])
cluster_data$cell_id <- colnames(spe)
cluster_data$x <- spatialCoords(spe)[, 1]
cluster_data$y <- spatialCoords(spe)[, 2]

csv_file <- file.path(config$output_dir, paste0(config$sample_name, "_clusters.csv"))
write.csv(cluster_data, csv_file, row.names = FALSE)
cat("  Cluster assignments saved:", csv_file, "\\n")

# Save summary
summary_file <- file.path(config$output_dir, paste0(config$sample_name, "_summary.csv"))
write.csv(results_summary, summary_file, row.names = FALSE)
cat("  Summary saved:", summary_file, "\\n")

# -----------------------------------------------------------------------------
# 8. GENERATE VISUALIZATIONS
# -----------------------------------------------------------------------------

cat("Generating visualizations...\\n")

# Create spatial plots for each lambda value
for (cc in cluster_cols) {{

    plot_data <- data.frame(
        x = spatialCoords(spe)[, 1],
        y = spatialCoords(spe)[, 2],
        cluster = factor(colData(spe)[[cc]])
    )

    p <- ggplot(plot_data, aes(x = x, y = y, color = cluster)) +
        geom_point(size = 0.5, alpha = 0.7) +
        theme_minimal() +
        theme(
            legend.position = "right",
            plot.title = element_text(hjust = 0.5, face = "bold"),
            panel.grid = element_blank(),
            axis.text = element_blank(),
            axis.ticks = element_blank()
        ) +
        labs(
            title = paste("BANKSY Spatial Clustering -", cc),
            x = "Spatial X",
            y = "Spatial Y",
            color = "Cluster"
        ) +
        coord_fixed() +
        scale_color_discrete()

    plot_file <- file.path(config$output_dir, paste0(config$sample_name, "_", cc, "_spatial.png"))
    ggsave(plot_file, p, width = 10, height = 8, dpi = 150)
    cat("  Spatial plot saved:", plot_file, "\\n")
}}

# Create UMAP plots
umap_names <- reducedDimNames(spe)[grep("^UMAP", reducedDimNames(spe))]

for (umap_name in umap_names) {{

    umap_coords <- reducedDim(spe, umap_name)

    # Get corresponding cluster column
    lambda_str <- gsub("UMAP_", "", umap_name)
    cc <- paste0("clust_", lambda_str)

    if (cc %in% cluster_cols) {{
        plot_data <- data.frame(
            UMAP1 = umap_coords[, 1],
            UMAP2 = umap_coords[, 2],
            cluster = factor(colData(spe)[[cc]])
        )

        p <- ggplot(plot_data, aes(x = UMAP1, y = UMAP2, color = cluster)) +
            geom_point(size = 0.5, alpha = 0.7) +
            theme_minimal() +
            theme(
                legend.position = "right",
                plot.title = element_text(hjust = 0.5, face = "bold")
            ) +
            labs(
                title = paste("BANKSY UMAP -", lambda_str),
                color = "Cluster"
            ) +
            scale_color_discrete()

        plot_file <- file.path(config$output_dir, paste0(config$sample_name, "_", umap_name, ".png"))
        ggsave(plot_file, p, width = 8, height = 6, dpi = 150)
        cat("  UMAP plot saved:", plot_file, "\\n")
    }}
}}

# -----------------------------------------------------------------------------
# 9. COMPLETION
# -----------------------------------------------------------------------------

end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = "mins")

cat("\\n========================================\\n")
cat("ANALYSIS COMPLETE\\n")
cat("========================================\\n")
cat("Duration:", round(as.numeric(duration), 2), "minutes\\n")
cat("Output directory:", config$output_dir, "\\n")
cat("\\n")

# Write completion marker
completion_file <- file.path(config$output_dir, ".banksy_complete")
writeLines(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), completion_file)

cat("SUCCESS: BANKSY analysis completed successfully\\n")
'''

    logger.info("R script generated (%d characters)", len(r_script))
    return r_script

# =============================================================================
# DOCKER EXECUTION
# =============================================================================

def run_docker_analysis(
    r_script: str,
    input_path: Path,
    output_dir: Path,
    logger: logging.Logger,
    timeout: int = 3600
) -> Tuple[bool, str]:
    """
    Execute the BANKSY analysis in Docker container.

    Args:
        r_script: R script content to execute.
        input_path: Path to input h5ad file.
        output_dir: Path to output directory.
        logger: Logger instance for output.
        timeout: Maximum execution time in seconds.

    Returns:
        Tuple of (success: bool, message: str)
    """
    logger.info("Preparing Docker execution...")

    # Create temporary directory for R script
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write R script to temp directory
        r_script_path = tmpdir_path / "banksy_analysis.R"
        with open(r_script_path, 'w') as f:
            f.write(r_script)

        logger.info("R script written to: %s", r_script_path)

        # Prepare Docker command
        # Mount: input file, output directory, and script
        input_path_abs = input_path.resolve()
        output_dir_abs = output_dir.resolve()

        docker_cmd = [
            "docker", "run",
            "--rm",
            "--platform", DOCKER_PLATFORM,
            "-v", f"{input_path_abs}:/data/input.h5ad:ro",
            "-v", f"{output_dir_abs}:/output",
            "-v", f"{r_script_path}:/scripts/run_banksy.R:ro",
            DOCKER_IMAGE,
            "Rscript", "/scripts/run_banksy.R"
        ]

        logger.info("Docker command: %s", " ".join(docker_cmd))
        logger.info("Starting BANKSY analysis (this may take several minutes)...")

        try:
            # Execute Docker command
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output in real-time
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.rstrip()
                    output_lines.append(line)
                    # Log R output with appropriate level
                    if "ERROR" in line.upper():
                        logger.error("[R] %s", line)
                    elif "WARNING" in line.upper():
                        logger.warning("[R] %s", line)
                    else:
                        logger.info("[R] %s", line)

            # Wait for completion
            return_code = process.wait(timeout=timeout)

            full_output = "\n".join(output_lines)

            if return_code == 0:
                logger.info("Docker container exited successfully")
                return True, full_output
            else:
                logger.error("Docker container exited with code: %d", return_code)
                return False, full_output

        except subprocess.TimeoutExpired:
            logger.error("Docker execution timed out after %d seconds", timeout)
            process.kill()
            return False, "Execution timed out"

        except Exception as e:
            logger.error("Docker execution failed: %s", str(e))
            return False, str(e)

# =============================================================================
# RESULT VERIFICATION
# =============================================================================

def verify_results(output_dir: Path, sample_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Verify that BANKSY analysis completed successfully and outputs exist.

    Args:
        output_dir: Path to output directory.
        sample_name: Sample name used for output files.
        logger: Logger instance for output.

    Returns:
        Dictionary with verification results.
    """
    logger.info("Verifying analysis outputs...")

    results = {
        "success": False,
        "outputs": {},
        "errors": []
    }

    # Expected output files
    expected_files = {
        "rds": output_dir / f"{sample_name}_banksy_results.rds",
        "clusters_csv": output_dir / f"{sample_name}_clusters.csv",
        "summary_csv": output_dir / f"{sample_name}_summary.csv",
        "completion_marker": output_dir / ".banksy_complete"
    }

    # Check each expected file
    for file_type, file_path in expected_files.items():
        if file_path.exists():
            file_size = file_path.stat().st_size
            results["outputs"][file_type] = {
                "path": str(file_path),
                "size_bytes": file_size,
                "exists": True
            }
            logger.info("  [OK] %s: %s (%.2f KB)", file_type, file_path.name, file_size / 1024)
        else:
            results["outputs"][file_type] = {"exists": False, "path": str(file_path)}
            results["errors"].append(f"Missing output file: {file_path.name}")
            logger.warning("  [MISSING] %s: %s", file_type, file_path.name)

    # Check for plot files
    plot_files = list(output_dir.glob(f"{sample_name}_*.png"))
    results["outputs"]["plots"] = {
        "count": len(plot_files),
        "files": [str(p) for p in plot_files]
    }
    logger.info("  [OK] %d visualization plots generated", len(plot_files))

    # Determine overall success
    critical_files = ["rds", "clusters_csv", "completion_marker"]
    results["success"] = all(
        results["outputs"].get(f, {}).get("exists", False)
        for f in critical_files
    )

    if results["success"]:
        logger.info("All critical outputs verified successfully")
    else:
        logger.error("Some critical outputs are missing")
        for error in results["errors"]:
            logger.error("  - %s", error)

    return results

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for BANKSY analysis pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run BANKSY spatial transcriptomics clustering analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default parameters
    python banksy.py --input data/sample.h5ad --output results/banksy

    # Custom lambda values for domain segmentation
    python banksy.py --input data/sample.h5ad --output results/ --lambda 0.8

    # Verbose output with custom resolution
    python banksy.py --input data/sample.h5ad --output results/ --resolution 1.0 --verbose
        """
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input h5ad file with spatial transcriptomics data"
    )

    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: same as input with _banksy suffix)"
    )

    parser.add_argument(
        "--sample-name", "-n",
        type=str,
        default=None,
        help="Sample name for output files (default: derived from input filename)"
    )

    parser.add_argument(
        "--lambda",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDA,
        dest="lambda_values",
        help=f"Lambda parameter(s) for BANKSY (default: {DEFAULT_LAMBDA}). "
             "Use 0.2 for cell-typing, 0.8 for domain segmentation."
    )

    parser.add_argument(
        "--k-geom",
        type=int,
        nargs="+",
        default=DEFAULT_K_GEOM,
        help=f"k values for neighborhood computation (default: {DEFAULT_K_GEOM})"
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
        help=f"Leiden clustering resolution (default: {DEFAULT_RESOLUTION})"
    )

    parser.add_argument(
        "--n-pcs",
        type=int,
        default=DEFAULT_N_PCS,
        help=f"Number of principal components (default: {DEFAULT_N_PCS})"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Maximum execution time in seconds (default: 3600)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output (debug logging)"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: output_dir/banksy.log)"
    )

    args = parser.parse_args()

    # Resolve paths
    input_path = Path(args.input).resolve()

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = input_path.parent / f"{input_path.stem}_banksy"

    if args.sample_name:
        sample_name = args.sample_name
    else:
        sample_name = input_path.stem

    if args.log_file:
        log_file = Path(args.log_file).resolve()
    else:
        log_file = output_dir / "banksy.log"

    # Setup logging
    logger = setup_logging(log_file=log_file, verbose=args.verbose)

    # Print banner
    logger.info("=" * 60)
    logger.info("BANKSY SPATIAL TRANSCRIPTOMICS ANALYSIS")
    logger.info("=" * 60)
    logger.info("Input: %s", input_path)
    logger.info("Output: %s", output_dir)
    logger.info("Sample: %s", sample_name)
    logger.info("Lambda: %s", args.lambda_values)
    logger.info("k_geom: %s", args.k_geom)
    logger.info("Resolution: %s", args.resolution)
    logger.info("=" * 60)

    # Track overall start time
    start_time = time.time()

    # Step 1: Validate environment
    logger.info("")
    logger.info("STEP 1: Validating environment")
    logger.info("-" * 40)

    if not validate_docker_available(logger):
        logger.critical("Docker validation failed. Exiting.")
        sys.exit(1)

    if not validate_docker_image(logger):
        logger.critical("Docker image validation failed. Exiting.")
        sys.exit(1)

    # Step 2: Validate input/output
    logger.info("")
    logger.info("STEP 2: Validating input and output")
    logger.info("-" * 40)

    if not validate_input_file(input_path, logger):
        logger.critical("Input file validation failed. Exiting.")
        sys.exit(1)

    if not validate_output_directory(output_dir, logger):
        logger.critical("Output directory validation failed. Exiting.")
        sys.exit(1)

    # Step 3: Generate R script
    logger.info("")
    logger.info("STEP 3: Generating analysis script")
    logger.info("-" * 40)

    r_script = generate_r_script(
        input_file="/data/input.h5ad",
        output_dir="/output",
        lambda_values=args.lambda_values,
        k_geom=args.k_geom,
        resolution=args.resolution,
        n_pcs=args.n_pcs,
        sample_name=sample_name,
        logger=logger
    )

    # Save R script for reference
    r_script_backup = output_dir / "banksy_analysis.R"
    with open(r_script_backup, 'w') as f:
        f.write(r_script)
    logger.info("R script saved to: %s", r_script_backup)

    # Step 4: Execute analysis
    logger.info("")
    logger.info("STEP 4: Running BANKSY analysis")
    logger.info("-" * 40)

    success, output = run_docker_analysis(
        r_script=r_script,
        input_path=input_path,
        output_dir=output_dir,
        logger=logger,
        timeout=args.timeout
    )

    # Step 5: Verify results
    logger.info("")
    logger.info("STEP 5: Verifying results")
    logger.info("-" * 40)

    results = verify_results(output_dir, sample_name, logger)

    # Save results summary
    results_json = output_dir / "banksy_run_info.json"
    run_info = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "sample_name": sample_name,
        "parameters": {
            "lambda": args.lambda_values,
            "k_geom": args.k_geom,
            "resolution": args.resolution,
            "n_pcs": args.n_pcs
        },
        "docker_image": DOCKER_IMAGE,
        "success": success and results["success"],
        "duration_seconds": time.time() - start_time,
        "outputs": results["outputs"],
        "errors": results["errors"],
        "timestamp": datetime.datetime.now().isoformat()
    }

    with open(results_json, 'w') as f:
        json.dump(run_info, f, indent=2)
    logger.info("Run info saved to: %s", results_json)

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    duration_mins = (time.time() - start_time) / 60

    if success and results["success"]:
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("Duration: %.2f minutes", duration_mins)
        logger.info("Output directory: %s", output_dir)
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("ANALYSIS FAILED")
        logger.error("Duration: %.2f minutes", duration_mins)
        for error in results["errors"]:
            logger.error("  - %s", error)
        logger.info("=" * 60)
        logger.info("Check log file for details: %s", log_file)
        sys.exit(1)


if __name__ == "__main__":
    main()
