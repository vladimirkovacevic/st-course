#!/usr/bin/env python3
"""
Cell Type Annotation Framework for Spatial Transcriptomics
===========================================================

This script provides a unified framework for running multiple reference-based
cell type annotation tools on spatial transcriptomics data, including:
- Tangram: Spatial mapping and cell type projection
- Seurat: Label transfer (via Docker)
- CoDi: Contrastive distance annotation

The framework runs available tools, saves individual predictions, and generates
ensemble predictions via majority voting across all successful annotations.

Author: Hospital Bioinformatics Team
Version: 1.0.0
License: MIT

Usage:
    python cell_annotation_framework.py \\
        --reference path/to/reference.h5ad \\
        --query path/to/query.h5ad \\
        --annotation-field cell_type \\
        --tools all \\
        --output results/

Requirements:
    - See requirements.txt for Python dependencies
    - Docker installed for Seurat (docker pull satijalab/seurat)
    - Reference dataset with cell type annotations
    - Query dataset to be annotated

Output:
    - Individual tool predictions: {sample}_{tool}_predictions.csv
    - Merged predictions (last column = ensemble): {sample}_merged_predictions.csv
    - Summary metadata: {sample}_annotation_summary.json
    - Log file: cell_annotation_framework.log
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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# TOOL IMPORTS WITH AVAILABILITY TRACKING
# =============================================================================

# Core dependencies (always required)
try:
    import scanpy as sc
    import anndata as ad
except ImportError as e:
    print(f"ERROR: Core dependency missing: {e}")
    print("Install with: pip install scanpy anndata")
    sys.exit(1)

# Tool availability tracking
TOOLS_AVAILABLE = {
    'tangram': False,
    'seurat': False,
    'codi': False
}

# Try importing Tangram
try:
    import tangram as tg
    TOOLS_AVAILABLE['tangram'] = True
except ImportError:
    pass

# Seurat and CoDi run in Docker — availability is checked at runtime via
# check_docker_image() in Step 2 of the pipeline.

# =============================================================================
# CONFIGURATION
# =============================================================================

# Docker image names
SEURAT_DOCKER_IMAGE = "satijalab/seurat"
CODI_DOCKER_IMAGE   = "vladimirkovacevic/codi"

# Logging configuration
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Output file templates
CSV_TEMPLATE = "{sample}_{tool}_predictions.csv"
MERGED_CSV = "{sample}_merged_predictions.csv"
SUMMARY_JSON  = "{sample}_annotation_summary.json"
FIGURE_PNG    = "{sample}_annotation_panels.png"
LOG_FILE      = "cell_annotation_framework.log"

# Validation thresholds
MIN_GENE_OVERLAP_WARNING = 0.5  # Warn if gene overlap < 50%
MIN_CELL_TYPES = 2  # Minimum expected cell types in reference

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
    logger = logging.getLogger("CellAnnotation")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler (INFO level for user-facing output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console_handler)

    # File handler (DEBUG level for complete audit trail)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(file_handler)
        logger.debug(f"Logging to file: {log_file}")

    return logger

# =============================================================================
# DOCKER UTILITIES
# =============================================================================

def check_docker_available(logger: logging.Logger) -> bool:
    """
    Check if Docker is installed and running.

    Args:
        logger: Logger instance.

    Returns:
        True if Docker is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.debug(f"Docker version: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    logger.warning("Docker not found or not running")
    return False

def check_docker_image(image: str, logger: logging.Logger) -> bool:
    """
    Check if a Docker image is available locally; if not, attempt to pull it.

    The pull is attempted automatically so that users do not need to manually
    run ``docker pull`` before the pipeline.  A failed pull is logged but does
    not raise — the tool is simply marked unavailable.

    Args:
        image: Docker image name (e.g., 'satijalab/seurat').
        logger: Logger instance.

    Returns:
        True if image is available (locally or after pull), False otherwise.
    """
    try:
        result = subprocess.run(
            ['docker', 'images', '-q', image],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.debug(f"Docker image found locally: {image}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Image not found locally — attempt automatic pull
    logger.info(f"Docker image not found locally, pulling: {image}")
    pull_cmd = ['docker', 'pull', image]
    logger.info(f"Command: {' '.join(pull_cmd)}")
    try:
        pull_result = subprocess.run(
            pull_cmd,
            capture_output=True,
            text=True,
            timeout=1200  # 10-minute timeout for large images
        )
        if pull_result.returncode == 0:
            logger.info(f"Successfully pulled Docker image: {image}")
            return True
        else:
            logger.warning(f"Docker pull failed for {image}: {pull_result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Docker pull timed out for {image}")
        return False
    except Exception as e:
        logger.warning(f"Docker pull error for {image}: {e}")
        return False

# =============================================================================
# FILE-LINKING UTILITY
# =============================================================================

def link_or_copy(src: str, dst: Path, logger: logging.Logger) -> None:
    """
    Hard-link src into dst; fall back to shutil.copy if the two paths
    are on different filesystems (hard links cannot cross filesystems).

    Hard links are instant and consume zero extra disk space — the
    destination is just a new directory entry pointing to the same inode.

    Args:
        src: Source file path (original h5ad).
        dst: Destination path inside the temp directory.
        logger: Logger instance.
    """
    try:
        os.link(src, dst)
        logger.debug(f"Hard-linked: {src} -> {dst}")
    except OSError:
        # Cross-filesystem — fall back to a full copy
        logger.debug(f"Hard-link failed (cross-filesystem), copying: {src} -> {dst}")
        shutil.copy2(src, dst)

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_file_exists(file_path: str, logger: logging.Logger) -> bool:
    """
    Validate that a file exists and is readable.

    Args:
        file_path: Path to file.
        logger: Logger instance.

    Returns:
        True if valid, False otherwise.
    """
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    if not path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return False

    if not os.access(path, os.R_OK):
        logger.error(f"File not readable: {file_path}")
        return False

    # Log file size
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.debug(f"File size: {size_mb:.2f} MB")

    return True

def validate_h5ad_file(file_path: str, logger: logging.Logger,
                      backed: Optional[str] = None) -> Optional[ad.AnnData]:
    """
    Validate and load an h5ad file.

    Args:
        file_path: Path to h5ad file.
        logger: Logger instance.
        backed: If 'r', open in backed (read-only on-disk) mode so the
                expression matrix is never loaded into RAM.  Use this when
                only Docker-based tools will run and no in-process matrix
                access is needed.

    Returns:
        AnnData object if valid, None otherwise.
    """
    if not file_path.endswith('.h5ad'):
        logger.warning(f"File does not have .h5ad extension: {file_path}")

    try:
        mode_label = "backed (metadata only)" if backed else "full"
        logger.debug(f"Loading h5ad file [{mode_label}]: {file_path}")
        adata = sc.read_h5ad(file_path, backed=backed)
        logger.info(f"Loaded dataset: {adata.n_obs} cells, {adata.n_vars} genes [{mode_label}]")
        return adata
    except Exception as e:
        logger.error(f"Failed to read h5ad file: {e}", exc_info=True)
        return None

def validate_annotation_field(adata: ad.AnnData, field: str, logger: logging.Logger) -> bool:
    """
    Validate that annotation field exists in dataset.

    Args:
        adata: AnnData object.
        field: Annotation field name.
        logger: Logger instance.

    Returns:
        True if valid, False otherwise.
    """
    if field not in adata.obs.columns:
        logger.error(f"Annotation field '{field}' not found in dataset")
        logger.info(f"Available fields: {', '.join(adata.obs.columns[:20])}")
        if len(adata.obs.columns) > 20:
            logger.info(f"... and {len(adata.obs.columns) - 20} more")
        return False

    # Check for missing values
    n_missing = adata.obs[field].isna().sum()
    if n_missing > 0:
        logger.warning(f"Annotation field has {n_missing} missing values")

    # Check number of unique cell types
    cell_types = adata.obs[field].dropna().unique()
    n_types = len(cell_types)
    logger.info(f"Cell types in annotation: {n_types}")
    logger.debug(f"Cell types: {', '.join(str(ct) for ct in cell_types[:10])}")
    if n_types > 10:
        logger.debug(f"... and {n_types - 10} more")

    if n_types < MIN_CELL_TYPES:
        logger.warning(f"Only {n_types} cell type(s) found (expected >={MIN_CELL_TYPES})")

    return True

def calculate_gene_overlap(adata_ref: ad.AnnData, adata_query: ad.AnnData,
                          logger: logging.Logger) -> Tuple[List[str], float]:
    """
    Calculate gene overlap between reference and query datasets.

    Args:
        adata_ref: Reference AnnData object.
        adata_query: Query AnnData object.
        logger: Logger instance.

    Returns:
        Tuple of (common genes list, overlap percentage).
    """
    ref_genes = set(adata_ref.var_names)
    query_genes = set(adata_query.var_names)

    common_genes = sorted(ref_genes & query_genes)
    overlap_pct = len(common_genes) / len(ref_genes) if len(ref_genes) > 0 else 0.0

    logger.info(f"Gene overlap: {len(common_genes)} genes ({overlap_pct:.1%})")
    logger.debug(f"Reference genes: {len(ref_genes)}, Query genes: {len(query_genes)}")

    if overlap_pct < MIN_GENE_OVERLAP_WARNING:
        logger.warning(f"Low gene overlap ({overlap_pct:.1%}) - results may be unreliable")

    return common_genes, overlap_pct

def detect_matrix_state(adata: ad.AnnData, name: str, logger: logging.Logger) -> str:
    """Detect whether .X contains raw counts, normalised, or log-transformed data.

    Samples up to 200 cells so the heuristic stays fast on large matrices.
    Decision rules (applied in order):
        1. Any negative value                             →  'log'
           (batch-corrected / SCTransform output; never raw)
        2. All non-zero values are integers AND max ≥ 1   →  'raw'
        3. Floats with max > LOG_UPPER_BOUND (20)         →  'normalized'
           (e.g. normalize_total with target_sum = 1 e4)
        4. Floats with max ≤ 20                           →  'log'
           (log1p of 1e4-normalised data tops out at ~9; 20 is a safe ceiling)

    Args:
        adata:  AnnData whose .X will be inspected.
        name:   Label for log messages ('Reference' / 'Query').
        logger: Logger instance.

    Returns:
        One of 'raw', 'normalized', 'log', or 'unknown'.
    """
    import scipy.sparse as sp

    # Upper bound for log1p-transformed values.  log1p of the most extreme
    # gene in 1e4-normalised 10x data is ~9; we use 20 to be safe.
    LOG_UPPER_BOUND = 20.0

    n_sample = min(200, adata.n_obs)
    X_sample = adata.X[:n_sample]
    if sp.issparse(X_sample):
        X_sample = X_sample.toarray()
    X_sample = np.asarray(X_sample, dtype=np.float64)

    non_zero = X_sample[X_sample != 0]
    if len(non_zero) == 0:
        logger.warning(f"[{name}] Sampled expression matrix is all zeros — state unknown")
        return 'unknown'

    min_val    = float(X_sample.min())
    max_val    = float(non_zero.max())
    is_integer = bool(np.allclose(non_zero, np.round(non_zero), atol=1e-6))

    if min_val < 0:
        # Negative values rule out raw counts; most likely batch-corrected
        # or SCTransform residuals — treat as log-like for tool purposes.
        state = 'log'
        logger.info(f"  [{name}] Negative values detected (min={min_val:.4g}) — "
                    f"likely batch-corrected or SCTransform residuals")
    elif is_integer and max_val >= 1:
        state = 'raw'
    elif max_val > LOG_UPPER_BOUND:
        state = 'normalized'   # e.g. normalize_total(target_sum=1e4) without log1p
    else:
        state = 'log'          # log1p-transformed

    logger.info(f"  [{name}] .X state: {state}  "
                f"(min={min_val:.4g}, max={max_val:.4g}, "
                f".raw={'present' if adata.raw is not None else 'absent'})")
    return state

# =============================================================================
# ABSTRACT BASE CLASS FOR ANNOTATION TOOLS
# =============================================================================

class AnnotationTool(ABC):
    """
    Abstract base class for cell type annotation tools.

    All annotation tools must implement the run() method which returns
    predictions in a standardized DataFrame format.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize annotation tool.

        Args:
            logger: Logger instance.
        """
        self.logger = logger
        self.tool_name = self.__class__.__name__.replace('Annotator', '').lower()

    @abstractmethod
    def run(self, adata_ref: ad.AnnData, adata_query: ad.AnnData,
            annotation_field: str, common_genes: List[str],
            ref_path: str, query_path: str,
            marker_genes: Optional[List[str]] = None,
            args: Optional[argparse.Namespace] = None) -> pd.DataFrame:
        """
        Run annotation tool.

        Args:
            adata_ref: Reference AnnData (may be backed/metadata-only when only
                       Docker tools run — do not access .X in that case).
            adata_query: Query AnnData (same caveat as adata_ref).
            annotation_field: Column name in adata_ref.obs containing cell types.
            common_genes: Pre-computed gene intersection (computed once by pipeline).
            ref_path: Original file path to reference h5ad (for Docker volume mounts).
            query_path: Original file path to query h5ad (for Docker volume mounts).
            marker_genes: Pre-computed marker genes intersected with common_genes
                         (computed once by pipeline; only Tangram uses this).
            args: Parsed CLI namespace — tool-specific parameters live here with
                  tool-name prefixes (e.g. tangram_epochs, seurat_nfeatures).

        Returns:
            DataFrame with columns: [cell_id, cell_type_annotation, prediction_score]

        Raises:
            Exception: If annotation fails.
        """
        pass

# =============================================================================
# TANGRAM ANNOTATOR
# =============================================================================

class TangramAnnotator(AnnotationTool):
    """
    Tangram spatial mapping and cell type annotation.

    Uses spatial information to map single-cell reference data to spatial
    locations and project cell type annotations.
    """

    def run(self, adata_ref: ad.AnnData, adata_query: ad.AnnData,
            annotation_field: str, common_genes: List[str],
            ref_path: str, query_path: str,
            marker_genes: Optional[List[str]] = None,
            args: Optional[argparse.Namespace] = None) -> pd.DataFrame:
        """Run Tangram annotation using pre-computed marker genes.

        Tangram is the only Python-API tool — it slices AnnData directly.
        ref_path / query_path are unused here.
        """
        self.logger.info("Starting Tangram annotation")

        if marker_genes is None:
            raise ValueError("Tangram requires pre-computed marker_genes from pipeline")

        self.logger.info(f"Using {len(marker_genes)} pre-computed marker genes")

        # User-configurable parameters (CLI defaults match Tangram's recommended values)
        num_epochs    = getattr(args, 'tangram_epochs',        1000)              if args else 1000
        mode          = getattr(args, 'tangram_mode',          'cells')           if args else 'cells'
        density_prior = getattr(args, 'tangram_density_prior', 'rna_count_based') if args else 'rna_count_based'
        self.logger.info(f"Parameters — mode: {mode}, density_prior: {density_prior}, epochs: {num_epochs}")

        # Slice to marker genes only, then copy — avoids copying the full dataset
        ref = adata_ref[:, marker_genes].copy()
        query = adata_query[:, marker_genes].copy()

        # Prepare data for Tangram (sets internal metadata; genes already aligned)
        tg.pp_adatas(ref, query, genes=marker_genes)

        # Map cells to space
        self.logger.info("Mapping cells to spatial locations (this may take several minutes)")
        ad_map = tg.map_cells_to_space(
            adata_sc=ref,
            adata_sp=query,
            mode=mode,
            density_prior=density_prior,
            num_epochs=num_epochs,
            device='cpu'
        )

        # Project cell type annotations
        self.logger.info("Projecting cell type annotations")
        tg.project_cell_annotations(
            ad_map,
            query,
            annotation=annotation_field
        )

        # Extract predictions from the cell-type probability matrix that
        # project_cell_annotations stores in obsm['tangram_ct_pred'].
        # Columns = cell types, rows = spatial spots.
        ct_probs = query.obsm['tangram_ct_pred']
        predictions = pd.DataFrame({
            'cell_id':               query.obs_names,
            'cell_type_annotation':  ct_probs.idxmax(axis=1).astype(str),
            'prediction_score':      ct_probs.max(axis=1)
        })

        self.logger.info(f"Tangram completed: {len(predictions)} cells annotated")
        return predictions

# =============================================================================
# SEURAT ANNOTATOR (DOCKER)
# =============================================================================

class SeuratAnnotator(AnnotationTool):
    """
    Runs Seurat's anchor-based label transfer in a Docker container.
    """

    def run(self, adata_ref: ad.AnnData, adata_query: ad.AnnData,
            annotation_field: str, common_genes: List[str],
            ref_path: str, query_path: str,
            marker_genes: Optional[List[str]] = None,
            args: Optional[argparse.Namespace] = None) -> pd.DataFrame:
        """Run Seurat annotation via Docker.

        The satijalab/seurat image (v5.4) ships with broken HDF5 R libs and no
        SeuratDisk — it cannot read .h5ad directly.  We therefore convert both
        datasets to 10X-style MTX directories on the host (using scanpy, which
        is already loaded) and mount those read-only.  A separate writable
        mount carries the generated R script and the output CSV.
        """
        self.logger.info("Starting Seurat annotation via Docker")

        # User-configurable parameters
        nfeatures = getattr(args, 'seurat_nfeatures', 2000) if args else 2000
        dims      = getattr(args, 'seurat_dims',      30)   if args else 30
        self.logger.info(f"Parameters — nfeatures: {nfeatures}, dims: 1:{dims}")

        # Temp dir: ref_mtx/, query_mtx/, seurat_transfer.R, predictions.csv
        tmpdir = Path(tempfile.mkdtemp(prefix='seurat_'))
        self.logger.debug(f"Temporary directory: {tmpdir}")

        try:
            # --- convert h5ad → 10X MTX on the host side ---
            self._write_10x(adata_ref,  annotation_field, tmpdir / "ref_mtx")
            self._write_10x(adata_query, None,            tmpdir / "query_mtx")
            self.logger.debug("Converted both datasets to 10X MTX format")

            # --- generate R script (reads from /data/, writes to /scripts/) ---
            r_script = self._generate_seurat_script(annotation_field, nfeatures, dims)
            (tmpdir / "seurat_transfer.R").write_text(r_script)
            self.logger.debug("Generated Seurat R script")

            # Docker mounts (all inside one host dir for simplicity):
            #   /data/ref_mtx/   ← 10X matrix for reference (read-only)
            #   /data/query_mtx/ ← 10X matrix for query    (read-only)
            #   /scripts/        ← R script + output CSV   (writable)
            self.logger.info("Running Seurat in Docker container")
            cmd = [
                'docker', 'run', '--rm',
                '-v', f'{(tmpdir / "ref_mtx").absolute()}:/data/ref_mtx:ro',
                '-v', f'{(tmpdir / "query_mtx").absolute()}:/data/query_mtx:ro',
                '-v', f'{tmpdir.absolute()}:/scripts',
                SEURAT_DOCKER_IMAGE,
                'Rscript', '/scripts/seurat_transfer.R'
            ]

            self.logger.info(f"Docker command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.logger.error(f"Seurat Docker failed with exit code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Seurat execution failed: {result.stderr}")

            if result.stdout:
                self.logger.info(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                self.logger.info(f"STDERR:\n{result.stderr}")

            # Read output CSV from the temp dir
            output_csv = tmpdir / "predictions.csv"
            if not output_csv.exists():
                raise RuntimeError("Seurat did not produce output CSV")

            predictions = pd.read_csv(output_csv)

            # Validate output columns
            required_cols = ['cell_id', 'cell_type_annotation', 'prediction_score']
            if not all(col in predictions.columns for col in required_cols):
                self.logger.error(f"Invalid CSV format. Columns: {predictions.columns.tolist()}")
                raise RuntimeError("Seurat output has incorrect format")

            self.logger.info(f"Seurat completed: {len(predictions)} cells annotated")
            return predictions

        finally:
            self.logger.debug("Cleaning up temporary files")
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _write_10x(adata: ad.AnnData, annotation_field: Optional[str], out_dir: Path) -> None:
        """Write an AnnData object as a 10X-style directory (matrix.mtx.gz +
        barcodes.tsv.gz + features.tsv.gz).  If annotation_field is provided,
        also save a barcodes_annotation.csv so the R script can attach labels.

        Uses scipy sparse output; the matrix is written directly from .X
        without materialising a dense copy.
        """
        import scipy.io as sio
        import scipy.sparse as sp
        import gzip

        out_dir.mkdir(parents=True, exist_ok=True)

        # Ensure .X is sparse
        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        # matrix.mtx.gz — genes x cells (10X convention: rows = genes)
        # scipy.io.mmwrite cannot write directly into a gzip stream,
        # so write plain .mtx first, then compress in place.
        mtx_path = out_dir / "matrix.mtx"
        sio.mmwrite(str(mtx_path), X.T.tocsc())
        with open(mtx_path, 'rb') as f_in, gzip.open(str(mtx_path) + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        mtx_path.unlink()  # remove the uncompressed file

        # barcodes.tsv.gz
        with gzip.open(out_dir / "barcodes.tsv.gz", 'wt') as fh:
            fh.write('\n'.join(adata.obs_names.tolist()) + '\n')

        # features.tsv.gz  (gene_id \t gene_name \t Expression)
        with gzip.open(out_dir / "features.tsv.gz", 'wt') as fh:
            for g in adata.var_names:
                fh.write(f'{g}\t{g}\tExpression\n')

        # optional: annotation CSV (cell_id, cell_type)
        if annotation_field and annotation_field in adata.obs.columns:
            pd.DataFrame({
                'cell_id':  adata.obs_names,
                'cell_type': adata.obs[annotation_field].astype(str)
            }).to_csv(out_dir / "annotations.csv", index=False)

    @staticmethod
    def _generate_seurat_script(annotation_field: str, nfeatures: int = 2000, dims: int = 30) -> str:
        """Generate R script for Seurat label transfer.

        Reads 10X MTX directories from /data/ref_mtx and /data/query_mtx.
        Attaches reference labels from annotations.csv.
        Writes predictions.csv to /scripts/ (the writable mount).
        """
        return f'''#!/usr/bin/env Rscript

# Seurat Label Transfer Script — generated by cell_annotation_framework.py
# Reads 10X-style MTX directories; does NOT require SeuratDisk or hdf5r.

library(Seurat)
library(Matrix)

cat("Loading 10X MTX datasets...\\n")
ref_mat   <- Read10X("/data/ref_mtx")
query_mat <- Read10X("/data/query_mtx")

ref   <- CreateSeuratObject(counts = ref_mat)
query <- CreateSeuratObject(counts = query_mat)
cat(sprintf("Reference: %d cells, %d genes\\n", ncol(ref), nrow(ref)))
cat(sprintf("Query:     %d cells, %d genes\\n", ncol(query), nrow(query)))

# Attach reference labels from the annotations CSV
annot <- read.csv("/data/ref_mtx/annotations.csv", stringsAsFactors=FALSE)
rownames(annot) <- annot$cell_id
ref$cell_type <- annot[colnames(ref), "cell_type"]
cat(sprintf("Cell types in reference: %s\\n", paste(unique(ref$cell_type), collapse=", ")))

# Preprocessing
cat("Normalizing and scaling...\\n")
ref   <- ScaleData(FindVariableFeatures(NormalizeData(ref,   verbose=FALSE), nfeatures={nfeatures}, verbose=FALSE), verbose=FALSE)
query <- ScaleData(FindVariableFeatures(NormalizeData(query, verbose=FALSE), nfeatures={nfeatures}, verbose=FALSE), verbose=FALSE)

# Find anchors and transfer labels
cat("Finding transfer anchors...\\n")
anchors <- FindTransferAnchors(reference=ref, query=query, dims=1:{dims}, verbose=TRUE)
cat(sprintf("Found %d anchors\\n", nrow(anchors@anchors)))

cat("Transferring labels...\\n")
predictions <- TransferData(
    anchorset = anchors,
    refdata   = ref$cell_type,
    dims      = 1:{dims},
    verbose   = TRUE
)

# Save results to /scripts/ (writable mount)
results <- data.frame(
    cell_id              = rownames(predictions),
    cell_type_annotation = predictions$predicted.id,
    prediction_score     = apply(predictions[, grepl("prediction.score", colnames(predictions))], 1, max, na.rm=TRUE),
    stringsAsFactors     = FALSE
)
write.csv(results, "/scripts/predictions.csv", row.names=FALSE)
cat(sprintf("Saved %d predictions to /scripts/predictions.csv\\n", nrow(results)))
'''

# =============================================================================
# CODI ANNOTATOR
# =============================================================================

class CodiAnnotator(AnnotationTool):
    """
    CoDi (Contrastive Distance) cell type annotation via Docker.

    Runs the vladimirkovacevic/codi container with the reference (--sc_path)
    and query (--st_path) h5ad files and the annotation field (-a).
    """

    def run(self, adata_ref: ad.AnnData, adata_query: ad.AnnData,
            annotation_field: str, common_genes: List[str],
            ref_path: str, query_path: str,
            marker_genes: Optional[List[str]] = None,
            args: Optional[argparse.Namespace] = None) -> pd.DataFrame:
        """Run CoDi annotation via Docker.

        Inputs are hard-linked (zero-copy) into the temp dir that is mounted
        as /data.  CoDi reads from and writes output into the same /data.
        """
        self.logger.info("Starting CoDi annotation via Docker")

        # User-configurable parameter
        codi_epochs = getattr(args, 'codi_epochs', None) if args else None
        self.logger.info(f"Parameters — epochs: {codi_epochs or 'container default'}")

        # Temp dir: hard-linked inputs + CoDi output CSV
        tmpdir = Path(tempfile.mkdtemp(prefix='codi_'))
        self.logger.debug(f"Temporary directory: {tmpdir}")

        try:
            # Hard-link originals — no Python-side matrix load or serialisation
            link_or_copy(ref_path,   tmpdir / "reference.h5ad",  self.logger)
            link_or_copy(query_path, tmpdir / "query.h5ad",      self.logger)

            # The vladimirkovacevic/codi image has ENTRYPOINT set to
            # ["python3","/opt/CoDi/core/CoDi.py"].  Docker appends CMD args
            # after the entrypoint, so we pass only the CoDi arguments here.
            # Do NOT repeat "python /opt/codi/CoDi.py" — that would be
            # interpreted as positional arguments by argparse.
            cmd = [
                'docker', 'run', '--rm',
                '-v', f'{tmpdir.absolute()}:/data',
                CODI_DOCKER_IMAGE,
                '--sc_path', '/data/reference.h5ad',
                '--st_path', '/data/query.h5ad',
                '-a', annotation_field,
                '--out_path', '/data',
                '--verbose'
            ]

            # Append optional epochs override (only if user specified)
            if codi_epochs is not None:
                cmd.extend(['--epochs', str(codi_epochs)])

            self.logger.info(f"Docker command: {' '.join(cmd)}")
            self.logger.info("Running CoDi in Docker container")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.logger.error(f"CoDi Docker failed with exit code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"CoDi execution failed: {result.stderr}")

            # Log both streams — CoDi's --verbose output goes to stderr
            if result.stdout:
                self.logger.info(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                self.logger.info(f"STDERR:\n{result.stderr}")

            # CoDi writes its output CSV into the mounted volume (/data).
            # Find the new CSV that was not one of our inputs.
            input_names = {'reference.h5ad', 'query.h5ad'}
            output_csvs = [
                f for f in tmpdir.iterdir()
                if f.suffix == '.csv' and f.name not in input_names
            ]

            if len(output_csvs) == 0:
                raise RuntimeError(
                    "CoDi did not produce an output CSV in the mounted volume"
                )

            # If multiple CSVs appeared, use the most recently modified one
            output_csv = sorted(output_csvs, key=lambda f: f.stat().st_mtime)[-1]
            self.logger.info(f"CoDi output file: {output_csv.name}")

            # Read raw output and normalise to the standard schema
            raw = pd.read_csv(output_csv)
            predictions = self._normalise_output(raw)

            self.logger.info(f"CoDi completed: {len(predictions)} cells annotated")
            return predictions

        finally:
            # Cleanup temporary directory
            self.logger.debug("Cleaning up temporary files")
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _normalise_output(raw: pd.DataFrame) -> pd.DataFrame:
        """
        Map whatever column names CoDi uses into the standard three-column schema.

        Tries common CoDi output column names; falls back to positional columns
        if none match.
        """
        # Candidate column names CoDi may use (case-insensitive search)
        lower_map = {c.lower(): c for c in raw.columns}

        # --- cell_id ---
        cell_id_keys = ['cell_id', 'barcode', 'cell_barcode', 'spot_id', 'index']
        cell_id_col = next((lower_map[k] for k in cell_id_keys if k in lower_map), None)
        if cell_id_col is None:
            # Use the index or first column as cell IDs
            cell_ids = raw.index.astype(str) if raw.index.dtype != object else raw.index
            cell_ids = cell_ids.tolist()
        else:
            cell_ids = raw[cell_id_col].astype(str).tolist()

        # --- cell_type_annotation ---
        # 'codi' is the primary prediction column in CoDi's output CSV
        type_keys = ['codi', 'cell_type', 'cell_type_annotation', 'predicted_type',
                     'annotation', 'predicted', 'label', 'class']
        type_col = next((lower_map[k] for k in type_keys if k in lower_map), None)
        if type_col is None:
            type_col = raw.columns[-1]

        # --- prediction_score ---
        # 'codi_confidence' is the confidence column in CoDi's output CSV
        score_keys = ['codi_confidence', 'score', 'prediction_score', 'confidence', 'probability']
        score_col = next((lower_map[k] for k in score_keys if k in lower_map), None)

        return pd.DataFrame({
            'cell_id':               cell_ids,
            'cell_type_annotation':  raw[type_col].astype(str),
            'prediction_score':      raw[score_col].astype(float) if score_col else 1.0
        })

# =============================================================================
# RESULT MERGING AND ENSEMBLE
# =============================================================================

def merge_predictions(results: Dict[str, pd.DataFrame], logger: logging.Logger) -> pd.DataFrame:
    """
    Merge prediction DataFrames from multiple tools into a single wide DataFrame.

    Args:
        results: Dict mapping tool names to prediction DataFrames.
        logger: Logger instance.

    Returns:
        Merged DataFrame with columns: cell_id, tool1, tool2, ...
    """
    logger.info(f"Merging predictions from {len(results)} tools")

    if len(results) == 0:
        raise ValueError("No results to merge")

    # Start with cell IDs from first tool
    first_tool = list(results.keys())[0]
    merged = results[first_tool][['cell_id', 'cell_type_annotation']].copy()
    merged = merged.rename(columns={'cell_type_annotation': first_tool})

    # Add predictions from other tools
    for tool_name, predictions in list(results.items())[1:]:
        tool_preds = predictions[['cell_id', 'cell_type_annotation']].copy()
        tool_preds = tool_preds.rename(columns={'cell_type_annotation': tool_name})

        merged = merged.merge(tool_preds, on='cell_id', how='outer')

    logger.info(f"Merged DataFrame: {len(merged)} cells, {len(results)} tools")

    return merged

def majority_voting(merged_df: pd.DataFrame, tool_columns: List[str],
                   logger: logging.Logger) -> pd.DataFrame:
    """
    Generate ensemble predictions via majority voting.

    Args:
        merged_df: DataFrame with cell_id and tool prediction columns.
        tool_columns: List of column names containing predictions.
        logger: Logger instance.

    Returns:
        DataFrame with ensemble predictions:
        [cell_id, ensemble_prediction, confidence, agreement_count, n_methods]
    """
    logger.info(f"Generating ensemble predictions via majority voting")

    results = []

    for _, row in merged_df.iterrows():
        cell_id = row['cell_id']

        # Collect non-null predictions
        votes = []
        for tool in tool_columns:
            if tool in row and pd.notna(row[tool]):
                votes.append(str(row[tool]))

        if len(votes) == 0:
            # No predictions available
            results.append({
                'cell_id': cell_id,
                'ensemble_prediction': 'Unknown',
                'confidence': 0.0,
                'agreement_count': 0,
                'n_methods': 0
            })
            continue

        # Count votes
        vote_counts = pd.Series(votes).value_counts()
        winner = vote_counts.index[0]
        winner_count = vote_counts.iloc[0]

        # Handle ties: alphabetical order (deterministic)
        if len(vote_counts) > 1 and vote_counts.iloc[1] == winner_count:
            tied_types = vote_counts[vote_counts == winner_count].index.tolist()
            winner = sorted(tied_types)[0]
            logger.debug(f"Tie at {cell_id}: {tied_types} -> choosing {winner}")

        # Calculate confidence
        confidence = winner_count / len(votes)

        results.append({
            'cell_id': cell_id,
            'ensemble_prediction': winner,
            'confidence': confidence,
            'agreement_count': winner_count,
            'n_methods': len(votes)
        })

    ensemble_df = pd.DataFrame(results)

    # Log statistics
    mean_confidence = ensemble_df['confidence'].mean()
    unanimous = (ensemble_df['confidence'] == 1.0).sum()

    logger.info(f"Ensemble statistics:")
    logger.info(f"  Mean confidence: {mean_confidence:.3f}")
    logger.info(f"  Unanimous predictions: {unanimous}/{len(ensemble_df)} ({100*unanimous/len(ensemble_df):.1f}%)")
    logger.info(f"  Unique cell types: {ensemble_df['ensemble_prediction'].nunique()}")

    return ensemble_df

# =============================================================================
# SPATIAL ANNOTATION FIGURE
# =============================================================================

def generate_annotation_figure(
        adata_query: ad.AnnData,
        merged_df: pd.DataFrame,
        tool_names: List[str],
        output_dir: Path,
        sample_name: str,
        logger: logging.Logger) -> Optional[Path]:
    """Generate a Nature-style multi-panel spatial annotation figure.

    One panel per annotation tool plus a final panel for the ensemble
    prediction.  All panels share a consistent colour palette so that the
    same cell type always maps to the same colour.  The figure is saved as
    a PNG at 200 dpi with no axis boxes — ready for journal submission.

    Args:
        adata_query: Query AnnData (must contain obsm['spatial']).
        merged_df:   Merged predictions DataFrame (tool columns + ensemble_prediction).
        tool_names:  Ordered list of successful tool names (column keys in merged_df).
        output_dir:  Directory to save the figure.
        sample_name: Prefix for the output filename.
        logger:      Logger instance.

    Returns:
        Path to the saved PNG, or None if spatial coordinates are unavailable.
    """
    import matplotlib
    matplotlib.use('Agg')                          # non-interactive; safe even if already set
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.font_manager import fontManager

    # --- font: Arial is the Nature standard; fall back to sans-serif ---
    _font_family = 'Arial' if 'Arial' in {f.name for f in fontManager.ttflist} else 'sans-serif'
    matplotlib.rcParams['font.family'] = _font_family

    # --- spatial coordinates (standard scanpy key) ---
    if 'spatial' not in adata_query.obsm:
        logger.warning("obsm['spatial'] not found in query — skipping annotation figure")
        return None

    coords = np.array(adata_query.obsm['spatial'])   # (n_spots, 2)
    spot_ids = adata_query.obs_names.tolist()

    # --- panels: one per tool, then ensemble (last) ---
    panel_order = list(tool_names) + ['Ensemble']
    # column name in merged_df for each panel
    col_map = {t: t for t in tool_names}
    col_map['Ensemble'] = 'ensemble_prediction'
    n_panels = len(panel_order)

    # --- unified colour palette across all panels ---
    all_types: set = set()
    for col in list(tool_names) + ['ensemble_prediction']:
        if col in merged_df.columns:
            all_types.update(merged_df[col].dropna().astype(str).unique())
    all_types_sorted = sorted(all_types)
    n_types = len(all_types_sorted)

    # Nature-compatible categorical palette: tab10 for ≤10, tab20 for >10
    cmap_name = 'tab10' if n_types <= 10 else 'tab20'
    cmap = plt.cm.get_cmap(cmap_name, max(n_types, 1))
    type_to_color = {t: cmap(i / max(n_types - 1, 1)) for i, t in enumerate(all_types_sorted)}

    # --- layout: 2 columns for ≤4 panels, 3 columns otherwise ---
    n_cols = 2 if n_panels <= 4 else 3
    n_rows = int(np.ceil(n_panels / n_cols))

    # Nature double-column width ≈ 170 mm = 6.7 in; row height 2.4 in
    fig_w = 6.7
    fig_h = n_rows * 2.4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)
    axes_flat = axes.flatten()

    # Build per-panel annotation arrays aligned to spatial coordinate order
    for panel_idx, label in enumerate(panel_order):
        ax = axes_flat[panel_idx]
        col = col_map[label]

        # Map merged_df predictions to the coordinate array order
        pred_map = dict(zip(merged_df['cell_id'].astype(str), merged_df[col].astype(str)))
        annotations = np.array([pred_map.get(sid, 'Unknown') for sid in spot_ids])
        colors = np.array([type_to_color.get(a, (0.65, 0.65, 0.65, 1.0)) for a in annotations])

        # Scatter — small markers, slight transparency for density
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=colors, s=5, alpha=0.85, linewidths=0, edgecolors='none')

        # Panel title — bold for the ensemble panel
        ax.set_title(label.replace('_', ' ').title(),
                     fontsize=9,
                     fontweight='bold' if label == 'Ensemble' else 'normal',
                     pad=5, fontfamily=_font_family)

        # Nature style: strip everything except the scatter
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide any unused subplot cells
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # --- shared legend centred below the panels ---
    legend_handles = [Patch(facecolor=type_to_color[t], edgecolor='none', label=t)
                      for t in all_types_sorted]
    legend_ncol = min(n_types, 5)

    # Reserve vertical space at the bottom for the legend
    legend_height = 0.04 * int(np.ceil(n_types / legend_ncol)) + 0.03   # adaptive

    fig.tight_layout(rect=[0, legend_height, 1, 1])

    fig.legend(handles=legend_handles,
               loc='lower center',
               ncol=legend_ncol,
               fontsize=7,
               frameon=False,
               columnspacing=1.2,
               handlelength=1.0,
               bbox_to_anchor=(0.5, 0.005))

    # --- save at 200 dpi, white background ---
    fig_path = output_dir / FIGURE_PNG.format(sample=sample_name)
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"Saved annotation figure: {fig_path.name}")
    return fig_path

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_annotation_pipeline(args: argparse.Namespace, logger: logging.Logger) -> int:
    """
    Main annotation pipeline orchestration.

    Args:
        args: Parsed command-line arguments.
        logger: Logger instance.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    start_time = time.time()

    # =========================================================================
    # STEP 1: Setup and validation
    # =========================================================================

    logger.info("=" * 80)
    logger.info("CELL TYPE ANNOTATION FRAMEWORK")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")

    # Determine sample name
    sample_name = args.sample_name
    if sample_name is None:
        sample_name = Path(args.query).stem
    logger.info(f"Sample name: {sample_name}")

    # =========================================================================
    # STEP 2: Check tool availability
    # =========================================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("TOOL AVAILABILITY CHECK")
    logger.info("-" * 40)

    # Check Docker-based tools (Seurat, CoDi)
    docker_ok = check_docker_available(logger)
    if docker_ok:
        TOOLS_AVAILABLE['seurat'] = check_docker_image(SEURAT_DOCKER_IMAGE, logger)
        TOOLS_AVAILABLE['codi']   = check_docker_image(CODI_DOCKER_IMAGE,   logger)

    # Display availability
    for tool, available in TOOLS_AVAILABLE.items():
        status = "✓ Available" if available else "✗ Not available"
        logger.info(f"  {tool:15s}: {status}")

    # =========================================================================
    # STEP 3: Determine which tools to run
    # =========================================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("TOOL SELECTION")
    logger.info("-" * 40)

    # Parse requested tools — flatten comma-separated entries so that both
    # "--tools tangram codi" and "--tools tangram,codi" (and mixed) work.
    VALID_TOOLS = {'tangram', 'seurat', 'codi', 'all'}
    requested_tools = []
    for entry in args.tools:
        requested_tools.extend(t.strip().lower() for t in entry.split(',') if t.strip())

    invalid = set(requested_tools) - VALID_TOOLS
    if invalid:
        logger.error(f"Unknown tool(s): {', '.join(sorted(invalid))}. "
                     f"Valid options: {', '.join(sorted(VALID_TOOLS))}")
        return 1

    if 'all' in requested_tools:
        requested_tools = list(TOOLS_AVAILABLE.keys())

    logger.info(f"Requested tools: {', '.join(requested_tools)}")

    # Filter by availability
    tools_to_run = [t for t in requested_tools if TOOLS_AVAILABLE.get(t, False)]

    if len(tools_to_run) == 0:
        logger.error("No tools available to run!")
        logger.error("Please install at least one annotation tool.")
        logger.error("See requirements.txt for installation instructions.")
        return 1

    logger.info(f"Tools to run: {', '.join(tools_to_run)}")

    unavailable = set(requested_tools) - set(tools_to_run)
    if unavailable:
        logger.warning(f"Skipping unavailable tools: {', '.join(unavailable)}")

    # =========================================================================
    # STEP 4: Load and validate input data
    # =========================================================================

    # Only Tangram accesses the expression matrix in-process.
    # When every selected tool is Docker-based we can open in backed mode
    # so the (potentially huge) .X matrix stays on disk.
    # Tools that need the expression matrix (.X) loaded into host memory:
    # - tangram: runs entirely in-process via the Python API.
    # - seurat:  host converts h5ad → 10X MTX before mounting into Docker.
    # Only CoDi hard-links the original h5ad and needs no .X on the host.
    PYTHON_API_TOOLS = {'tangram', 'seurat'}
    needs_full_load  = bool(set(tools_to_run) & PYTHON_API_TOOLS)
    backed_mode      = None if needs_full_load else 'r'

    if backed_mode:
        logger.info("All selected tools are Docker-based — opening files in backed mode (metadata only)")
    else:
        logger.info("At least one Python-API tool selected — full matrix load required")

    logger.info("")
    logger.info("=" * 80)
    logger.info("LOADING INPUT DATA")
    logger.info("-" * 40)

    # Validate reference file
    logger.info(f"Reference: {args.reference}")
    if not validate_file_exists(args.reference, logger):
        return 1

    adata_ref = validate_h5ad_file(args.reference, logger, backed=backed_mode)
    if adata_ref is None:
        return 1

    # Validate annotation field
    if not validate_annotation_field(adata_ref, args.annotation_field, logger):
        return 1

    # Validate query file
    logger.info("")
    logger.info(f"Query: {args.query}")
    if not validate_file_exists(args.query, logger):
        return 1

    adata_query = validate_h5ad_file(args.query, logger, backed=backed_mode)
    if adata_query is None:
        return 1

    # h5ad files often contain duplicate gene names — make them unique so that
    # gene-name-based slicing works reliably.  Only needed (and possible) when
    # the full matrix is loaded; backed mode keeps names as-is on disk.
    if not backed_mode:
        adata_ref.var_names_make_unique()
        adata_query.var_names_make_unique()
        logger.debug("var_names made unique for both datasets")

    # =========================================================================
    # STEP 4a: Detect & harmonize expression-matrix states
    # =========================================================================
    # All annotation tools expect raw counts.  When .X has been normalised
    # or log-transformed but .raw is present we swap .X ← .raw.X so that
    # downstream code sees raw counts.  CoDi hard-links the *original*
    # on-disk file, so it cannot benefit from the in-memory swap — we warn
    # about that separately.
    ref_state   = 'unknown'
    query_state = 'unknown'

    if not backed_mode:
        logger.info("")
        logger.info("Checking expression matrix states...")
        ref_state   = detect_matrix_state(adata_ref,   'Reference', logger)
        query_state = detect_matrix_state(adata_query, 'Query',     logger)

        # Save original on-disk states before any swap (used for CoDi warning)
        ref_state_orig   = ref_state
        query_state_orig = query_state

        # Restore raw counts from .raw where available
        for label, adata, state in [('Reference', adata_ref,   ref_state),
                                    ('Query',     adata_query, query_state)]:
            if state != 'raw' and adata.raw is not None:
                logger.warning(
                    f"[{label}] .X is {state}-transformed but .raw is available — "
                    f"replacing .X with raw counts from .raw for annotation tools"
                )
                adata.X = adata.raw[:, adata.var_names].X
                # Update local state so marker-gene block uses the correct value
                if label == 'Reference':
                    ref_state = 'raw'
                else:
                    query_state = 'raw'
            elif state != 'raw' and adata.raw is None:
                logger.warning(
                    f"[{label}] .X is {state}-transformed and .raw is NOT available. "
                    f"Annotation tools expect raw counts — results may be unreliable"
                )

        # CoDi reads the original on-disk file via hard-link; our in-memory
        # swap cannot help it.  Warn when on-disk data is not raw counts.
        if 'codi' in tools_to_run:
            if ref_state_orig != 'raw' or query_state_orig != 'raw':
                logger.warning(
                    f"[CoDi] reads original h5ad files via hard-link (on-disk data "
                    f"is unchanged).  Reference: {ref_state_orig}, Query: "
                    f"{query_state_orig} on disk — in-memory .raw swap does NOT apply"
                )
    else:
        logger.info("Backed mode — skipping matrix-state detection (no .X in memory)")

    # Check gene overlap
    logger.info("")
    common_genes, overlap_pct = calculate_gene_overlap(adata_ref, adata_query, logger)

    if len(common_genes) == 0:
        logger.error("No common genes between reference and query!")
        return 1

    # =========================================================================
    # STEP 4b: Pre-compute marker genes once (used by Tangram)
    # =========================================================================

    # Only run rank_genes_groups if Tangram will actually execute — and only if
    # the result does not already exist (avoids redundant computation).
    marker_genes = None  # type: Optional[List[str]]
    if 'tangram' in tools_to_run:
        logger.info("")
        logger.info("Pre-computing marker genes (run once, reused by Tangram)")

        # rank_genes_groups requires log-normalised data.  If .X is raw
        # (the common case after the .raw swap above) we normalise in place
        # but first save .X so we can restore raw counts for Tangram / Seurat.
        _raw_X = None
        if 'rank_genes_groups' not in adata_ref.uns:
            if ref_state == 'raw':
                logger.debug("Saving raw .X; normalising for marker-gene detection")
                _raw_X = adata_ref.X.copy()          # sparse copy is cheap
                sc.pp.normalize_total(adata_ref, target_sum=1e4)
                sc.pp.log1p(adata_ref)
            elif ref_state == 'normalized':
                logger.debug("Saving normalized .X; applying log1p for marker genes")
                _raw_X = adata_ref.X.copy()
                sc.pp.log1p(adata_ref)
            else:
                logger.debug(f"Reference .X is {ref_state} — using as-is for marker genes")

            sc.tl.rank_genes_groups(adata_ref, groupby=args.annotation_field, method='wilcoxon')
            logger.debug("Computed rank_genes_groups on reference")
        else:
            logger.debug("rank_genes_groups already present in reference — skipping")

        # Restore the original .X so annotation tools see the correct matrix
        if _raw_X is not None:
            adata_ref.X = _raw_X
            del _raw_X
            logger.debug("Restored original .X after marker-gene detection")

        # Extract top N markers per cell type, then intersect with common_genes once
        n_markers_per_type = 100
        raw_markers: List[str] = []
        for ct in adata_ref.obs[args.annotation_field].unique():
            ct_df = sc.get.rank_genes_groups_df(
                adata_ref, group=ct, key='rank_genes_groups'
            )
            raw_markers.extend(ct_df.head(n_markers_per_type)['names'].tolist())

        marker_genes = sorted(set(raw_markers) & set(common_genes))
        logger.info(f"Marker genes (intersected with common genes): {len(marker_genes)}")

    # =========================================================================
    # STEP 5: Run annotation tools
    # =========================================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("RUNNING ANNOTATION TOOLS")
    logger.info("=" * 80)

    # Tool class mapping
    TOOL_CLASSES = {
        'tangram': TangramAnnotator,
        'seurat': SeuratAnnotator,
        'codi': CodiAnnotator
    }

    results = {}

    for tool_name in tools_to_run:
        logger.info("")
        logger.info(f"{'=' * 80}")
        logger.info(f"RUNNING {tool_name.upper()}")
        logger.info(f"{'-' * 40}")

        tool_start = time.time()

        try:
            # Initialize tool
            tool_class = TOOL_CLASSES[tool_name]
            tool = tool_class(logger)

            # Run annotation
            # common_genes / marker_genes  — pre-computed once by pipeline
            # ref_path / query_path        — original paths for Docker volume mounts
            predictions = tool.run(
                adata_ref, adata_query, args.annotation_field,
                common_genes, args.reference, args.query, marker_genes, args
            )

            # Validate output
            if len(predictions) == 0:
                logger.warning(f"{tool_name} produced no predictions")
                continue

            # Save results
            results[tool_name] = predictions

            # Save CSV
            csv_filename = CSV_TEMPLATE.format(sample=sample_name, tool=tool_name)
            csv_path = output_dir / csv_filename
            predictions.to_csv(csv_path, index=False)

            tool_elapsed = time.time() - tool_start
            logger.info(f"{tool_name} completed in {tool_elapsed:.1f} seconds")
            logger.info(f"Saved: {csv_path.name}")

        except Exception as e:
            tool_elapsed = time.time() - tool_start
            logger.error(f"{tool_name} failed after {tool_elapsed:.1f} seconds: {e}", exc_info=True)
            logger.warning(f"Continuing with remaining tools...")

    # =========================================================================
    # STEP 6: Check if any tools succeeded
    # =========================================================================

    if len(results) == 0:
        logger.error("")
        logger.error("=" * 80)
        logger.error("ALL TOOLS FAILED")
        logger.error("=" * 80)
        logger.error("No annotations were generated. Please check the logs above.")
        return 1

    logger.info("")
    logger.info(f"Successfully completed {len(results)}/{len(tools_to_run)} tools")

    # =========================================================================
    # STEP 7: Merge results from all tools
    # =========================================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("MERGING RESULTS")
    logger.info("-" * 40)

    merged_df = merge_predictions(results, logger)

    # =========================================================================
    # STEP 8: Generate ensemble and append as last column of merged CSV
    # =========================================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("ENSEMBLE PREDICTION")
    logger.info("-" * 40)

    if len(results) == 1:
        # Only one tool - ensemble is identical to that tool's predictions
        tool_name = list(results.keys())[0]
        logger.info(f"Only one tool succeeded ({tool_name}) - using its predictions directly")

        merged_df['ensemble_prediction'] = merged_df[tool_name]

        # Build ensemble_df for summary stats (confidence is 1.0 with a single tool)
        ensemble_df = pd.DataFrame({
            'ensemble_prediction': merged_df['ensemble_prediction'],
            'confidence': 1.0
        })
    else:
        # Multiple tools - run majority voting
        tool_columns = list(results.keys())
        ensemble_df = majority_voting(merged_df, tool_columns, logger)

        # Append ensemble prediction as the last column of the merged DataFrame
        merged_df['ensemble_prediction'] = ensemble_df['ensemble_prediction'].values

    # Save merged CSV (ensemble prediction is the last column)
    merged_filename = MERGED_CSV.format(sample=sample_name)
    merged_path = output_dir / merged_filename
    merged_df.to_csv(merged_path, index=False)
    logger.info(f"Saved merged predictions (with ensemble): {merged_path.name}")

    # =========================================================================
    # STEP 9: Generate publication-quality spatial annotation figure
    # =========================================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATING ANNOTATION FIGURE")
    logger.info("-" * 40)

    generate_annotation_figure(
        adata_query=adata_query,
        merged_df=merged_df,
        tool_names=list(results.keys()),
        output_dir=output_dir,
        sample_name=sample_name,
        logger=logger
    )

    # =========================================================================
    # STEP 10: Generate summary JSON
    # =========================================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATING SUMMARY")
    logger.info("-" * 40)

    elapsed = time.time() - start_time

    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 2),
        'reference_file': str(Path(args.reference).absolute()),
        'query_file': str(Path(args.query).absolute()),
        'annotation_field': args.annotation_field,
        'sample_name': sample_name,
        'output_directory': str(output_dir.absolute()),
        'tools_requested': requested_tools,
        'tools_succeeded': list(results.keys()),
        'n_cells_annotated': len(ensemble_df),
        'n_cell_types_reference': int(adata_ref.obs[args.annotation_field].nunique()),
        'n_cell_types_ensemble': int(ensemble_df['ensemble_prediction'].nunique()),
        'mean_confidence': float(ensemble_df['confidence'].mean()),
        'gene_overlap_percent': round(overlap_pct * 100, 2)
    }

    summary_filename = SUMMARY_JSON.format(sample=sample_name)
    summary_path = output_dir / summary_filename

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary: {summary_path.name}")

    # =========================================================================
    # STEP 10: Final summary
    # =========================================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("ANNOTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Cells annotated: {len(ensemble_df)}")
    logger.info(f"Cell types identified: {ensemble_df['ensemble_prediction'].nunique()}")
    logger.info(f"Mean confidence: {ensemble_df['confidence'].mean():.3f}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  - Merged predictions (ensemble = last column): {merged_path.name}")
    for tool_name in results.keys():
        csv_name = CSV_TEMPLATE.format(sample=sample_name, tool=tool_name)
        logger.info(f"  - {tool_name}: {csv_name}")
    logger.info(f"  - Summary: {summary_path.name}")
    logger.info("")
    logger.info("=" * 80)

    return 0

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Cell Type Annotation Framework for Spatial Transcriptomics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run all available tools
  python cell_annotation_framework.py \\
      --reference data/reference.h5ad \\
      --query data/query.h5ad \\
      --annotation-field cell_type \\
      --tools all

  # Run specific tools (space-separated or comma-separated)
  python cell_annotation_framework.py \\
      --reference data/reference.h5ad \\
      --query data/query.h5ad \\
      --annotation-field cell_type \\
      --tools tangram,codi

  # Custom output directory
  python cell_annotation_framework.py \\
      --reference data/reference.h5ad \\
      --query data/query.h5ad \\
      --annotation-field cell_type \\
      --output results/sample1 \\
      --sample-name sample1 \\
      --verbose
        '''
    )

    # Required arguments
    parser.add_argument(
        '--reference',
        required=True,
        type=str,
        help='Path to reference scRNA-seq dataset (.h5ad file with cell type annotations)'
    )

    parser.add_argument(
        '--query',
        required=True,
        type=str,
        help='Path to query/test dataset to be annotated (.h5ad file)'
    )

    parser.add_argument(
        '--annotation-field',
        required=True,
        type=str,
        help='Name of the cell type annotation column in reference.obs'
    )

    # Optional arguments
    parser.add_argument(
        '--tools',
        nargs='+',
        default=['all'],
        help='Tools to run: space-separated, comma-separated, or "all". '
             'Examples: --tools all | --tools tangram codi | --tools tangram,codi'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./annotation_results',
        help='Output directory for results (default: ./annotation_results)'
    )

    parser.add_argument(
        '--sample-name',
        type=str,
        default=None,
        help='Sample name for output files (default: query filename)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )

    # -------------------------------------------------------------------------
    # Tool-specific parameters (prefixed with tool name)
    # -------------------------------------------------------------------------
    tool_params = parser.add_argument_group(
        'Tool-specific parameters',
        'Tunable parameters for individual tools.  Each is prefixed with the '
        'tool name so the help text is self-documenting and there are no clashes.'
    )

    # --- Tangram ---
    tool_params.add_argument(
        '--tangram_epochs',
        type=int, default=1000,
        help='[Tangram] Training epochs for cell-to-space mapping (default: 1000)'
    )
    tool_params.add_argument(
        '--tangram_mode',
        type=str, choices=['cells', 'genes'], default='cells',
        help='[Tangram] Mapping mode: "cells" maps individual cells, '
             '"genes" maps gene expression patterns (default: cells)'
    )
    tool_params.add_argument(
        '--tangram_density_prior',
        type=str, choices=['rna_count_based', 'uniform'], default='rna_count_based',
        help='[Tangram] Spatial density prior.  "rna_count_based" weights spots '
             'by total RNA count (default: rna_count_based)'
    )

    # --- Seurat ---
    tool_params.add_argument(
        '--seurat_nfeatures',
        type=int, default=2000,
        help='[Seurat] Highly variable features selected during preprocessing (default: 2000)'
    )
    tool_params.add_argument(
        '--seurat_dims',
        type=int, default=30,
        help='[Seurat] PCA dimensions for anchor finding, i.e. 1:dims (default: 30)'
    )

    # --- CoDi ---
    tool_params.add_argument(
        '--codi_epochs',
        type=int, default=None,
        help='[CoDi] Training epochs.  Omit to use the container default'
    )

    args = parser.parse_args()

    # Setup logging
    output_dir = Path(args.output)
    log_file = output_dir / LOG_FILE
    logger = setup_logging(log_file=log_file, verbose=args.verbose)

    # Run pipeline
    try:
        exit_code = run_annotation_pipeline(args, logger)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
