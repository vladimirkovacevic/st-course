#!/usr/bin/env Rscript
# =============================================================================
# BANKSY Spatial Transcriptomics Analysis
# =============================================================================
# Automatically generated script for running BANKSY analysis
# Generated: 2026-01-28T12:43:06.303265
# Sample: adult_mouse_brain_ST4k
# =============================================================================

# -----------------------------------------------------------------------------
# 1. SETUP AND CONFIGURATION
# -----------------------------------------------------------------------------

cat("\n========================================\n")
cat("BANKSY SPATIAL CLUSTERING ANALYSIS\n")
cat("========================================\n\n")

start_time <- Sys.time()
cat("Analysis started:", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n\n")

# Suppress warnings for cleaner output
options(warn = -1)

# Load required libraries
cat("Loading required packages...\n")

suppressPackageStartupMessages({
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
})

cat("Packages loaded successfully\n\n")

# Configuration parameters
config <- list(
    input_file = "/data/input.h5ad",
    output_dir = "/output",
    sample_name = "adult_mouse_brain_ST4k",
    lambda = c(0.2, 0.8),
    k_geom = c(15, 30),
    resolution = 0.8,
    n_pcs = 30
)

cat("Configuration:\n")
cat("  Input file:", config$input_file, "\n")
cat("  Output directory:", config$output_dir, "\n")
cat("  Lambda values:", paste(config$lambda, collapse=", "), "\n")
cat("  k_geom values:", paste(config$k_geom, collapse=", "), "\n")
cat("  Clustering resolution:", config$resolution, "\n")
cat("  Number of PCs:", config$n_pcs, "\n\n")

# Create output directory
dir.create(config$output_dir, showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 2. DATA LOADING (using rhdf5 directly for speed)
# -----------------------------------------------------------------------------

cat("Loading h5ad data using rhdf5...\n")

tryCatch({
    # Read h5ad file structure
    h5_contents <- h5ls(config$input_file)
    cat("  H5AD file structure detected\n")

    # Read expression matrix (X)
    # Check if X is stored as sparse or dense
    x_group <- h5_contents[h5_contents$name == "X" | h5_contents$group == "/X", ]

    if (any(h5_contents$name == "data" & h5_contents$group == "/X")) {
        # Sparse matrix format (CSR/CSC)
        cat("  Reading sparse matrix...\n")
        data_vec <- h5read(config$input_file, "/X/data")
        indices <- h5read(config$input_file, "/X/indices")
        indptr <- h5read(config$input_file, "/X/indptr")

        # Get dimensions from obs and var
        obs_names <- h5read(config$input_file, "/obs/_index")
        # Try different possible gene name columns
        var_names <- tryCatch({
            h5read(config$input_file, "/var/_index")
        }, error = function(e) {
            tryCatch({
                h5read(config$input_file, "/var/Symbol")
            }, error = function(e2) {
                h5read(config$input_file, "/var/gene_ids")
            })
        })
        n_obs <- length(obs_names)
        n_var <- length(var_names)
        cat("  n_obs:", n_obs, "n_var:", n_var, "\n")

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
    } else {
        # Dense matrix
        cat("  Reading dense matrix...\n")
        expr_matrix <- t(h5read(config$input_file, "/X"))
    }

    cat("  Matrix dimensions:", nrow(expr_matrix), "genes x", ncol(expr_matrix), "cells\n")

    # Set row and column names (already read above)
    colnames(expr_matrix) <- obs_names
    rownames(expr_matrix) <- var_names

    cat("  Cells/spots loaded:", ncol(expr_matrix), "\n")
    cat("  Genes loaded:", nrow(expr_matrix), "\n")

    # Read spatial coordinates from obsm/spatial
    spatial_coords <- NULL

    if (any(h5_contents$group == "/obsm" & h5_contents$name == "spatial")) {
        spatial_coords <- h5read(config$input_file, "/obsm/spatial")
        cat("  Raw spatial coords dim:", paste(dim(spatial_coords), collapse=" x "), "\n")
        if (is.matrix(spatial_coords)) {
            # Data is stored as features x cells, transpose to cells x features
            if (nrow(spatial_coords) < ncol(spatial_coords)) {
                spatial_coords <- t(spatial_coords)
            }
            # Take only first 2 columns (x, y)
            if (ncol(spatial_coords) > 2) {
                spatial_coords <- spatial_coords[, 1:2]
            }
        }
        cat("  Spatial coordinates found in obsm/spatial\n")
    } else if (any(h5_contents$group == "/obsm" & h5_contents$name == "X_spatial")) {
        spatial_coords <- h5read(config$input_file, "/obsm/X_spatial")
        if (is.matrix(spatial_coords)) {
            if (nrow(spatial_coords) < ncol(spatial_coords)) {
                spatial_coords <- t(spatial_coords)
            }
            if (ncol(spatial_coords) > 2) {
                spatial_coords <- spatial_coords[, 1:2]
            }
        }
        cat("  Spatial coordinates found in obsm/X_spatial\n")
    } else {
        # Try to find coordinates in obs columns
        obs_cols <- h5_contents[h5_contents$group == "/obs", "name"]
        x_candidates <- c("x", "X", "x_coord", "spatial_x", "array_col", "imagecol", "x_centroid")
        y_candidates <- c("y", "Y", "y_coord", "spatial_y", "array_row", "imagerow", "y_centroid")

        x_col <- NULL
        y_col <- NULL

        for (xname in x_candidates) {
            if (xname %in% obs_cols) {
                x_col <- xname
                break
            }
        }

        for (yname in y_candidates) {
            if (yname %in% obs_cols) {
                y_col <- yname
                break
            }
        }

        if (!is.null(x_col) && !is.null(y_col)) {
            x_vals <- h5read(config$input_file, paste0("/obs/", x_col))
            y_vals <- h5read(config$input_file, paste0("/obs/", y_col))
            spatial_coords <- cbind(x_vals, y_vals)
            cat("  Spatial coordinates found in obs:", x_col, "/", y_col, "\n")
        }
    }

    if (is.null(spatial_coords)) {
        stop("No spatial coordinates found in h5ad file. Checked obsm/spatial, obsm/X_spatial, and obs columns.")
    }

    # Ensure spatial_coords has correct dimensions
    if (nrow(spatial_coords) != ncol(expr_matrix)) {
        if (ncol(spatial_coords) == ncol(expr_matrix)) {
            spatial_coords <- t(spatial_coords)
        }
    }

    cat("  Spatial coordinates shape:", nrow(spatial_coords), "x", ncol(spatial_coords), "\n")

    # Create SpatialExperiment object
    cat("  Creating SpatialExperiment object...\n")

    spe <- SpatialExperiment(
        assays = list(counts = expr_matrix),
        spatialCoords = spatial_coords
    )

    cat("  SpatialExperiment created successfully\n\n")

    # Close h5 file connections
    h5closeAll()

}, error = function(e) {
    h5closeAll()
    cat("ERROR loading data:", conditionMessage(e), "\n")
    cat("Stack trace:\n")
    traceback()
    quit(status = 1)
})

# -----------------------------------------------------------------------------
# 3. QUALITY CONTROL
# -----------------------------------------------------------------------------

cat("Performing quality control...\n")

# Calculate QC metrics
qc_metrics <- perCellQCMetrics(spe)
colData(spe) <- cbind(colData(spe), qc_metrics)

cat("  Total UMI range:", range(qc_metrics$sum), "\n")
cat("  Genes detected range:", range(qc_metrics$detected), "\n")

# Filter cells based on QC metrics
# Use adaptive thresholds
sum_threshold <- quantile(qc_metrics$sum, c(0.01, 0.99))
detected_threshold <- quantile(qc_metrics$detected, c(0.01, 0.99))

keep_cells <- (qc_metrics$sum >= sum_threshold[1]) &
              (qc_metrics$sum <= sum_threshold[2]) &
              (qc_metrics$detected >= detected_threshold[1]) &
              (qc_metrics$detected <= detected_threshold[2])

n_removed <- sum(!keep_cells)
cat("  Cells removed by QC:", n_removed, "(", round(100 * n_removed / ncol(spe), 1), "% )\n")

spe <- spe[, keep_cells]
cat("  Cells after QC:", ncol(spe), "\n\n")

# Filter genes - keep genes expressed in at least 1% of cells
min_cells <- max(10, floor(ncol(spe) * 0.01))
gene_counts <- rowSums(assay(spe, "counts") > 0)
keep_genes <- gene_counts >= min_cells

n_genes_removed <- sum(!keep_genes)
cat("  Genes removed (detected in <", min_cells, "cells):", n_genes_removed, "\n")

spe <- spe[keep_genes, ]
cat("  Genes after filtering:", nrow(spe), "\n\n")

# -----------------------------------------------------------------------------
# 4. NORMALIZATION
# -----------------------------------------------------------------------------

cat("Normalizing data...\n")

# Library size normalization
spe <- computeLibraryFactors(spe)
assay(spe, "normcounts") <- normalizeCounts(spe, log = FALSE)
assay(spe, "lognormcounts") <- log1p(assay(spe, "normcounts"))

cat("  Normalization complete\n\n")

# -----------------------------------------------------------------------------
# 5. BANKSY ANALYSIS
# -----------------------------------------------------------------------------

cat("Running BANKSY analysis...\n")
cat("  Computing neighborhood features...\n")

# Compute BANKSY neighborhood features
spe <- computeBanksy(
    spe,
    assay_name = "normcounts",
    compute_agf = TRUE,
    k_geom = config$k_geom
)

cat("  Running BANKSY PCA...\n")

# Run BANKSY PCA
set.seed(42)  # For reproducibility
spe <- runBanksyPCA(
    spe,
    use_agf = TRUE,
    lambda = config$lambda,
    npcs = config$n_pcs
)

cat("  Running BANKSY UMAP...\n")

# Run BANKSY UMAP
spe <- runBanksyUMAP(
    spe,
    use_agf = TRUE,
    lambda = config$lambda
)

cat("  Clustering with BANKSY...\n")

# Cluster using BANKSY
spe <- clusterBanksy(
    spe,
    use_agf = TRUE,
    lambda = config$lambda,
    resolution = config$resolution
)

# Connect clusters across lambda values (only if multiple lambdas)
if (length(config$lambda) > 1) {
    spe <- connectClusters(spe)
    cat("  Clusters connected across lambda values\n")
} else {
    cat("  Single lambda value - skipping connectClusters\n")
}

cat("  BANKSY analysis complete\n\n")

# -----------------------------------------------------------------------------
# 6. EXTRACT AND SUMMARIZE RESULTS
# -----------------------------------------------------------------------------

cat("Extracting results...\n")

# Get cluster assignments for each lambda value
cluster_cols <- grep("^clust_", colnames(colData(spe)), value = TRUE)
cat("  Cluster columns found:", paste(cluster_cols, collapse=", "), "\n")

# Summarize clusters
for (cc in cluster_cols) {
    n_clusters <- length(unique(colData(spe)[[cc]]))
    cat("  ", cc, ": ", n_clusters, " clusters\n", sep = "")
}

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
for (cc in cluster_cols) {
    results_summary[[paste0("n_", cc)]] <- length(unique(colData(spe)[[cc]]))
}

cat("\n")

# -----------------------------------------------------------------------------
# 7. SAVE OUTPUTS
# -----------------------------------------------------------------------------

cat("Saving outputs...\n")

# Save processed SpatialExperiment object as RDS
rds_file <- file.path(config$output_dir, paste0(config$sample_name, "_banksy_results.rds"))
saveRDS(spe, rds_file)
cat("  RDS object saved:", rds_file, "\n")

# Save cluster assignments as CSV
cluster_data <- as.data.frame(colData(spe)[, c(cluster_cols, "sum", "detected")])
cluster_data$cell_id <- colnames(spe)
cluster_data$x <- spatialCoords(spe)[, 1]
cluster_data$y <- spatialCoords(spe)[, 2]

csv_file <- file.path(config$output_dir, paste0(config$sample_name, "_clusters.csv"))
write.csv(cluster_data, csv_file, row.names = FALSE)
cat("  Cluster assignments saved:", csv_file, "\n")

# Save summary
summary_file <- file.path(config$output_dir, paste0(config$sample_name, "_summary.csv"))
write.csv(results_summary, summary_file, row.names = FALSE)
cat("  Summary saved:", summary_file, "\n")

# -----------------------------------------------------------------------------
# 8. GENERATE VISUALIZATIONS
# -----------------------------------------------------------------------------

cat("Generating visualizations...\n")

# Create spatial plots for each lambda value
for (cc in cluster_cols) {

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
    cat("  Spatial plot saved:", plot_file, "\n")
}

# Create UMAP plots
umap_names <- reducedDimNames(spe)[grep("^UMAP", reducedDimNames(spe))]

for (umap_name in umap_names) {

    umap_coords <- reducedDim(spe, umap_name)

    # Get corresponding cluster column
    lambda_str <- gsub("UMAP_", "", umap_name)
    cc <- paste0("clust_", lambda_str)

    if (cc %in% cluster_cols) {
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
        cat("  UMAP plot saved:", plot_file, "\n")
    }
}

# -----------------------------------------------------------------------------
# 9. COMPLETION
# -----------------------------------------------------------------------------

end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = "mins")

cat("\n========================================\n")
cat("ANALYSIS COMPLETE\n")
cat("========================================\n")
cat("Duration:", round(as.numeric(duration), 2), "minutes\n")
cat("Output directory:", config$output_dir, "\n")
cat("\n")

# Write completion marker
completion_file <- file.path(config$output_dir, ".banksy_complete")
writeLines(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), completion_file)

cat("SUCCESS: BANKSY analysis completed successfully\n")
