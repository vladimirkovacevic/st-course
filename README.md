# Spatial Transcriptomics Course

A hands-on course for analyzing spatial transcriptomics data (Stereo-seq), covering the full pipeline from preprocessing through comparative analysis. Uses an adult mouse brain dataset.

## Course Overview

### Lesson 1: Preprocessing & Clustering
- Spatial transcriptomics & Stereo-seq introduction
- Data loading and initial assessment with Scanpy
- Adjusted QC thresholds for sparse data
- Normalization and feature selection
- Spatial smoothing -- improve signal quality across neighborhood
- Dimensionality reduction (PCA, UMAP)
- Cell type & Spatial-aware clustering with BANKSY

### Lesson 2: Cell Type Identification
- Clustering & intersection of marker genes with signature genes
- Reference-based cell type identification (Tangram, CoDi, Seurat, cell2location)
- Ensemble of cell type annotation algorithms
- Verification of cell type annotation using marker genes retention

### Lesson 3: Spatial Analysis
- Moran's I spatial autocorrelation -- spatially variable genes
- Identifying spatially coherent gene modules with Hotspot
- Co-occurrence and neighborhood analysis (Squidpy)
- Spatial domains and cell communities

### Lesson 4: Comparative Analysis
- Sample integration with Harmony
- Differential abundance testing
- Comparing cell type proportions between conditions
- Treated vs. Untreated comparison

## Repository Structure

```
st-course/
├── notebooks/
│   ├── Load_GSE.ipynb                          # Download and load data from GEO
│   ├── Clustering_Scanpy.ipynb                 # QC, normalization, PCA, UMAP, clustering
│   └── Cell_type_annotation_Marker_genes.ipynb # CellTypist + manual marker-based annotation
├── banksy/
│   ├── banksy.py                               # BANKSY spatial-aware clustering (Python)
│   ├── banksy.R                                # BANKSY spatial-aware clustering (R)
│   ├── banksy_inside_docker.R                  # BANKSY R script for Docker execution
│   └── docker_run.sh                           # Docker wrapper for BANKSY
└── scripts/
    └── cell_annotation_framework.py            # Multi-tool annotation ensemble (Tangram, Seurat, CoDi)
```

## Getting Started

### Prerequisites
- **Python 3.10+** with scanpy, anndata, celltypist, tangram-sc
- **R 4.x** with Banksy (for spatial-aware clustering)
- **Docker** (optional, for running BANKSY and annotation tools in containers)

### Dataset
The course uses an **adult mouse brain Stereo-seq** dataset [`adult_mouse_brain_ST4k.h5ad`](https://drive.google.com/uc?id=1lRM-tR1MMbtgyKXLqRzV2YxokweiNUMz).


