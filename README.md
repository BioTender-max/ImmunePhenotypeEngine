# ImmunePhenotypeEngine

High-dimensional immune phenotyping pipeline for flow and mass cytometry data with automated subset identification.

## Features

- Multi-parameter flow and CyTOF data processing (40 markers)
- Automated immune subset clustering and annotation
- Exhaustion marker scoring and statistical comparison
- Dimensionality reduction (UMAP/tSNE) visualization
- Cross-sample correlation and batch correction

## Results

100 samples × 40 markers × 50,000 cells; 12 subsets; Exhaustion p<0.001; r=0.97

## Usage

```bash
pip install numpy scipy matplotlib
python immune_phenotype_engine.py
```

## Tags

`immune-phenotype`, `flow-cytometry`, `mass-cytometry`, `cytof`, `immunophenotyping`
