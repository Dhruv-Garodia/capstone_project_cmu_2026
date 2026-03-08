PuMA Synthetic Data Generation
==================

This folder holds all scripts and tools used to generate synthetic PFIB datasets
with PuMA (Porous Materials Analysis).

These synthetic volumes are used for:
- U-Net segmentation training
- Testing segmentation pipelines
- Producing synthetic lightened images and ground-truth masks

What this folder does:
- Generates 3D porous microstructures
- Creates ground-truth segmentation masks
- Configure pore distributions and pore structures

Notes:
- This module is standalone and has its own README and configs.
- Used only for generating data, not model training.

Basic usage:
Modify and run the supplied generation script, for example:
    ./make_synthetic_meth.sh
or directly:
    python scripts/generate_synthetic_pfib.py --spec input/config/adjusted_config.txt

This folder is only for synthetic data generation.
