#!/usr/bin/env bash
# Clean setup for pfib_sem environment

# 1. Remove old env if exists
conda deactivate
conda env remove -n pfib_sem || true

# 2. Create fresh env with Python 3.10
conda create -y -n pfib_sem python=3.10
conda activate pfib_sem

# 3. Install build tools and dependencies
conda install -y -c conda-forge numpy scipy meson ninja pkg-config cmake swig openblas

# 4. Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# 5. Set compiler flags for macOS M1/M2
export CFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib"
export ARCHFLAGS="-arch arm64"

# 6. Install scikit-umfpack and pumapy
python -m pip install --no-cache-dir scikit-umfpack pumapy

# 7. Test installations
python -c "import scikits.umfpack; print('scikit-umfpack OK')"
python -c "import pumapy; print('pumapy OK')"

echo "Environment pfib_sem is ready!"