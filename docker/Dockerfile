ARG nvc_tag
FROM nvcr.io/nvidia/pytorch:$nvc_tag


# Parallel HDF5 (1.12.3) -----------------------------------------------------
RUN wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_12_3.tar.gz && \
    tar -xvf hdf5-1_12_3.tar.gz && \
    cd hdf5-hdf5-1_12_3 && \
    ./configure --enable-parallel --enable-shared --prefix=/opt/hdf5 && \
    make -j$(nproc) && make install && \
    cd / && rm -rf hdf5-hdf5-1_12_3 hdf5-1_12_3.tar.gz

# h5py built against MPI-enabled HDF5
RUN HDF5_MPI=ON CC=mpicc HDF5_DIR=/opt/hdf5 \
    pip install --no-deps h5py

# Python packages -------------------------------------------------------------
RUN pip install --no-cache-dir \
    timm \
    torchinfo \
    mpi4py \
    tensorly tensorly-torch \
    scikit-image \
    xarray zarr netCDF4 pyproj pygrib cdsapi \
    properscoring \
    ruamel.yaml \
    dask \
    wandb \
    black ruff typer \
    parameterized \
    hydra-core \
    seaborn \
    --upgrade

# Clean up
RUN echo "Docker build finished successfully."

