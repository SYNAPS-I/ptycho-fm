ARG nvc_tag
FROM nvcr.io/nvidia/pytorch:${nvc_tag}

RUN pip install --no-cache-dir \
        cmcrameri \
        fvcore \
        h5py \
        hdf5plugin \
        mlflow \
        mpi4py \
        scikit-image \
        seaborn \
        timm \
        torchinfo \
        wandb \
        "ptychi @ git+https://github.com/AdvancedPhotonSource/pty-chi.git@main" \
    && python -c "import cmcrameri, fvcore, h5py, hdf5plugin, mlflow, mpi4py, skimage, seaborn, timm, torchinfo, wandb; from ptychi.image_proc import extract_patches_fourier_shift; assert all(hasattr(wandb, name) for name in ('login', 'init', 'Settings', 'Api', 'Image', 'Artifact'))"

