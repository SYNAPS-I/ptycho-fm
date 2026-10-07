#!/bin/bash

# Launch this from the top-level dir of repo as bash docker/build.sh

# We use the NERSC private container registry here
# should be shared across the group following registry.nersc.gov/
# Set base nvcr.io pytorch container version with NVC_TAG
# To access the registry, do: podman-hpc login registry.nersc.gov
# See https://docs.nersc.gov/development/shifter/how-to-use/#using-registrynerscgov

set -euxo pipefail

NVC_TAG=26.01
BASE=registry.nersc.gov/amsc006/shas1693/ptychofm
IMAGE=$BASE:$NVC_TAG

# Build and push only if every Dockerfile step, including the runtime import
# verification, succeeds.
podman-hpc build --build-arg nvc_tag=$NVC_TAG-py3 -t $IMAGE -f docker/Dockerfile .
podman-hpc push $IMAGE

# Refresh Shifter's converted image after replacing this tag.
shifterimg pull "$IMAGE"

