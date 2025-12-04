import os
import argparse
import logging

import pandas as pd
import h5py
import tqdm

logger = logging.getLogger(__name__)


def create_table() -> pd.DataFrame:
    table = pd.DataFrame(
        columns=[
            "dp_path", 
            "dp_height", 
            "dp_width", 
            "n_dps", 
            "n_opr_modes", 
            "n_incoherent_modes", 
            "probe_height", 
            "probe_width", 
            "pixel_size_m",
            "object_height", 
            "object_width"
        ]
    )
    return table


def update_table(
    table: pd.DataFrame, 
    dp_path: str, 
    dp_height: int, 
    dp_width: int, 
    n_dps: int, 
    n_opr_modes: int,
    n_incoherent_modes: int,
    probe_height: int,
    probe_width: int,
    pixel_size_m: float,
    object_height: int = None, 
    object_width: int = None
):
    table.loc[len(table)] = [
        dp_path, 
        dp_height, 
        dp_width, 
        n_dps, 
        n_opr_modes, 
        n_incoherent_modes, 
        probe_height, 
        probe_width, 
        pixel_size_m,
        object_height, 
        object_width
    ]
    return table


def create_index(
    data_root: str,
    output_path: str,
):
    table = create_table()
    
    pbar = tqdm.tqdm()
    for root, dirs, files in os.walk(data_root):
        for dp_fname in files:
            if "dp" in dp_fname and dp_fname.endswith(".hdf5") or dp_fname.endswith(".h5"):
                try:
                    dp_path = os.path.join(root, dp_fname)
                                    
                    with h5py.File(dp_path, "r") as f:
                        n_dps, dp_height, dp_width = f["dp"].shape
                    
                    para_path = os.path.join(root, dp_fname.replace("dp", "para"))
                    if not os.path.exists(para_path):
                        logger.warning("{} does not exist".format(para_path))
                        continue
                    
                    with h5py.File(para_path, "r") as f:
                        if f["probe"].ndim == 4:
                            n_opr_modes, n_incoherent_modes, probe_height, probe_width = f["probe"].shape
                        else:
                            n_opr_modes = 1
                            n_incoherent_modes, probe_height, probe_width = f["probe"].shape
                        
                        pixel_size_m = f["object"].attrs["pixel_height_m"]
                        
                        obj_mag = abs(f["object"][...])
                        if obj_mag.max() - obj_mag.min() > 1e-2:
                            object_height, object_width = obj_mag.shape[-2:]
                        else:
                            object_height, object_width = None, None
                    
                    table = update_table(
                        table=table, 
                        dp_path=dp_path, 
                        dp_height=dp_height, 
                        dp_width=dp_width, 
                        n_dps=n_dps, 
                        n_opr_modes=n_opr_modes, 
                        n_incoherent_modes=n_incoherent_modes, 
                        probe_height=probe_height, 
                        probe_width=probe_width,
                        pixel_size_m=pixel_size_m,
                        object_height=object_height,
                        object_width=object_width
                    )
                except Exception as e:
                    logger.error("Error processing {}: {}".format(dp_path, e))
                    continue
                
                pbar.update(1)
    table.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the converted data root directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file (*.csv)")
    args = parser.parse_args()
    
    create_index(
        data_root=args.data_root,
        output_path=args.output_path,
    )
