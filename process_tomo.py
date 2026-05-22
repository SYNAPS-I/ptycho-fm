import numpy as np
import os

scan_ids, angs = np.loadtxt('chip_ptycho_tomo_startID405699.txt', unpack=True)
scan_ids = scan_ids.astype(int)
angs = angs.astype(float)

scan_basefile = '/nsls2/data2/hxn/legacy/users/2026Q2/ZP_Commissioning_2026Q2/ptycho/scan_%d.h5'
scan_probefile = '/nsls2/data2/hxn/legacy/users/2026Q2/ZP_Commissioning_2026Q2/ptycho/recon_result/S%d/admm/recon_data/recon_%d_admm_probe.npy'
scan_objectfile = '/nsls2/data2/hxn/legacy/users/2026Q2/ZP_Commissioning_2026Q2/ptycho/recon_result/S%d/admm/recon_data/recon_%d_admm_object.npy'
output_dir = '/nsls2/data2/hxn/legacy/home/home/SYNAPS/hgoel1/ptycho-vit-demo/tomo_data'

# Put each scan through scripts/hxn_to_vit.py
for scan_id, ang in zip(scan_ids, angs):
    print(f'Processing scan {scan_id} at angle {ang} degrees...')
    basefile = scan_basefile % scan_id
    probefile = scan_probefile % (scan_id, scan_id)
    objectfile = scan_objectfile % (scan_id, scan_id)
    
    output_file = os.path.join(output_dir, f'{scan_id}_angle_{int(ang)}')
    
    cmd = f'python scripts/hxn_to_vit.py --src-hdf5 {basefile} --src-probe {probefile} --src-object {objectfile} --out-dir {output_file} --scan-id {scan_id}'
    print(f'Running command: {cmd}')
    os.system(cmd)