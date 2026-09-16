import argparse
import os
import numpy as np
import cortex
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
import time

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--pycortex_filestore', type=str, default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/pycortex_filestore')
parser.add_argument('--transform', type=str, default='align_auto')
args = parser.parse_args()

cortex.database.default_filestore = args.pycortex_filestore
subject_list = ['01', '02', '03']
subject_map = {'01': 'S1', '02': 'S2', '03': 'S3'}

BASE_PLOTS_DIR = '/home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/plots'
os.makedirs(BASE_PLOTS_DIR, exist_ok=True)

def get_volume(correlations, shape, voxel_coords):
    vol = np.zeros(shape)
    vol[voxel_coords[:, 2], voxel_coords[:, 1], voxel_coords[:, 0]] = correlations
    return vol

for sub_id in subject_list:
    print(f"\n Processing Subject {sub_id}...")
    pycortex_subject = subject_map[sub_id]

    fmri_metadata_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/prepared'
    fmri_metadata = np.load(os.path.join(fmri_metadata_dir, f'fmri_{sub_id}_metadata.npy'), allow_pickle=True).item()
    coords = fmri_metadata['fmri']['voxel_coords']
    vol_shape = (coords[:, 2].max() + 1, coords[:, 1].max() + 1, coords[:, 0].max() + 1)

    em_corrs_path = '/home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/correlations/meg2fmri_fusion'
    rsa_corrs_path = '/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/correlations/searchlight_fusion/eeg_rdm_metric-pearsonr/radius-10.0/aggregated_results'

    em_ctc = np.load(os.path.join(em_corrs_path, f'sub-{sub_id}_correlation_time_courses.npy'))[:141, :]
    rsa_ctc = np.load(os.path.join(rsa_corrs_path, f'subject-{sub_id}/subject-{sub_id}_timecourse.npy'))

    print(f"Shape of the correlation time courses (Encoding Models): {em_ctc.shape}")
    print(f"Shape of the correlation time courses (RSA): {rsa_ctc.shape}")

    n_voxels = em_ctc.shape[1]
    similarity_map = np.zeros(n_voxels)

    print(f"Computing Spearman similarity for {n_voxels} voxels...")
    for v in tqdm(range(n_voxels), desc=f"Sub-{sub_id} Voxel Loop"):
        rho, _ = spearmanr(em_ctc[:, v], rsa_ctc[:, v])
        similarity_map[v] = rho

    sim_vol_3d = get_volume(similarity_map, vol_shape, coords)

    volume = cortex.Volume(sim_vol_3d, pycortex_subject, args.transform,
                           vmin=0.0, vmax=max(similarity_map), cmap='viridis')

    png_path = os.path.join(BASE_PLOTS_DIR, f'sub-{sub_id}_wb_ctc_correlations.png')
    svg_path = os.path.join(BASE_PLOTS_DIR, f'sub-{sub_id}_wb_ctc_correlations.svg')
    left_svg_path = os.path.join(BASE_PLOTS_DIR, f'sub-{sub_id}_wb_ctc_correlations_left_hemisphere.svg')
    right_svg_path = os.path.join(BASE_PLOTS_DIR, f'sub-{sub_id}_wb_ctc_correlations_right_hemisphere.svg')

    # No title -- build the figure with make_figure and save it straight to both PNG and SVG
    # with fig.savefig, so the files show exactly the same flatmap (make_svg's own composite
    # pathway is a separate, more limited renderer than make_figure/make_png).
    dpi = 100
    fig = cortex.quickflat.make_figure(volume,
                                       with_curvature=True,
                                       with_rois=True,
                                       with_labels=True,
                                       roi_list=['V1', 'hV4', 'FFA', 'PPA', 'EBA'],
                                       labelsize='36pt')
    imsize = fig.get_axes()[0].get_images()[0].get_size()
    fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))

    # Whole-brain (both hemispheres)
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', transparent=True)
    fig.savefig(svg_path, bbox_inches='tight', transparent=True)

    # Additionally, each hemisphere separately in SVG -- pycortex flatmaps place both
    # hemispheres side by side split at the vertical midline (x=0), so cropping to that half
    # via ax.set_xlim is enough; the underlying data/curvature/ROI artists are unchanged, just
    # re-saved with a narrower view. Check the first output and flip the sign below if your
    # subjects' flatmaps have the hemispheres swapped.
    ax = fig.axes[0]
    xmin, xmax = ax.get_xlim()

    ax.set_xlim(xmin, 0)
    fig.savefig(left_svg_path, bbox_inches='tight', transparent=True)

    ax.set_xlim(0, xmax)
    fig.savefig(right_svg_path, bbox_inches='tight', transparent=True)

    fig.clf()
    plt.close(fig)

    print(f" Saved flatmap to: {png_path}")
    print(f" Saved flatmap to: {svg_path}")
    print(f" Saved flatmap to: {left_svg_path}")
    print(f" Saved flatmap to: {right_svg_path}")

print(f"\n All subjects processed. Total time: {time.time() - start_time:.2f} seconds.")