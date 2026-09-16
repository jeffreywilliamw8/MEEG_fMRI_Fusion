"""
This code was sourced from the NSD Visuo-Semantics repo, and used as-is for the current project. The original code can be found here:
https://github.com/adriendoerig/visuo_llm/blob/main/src/nsd_visuo_semantics/utils/nsd_get_data_light.py
"""

import glob
import json
import os
import re
import nibabel as nb
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import zscore
# from nsd_visuo_semantics.utils.utils import average_over_conditions


def get_model_rdms(models_dir, subj, filt=None, only_names=False):
    # filt is model name (e.g. fasttext_verbs_mean) - careful, uses a wildcard, so the wildcard must be specific
    if filt is not None:
        model_files = glob.glob(os.path.join(models_dir, f"{subj}_{filt}*_fullrdm.npy"))
    else:
        model_files = glob.glob(os.path.join(models_dir, f"{subj}*_fullrdm.npy"))
    model_files.sort()

    model_names = [re.split(f"{subj}_", re.split("_fullrdm.npy", os.path.basename(model_file))[0])[1]
        for model_file in model_files]

    if not only_names:
        all_rdms = [np.load(model_file).astype(np.float32) for model_file in model_files]
        if len(all_rdms) == 0:
            raise Exception(f"No rdm files found in {models_dir}.")
        return all_rdms, model_names
    else:
        return model_names


def get_masks(nsd_dir, sub, targetspace="func1pt8mm"):
    """[summary]

    Args:
        nsd_dir ([type]): [description]
        sub ([type]): [description]
        targetspace (str, optional): [description]. Defaults to 'func1pt8mm'.

    Returns:
        [type]: [description]
    """
    # initiate nsda
    ppdata_folder = os.path.join(nsd_dir, "nsddata", "ppdata")

    full_path = os.path.join(
        ppdata_folder, sub, targetspace, "brainmask.nii.gz"
    )

    brainmask = nb.load(full_path).get_fdata()

    return brainmask


def read_behavior(nsd_dir, subject, session_index, trial_index=[]):
    """read_behavior [summary]

    Parameters
    ----------
    subject : str
        subject identifier, such as 'subj01'
    session_index : int
        which session, counting from 0
    trial_index : list, optional
        which trials from this session's behavior to return, by default [], which returns all trials

    Returns
    -------
    pandas DataFrame
        DataFrame containing the behavioral information for the requested trials
    """
    nsd_folder = nsd_dir
    ppdata_folder = os.path.join(nsd_folder, "nsddata", "ppdata")

    behavior_file = os.path.join(
        ppdata_folder, f"{subject}", "behav", "responses.tsv"
    )

    behavior = pd.read_csv(behavior_file, delimiter="\t")

    # the behavior is encoded per run.
    # I'm now setting this function up so that it aligns with the timepoints in the fmri files,
    # i.e. using indexing per session, and not using the 'run' information.
    session_behavior = behavior[behavior["SESSION"] == session_index]

    if len(trial_index) == 0:
        trial_index = slice(0, len(session_behavior))

    return session_behavior.iloc[trial_index]


def average_over_conditions(data, conditions, conditions_to_avg):
    lookup = np.unique(conditions_to_avg)
    n_conds = lookup.shape[0]
    n_dims = data.ndim

    if n_dims == 2:
        n_voxels, _ = data.shape
        avg_data = np.empty((n_voxels, n_conds))
    else:
        x, y, z, _ = data.shape
        avg_data = np.empty((x, y, z, n_conds))

    for j, x in enumerate(lookup):
        conditions_bool = conditions == x
        if n_dims == 2:
            if np.sum(conditions_bool) == 0:
                break
            # print((j, np.sum(conditions_bool)))
            sliced = data[:, conditions_bool]

            avg_data[:, j] = np.nanmean(sliced, axis=1)
        else:
            avg_data[:, :, :, j] = np.nanmean(
                data[:, :, :, conditions_bool], axis=3
            )

    return avg_data


def load_or_compute_betas_average(betas_file, nsd_dir, subj, n_sessions, conditions, conditions_sampled, targetspace):
    
    if not os.path.exists(betas_file):
        print('betas average not found, computing..')
        print('\tloading betas')

        # get betas
        betas = get_betas(nsd_dir, subj, n_sessions, targetspace=targetspace)

        # concatenate trials
        print('\tconcatenating betas across runs..')
        betas = np.concatenate(betas, axis=-1)

        # average betas across three repeats
        print(f'\taveraging betas for {subj}')
        betas = average_over_conditions(betas, conditions, conditions_sampled)

        # saving betas
        print(f'saving betas for {subj}')
        np.save(betas_file, betas, allow_pickle=True)
        
    else:
        print(f'loading betas for {subj}')
        betas = np.load(betas_file, allow_pickle=True)

    return betas


def get_betas(nsd_dir, sub, n_sessions, mask=None, targetspace="func1pt8mm"):
    
    nsddata_betas_folder = os.path.join(nsd_dir, "nsddata_betas", "ppdata")
    data_folder = os.path.join(nsddata_betas_folder, sub, targetspace, "betas_fithrf_GLMdenoise_RR")

    betas = []
    # loop over sessions
    for ses in range(n_sessions):
        ses_i = ses + 1
        si_str = str(ses_i).zfill(2)  # e.g. '01'

        print(f"\r\t\tsub: {sub} fetching betas for trials in session: {ses_i}", end='')
        this_ses = read_behavior(nsd_dir, subject=sub, session_index=ses_i)
        # these are the 73K ids.
        ses_conditions = np.asarray(this_ses["73KID"])
        valid_trials = [j for j, x in enumerate(ses_conditions)]

        # this skips if say session 39 doesn't exist for subject x
        if valid_trials:
            if targetspace == "fsaverage":
                # no need to divide by 300 in this case
                cond_axis = -1
                # load lh
                img_lh = nb.load(os.path.join(data_folder, f"lh.betas_session{si_str}.mgh")).get_fdata().squeeze()
                # load rh
                img_rh = nb.load(os.path.join(data_folder, f"rh.betas_session{si_str}.mgh")).get_fdata().squeeze()
                # concatenate
                all_verts = np.vstack((img_lh, img_rh))
                # mask
                if mask is not None:
                    betas.append((zscore(all_verts, axis=cond_axis)[mask, :]).astype(np.float32))
                else:
                    betas.append((zscore(all_verts, axis=cond_axis)).astype(np.float32))

            elif targetspace == "func1pt8mm":
                # we will need to divide the loaded data by 300 in this case
                cond_axis = -1
                img = nb.load(os.path.join(data_folder, f"betas_session{si_str}.nii.gz")).get_fdata().squeeze()
                # img = nb.load(os.path.join(data_folder, f"betas_session{si_str}.nii.gz"))
                # re-hash the betas to save memory
                if mask is not None:
                    betas.append((zscore(img/300., axis=cond_axis)[mask, :]).astype(np.float32))
                else:
                    betas.append((zscore(img/300., axis=cond_axis)).astype(np.float32))

            else:
                raise Exception("targetspace not recognized")

    return betas


def get_conditions(nsd_dir, sub, n_sessions):
    """[summary]

    Args:
        nsd_dir ([type]): [description]
        sub ([type]): [description]
        n_sessions ([type]): [description]

    Returns:
        [type]: [description]
    """

    # read behaviour files for current subj
    conditions = []

    # loop over sessions
    for ses in range(n_sessions):
        ses_i = ses + 1
        print(f"\r\t\tsub: {sub} fetching condition trials in session: {ses_i}", end='')

        this_ses = np.asarray(read_behavior(nsd_dir, subject=sub, session_index=ses_i)["73KID"])

        # these are the 73K ids.
        valid_trials = [j for j, x in enumerate(this_ses)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
        if valid_trials:
            conditions.append(this_ses)

    return conditions



def get_subject_conditions(nsd_dir, subj, n_sessions, keep_only_3repeats=True):

    # extract conditions data.
    # NOTES ABOUT HOW THIS WORKS:
    # get_conditions returns a list with one item for each session the subject attended. Each of these items contains
    # the NSD_ids for the images presented in that session. Then, we reshape all this into a single array, which now
    # contains all the NSD_ids for the subject, in the order in which they were shown. Next, we create a boolean list of
    # the same size as the conditions array, which assigns True to NSD_ids that are present 3x in the condition array.
    # We use this boolean to create conditions_sampled, which now contains all NSD_indices for stimuli the subject has
    # seen 3x. This list still contains the 3 repetitions of each stimulus, and is still in the stimulus presentation
    # order. For example: [46003, 61883,   829, ...]
    # Hence, we need to only keep each NSD_id once (since we compute everything on the average fMRI data over
    # the 3 presentations), and we also need to order them in increasing NSD_id order (so that we can then easily
    # for all subjects/models). Both of these desiderata are addressed by using np.unique (which sorts the unique idx).
    # So sample contains the unique NSD_ids for that subject, in increasing order (e.g. [ 14,  28,  72, ...]).
    # Importantly, the average betas loaded above are arranged in the same way, so that if we want to find the betas
    # for NSD_id=72, we just need to find the idx of 72 in sample (in the present example: 2). Using this method, we can
    # find the avg_betas corresponding to the shared 515 images as done below with subj_indices_515 (hint: the trick to
    # go from an ordered list of nsd_ids to finding the idx as described above is to use enumerate).
    # For example sample[subj_indices_515[0]] = conditions_515[0].

    # extract conditions data
    conditions = get_conditions(nsd_dir, subj, n_sessions)
    # we also need to reshape conditions to be ntrials x 1
    conditions = np.asarray(conditions).ravel()
    if keep_only_3repeats:
        # then we find the valid trials for which we do have 3 repetitions.
        conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
    else:
        conditions_bool = [True for x in conditions]
    # and identify those.
    conditions_sampled = conditions[conditions_bool]
    # find the subject's condition list (sample pool)
    # this sample is the same order as the betas
    sample = np.unique(conditions[conditions_bool])

    return conditions, conditions_sampled, sample



def get_conditions_1000(nsd_dir):
    """[get condition indices for the special 1000 images.]

    Arguments:
        nsd_dir {[os.path]} -- [where is the nsd data?]

    Returns:
        [lit of inds] -- [indices related to the 1000 special
                          stimuli in a coco format]
    """
    stim1000_dir = os.path.join(
        nsd_dir, "nsddata", "stimuli", "nsd", "shared1000", "*.png"
    )

    stim1000 = [os.path.basename(x)[:-4] for x in glob.glob(stim1000_dir)]
    stim1000.sort()
    stim_ids = [
        int(re.split("nsd", stim1000[x])[1]) for x, n in enumerate(stim1000)
    ]

    stim_ids = list(np.asarray(stim_ids))
    return stim_ids


def get_conditions_100(nsd_dir):
    """[get condition indices for the special chosen 100 images.]

    Arguments:
        nsd_dir {[os.path]} -- [where is the nsd data?]

    Returns:
        [lit of inds] -- [indices related to the chosen 100 special stimuli in a coco format]
    """

    stim_ids = get_conditions_1000(nsd_dir)
    # kendrick's chosen 100
    chosen_100 = [
        4,
        8,
        22,
        30,
        33,
        52,
        64,
        69,
        73,
        137,
        139,
        140,
        145,
        157,
        159,
        163,
        186,
        194,
        197,
        211,
        234,
        267,
        287,
        300,
        307,
        310,
        318,
        326,
        334,
        350,
        358,
        362,
        369,
        378,
        382,
        404,
        405,
        425,
        463,
        474,
        487,
        488,
        491,
        498,
        507,
        520,
        530,
        535,
        568,
        570,
        579,
        588,
        589,
        591,
        610,
        614,
        616,
        623,
        634,
        646,
        650,
        689,
        694,
        695,
        700,
        727,
        730,
        733,
        745,
        746,
        754,
        764,
        768,
        786,
        789,
        790,
        797,
        811,
        825,
        853,
        857,
        869,
        876,
        882,
        896,
        905,
        910,
        925,
        936,
        941,
        944,
        948,
        960,
        962,
        968,
        969,
        974,
        986,
        991,
        999,
    ]
    chosen_100 = np.asarray(chosen_100) - 1

    chosen_ids = list(np.asarray(stim_ids)[chosen_100])

    return chosen_ids


def get_conditions_515(nsd_dir, n_sessions=40):
    """[get condition indices for the special 515 images.]

    Arguments:
        nsd_dir {[os.path]} -- [where is the nsd data?]

    Returns:
        [lit of inds] -- [indices related to the special 515
                          stimuli in a coco format]
    """
    stim_1000 = get_conditions_1000(nsd_dir)

    sub_conditions = []
    # loop over sessions
    for sub in range(8):
        subix = f"subj0{sub+1}"
        # extract conditions data and reshape conditions to be ntrials x 1
        conditions = np.asarray(get_conditions(nsd_dir, subix, n_sessions)).ravel()

        # find the 3 repeats
        conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]

        conditions = conditions[conditions_bool]

        conditions_1000 = [x for x in stim_1000 if x in conditions]
        print(f"{subix} saw {len(conditions_1000)} of the 1000")

        if sub == 0:
            sub_conditions = conditions_1000
        else:
            sub_conditions = [x for x in conditions_1000 if x in sub_conditions]

    return sub_conditions


def get_sentence_lists(nsda, image_indices, return_coco_ids=False):
    """gets a list of captions from nsd given indices
    nsda must be an instance of NSDAccess: nsda = NSDAccess(nsd_dir)"""

    print('Careful with the indices! You may need to subtract 1 from them.')

    # Read in captions
    # print('reading coco captions for the requested images')
    captions = nsda.read_image_coco_info(image_indices, info_type="captions", show_annot=False)

    sentence_lists = []
    coco_ids = []
    for caption in captions:
        image_capt = []
        for j, cap in enumerate(caption):
            image_capt.append(cap["caption"])
        coco_ids.append(caption[0]["image_id"])
        sentence_lists.append(image_capt)

    if return_coco_ids:
        return sentence_lists, coco_ids
    else:
        return sentence_lists



def get_rois(which_rois, roi_defs_dir):
    roi_names_file = os.path.join(roi_defs_dir, f"{which_rois}.mgz.ctab")
    try:
        with open(roi_names_file) as f:
            # get ROI names automatically. If you don't have the .ctab file
            # you can also enter them by hand. 0 is always "Unknown")
            roi_id2name = {int(x[0]): x[2:-1] for x in f}
    except ValueError:
        print(
            f"roi_names_file not found. Requested {roi_names_file}. Using {which_rois} as single ROI name."
        )
        roi_id2name = {0: "Unknown"}
        roi_id2name[1] = which_rois

    # load the roi masks
    try:
        lh_file = os.path.join(roi_defs_dir, f"lh.{which_rois}.mgz")
        rh_file = os.path.join(roi_defs_dir, f"rh.{which_rois}.mgz")
        maskdata_lh = nb.load(lh_file).get_fdata().squeeze()
        maskdata_rh = nb.load(rh_file).get_fdata().squeeze()
    except ValueError:
        lh_file = os.path.join(roi_defs_dir, f"lh.{which_rois}.npy")
        rh_file = os.path.join(roi_defs_dir, f"rh.{which_rois}.npy")
        maskdata_lh = np.load(lh_file, allow_pickle=True)
        maskdata_rh = np.load(rh_file, allow_pickle=True)

    maskdata = np.hstack((maskdata_lh, maskdata_rh))

    return maskdata, roi_id2name


import os
import h5py
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.ndimage import label
from berg import BERG




def load_fmri_hemi_data(subject, hemisphere):

    data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
    train_filename = f'fmri_train_sub-{subject:02d}_hemi-{hemisphere}.npy'
    test_filename = f'fmri_test_sub-{subject:02d}_hemi-{hemisphere}.npy'
    fmri_train = np.load(os.path.join(data_dir, train_filename), allow_pickle=True).item()['fmri_train'].astype(np.float32)
    fmri_test_dict = np.load(os.path.join(data_dir, test_filename),allow_pickle=True).item()
    fmri_test = np.mean(fmri_test_dict['fmri_test'], axis=1, dtype=np.float32)
    del fmri_test_dict

    return fmri_train, fmri_test

def load_fmri_roi_data(subject, hemisphere, roi, nc_threshold=0.2):
    data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
    berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
    train_filename = f'fmri_train_sub-{subject:02d}_hemi-{hemisphere}.npy'
    fmri_train = np.load(os.path.join(data_dir, train_filename), allow_pickle=True).item()['fmri_train'].astype(np.float32)

    test_filename = f'fmri_test_sub-{subject:02d}_hemi-{hemisphere}.npy'
    fmri_test_dict = np.load(os.path.join(data_dir, test_filename),allow_pickle=True).item()
    fmri_test = np.mean(fmri_test_dict['fmri_test'], axis=1, dtype=np.float32)
    del fmri_test_dict


    metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
    # Available ROIS:
    # dict_keys(['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FBA-1', 'FBA-2', 'mTL-bodies', 'OFA', 'FFA-1',
    #  'FFA-2', 'mTL-faces', 'aTL-faces', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words', 'early', 
    # 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal', 'nsdgeneral'])
    # Selecting the ROI indices
    roi_idx = metadata['fmri'][f'{hemisphere}_fsaverage_rois'][roi]
    roi_mask = np.zeros(fmri_train.shape[1], dtype=bool)
    roi_mask[roi_idx] = True
    wb_noise_ceilings = metadata['fmri'][f'{hemisphere}_ncsnr']
    nc_mask = np.zeros(fmri_train.shape[1], dtype=bool)
    nc_idx = np.where(wb_noise_ceilings >= nc_threshold)[0] # Selecting vertices above noise ceiling threshold of 20%
    nc_mask[nc_idx] = True
    valid_vertices = np.where(roi_mask & nc_mask)[0] # Selecting vertices above noise ceiling threshold of 20%
    
    fmri_train = fmri_train[:, valid_vertices].astype(np.float32)
    fmri_test = fmri_test[:, valid_vertices].astype(np.float32)

    del roi_mask, nc_mask
    return fmri_train, fmri_test



def load_fmri_roi_data2(subject, roi, nc_threshold=0.2):
    """
    Loads fMRI data for a given ROI across BOTH hemispheres and any sub-ROIs,
    concatenating them cleanly along the vertex dimension (axis 1).
    
    Parameters:
    - subject: int (e.g. 1, 6, 8)
    - roi: str (e.g. 'V1', 'FFA', 'V4')
    - roi_groups: dict, mappings of composite regions to atomic lists
    - nc_threshold: float, noise ceiling filtration cutoff
    """
    # 1. Determine the target sub-ROIs to load via your structural dictionary lookup
    roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V2': ['V2v', 'V2d'],
    'V3': ['V3v', 'V3d'],
    'hV4': ['hV4'],
    'FFA': ['FFA-1', 'FFA-2'],
    'OFA': ['OFA'],
    'EBA': ['EBA'],
    'PPA': ['PPA']
    }
    sub_rois = roi_groups[roi]
    train_vertex_pool = []
    test_vertex_pool = []
    
    # 2. Cycle systematically through both hemispheres and all sub-components
    for hemisphere in ['lh', 'rh']:
        for sub_roi in sub_rois:
            try:
                # Leverage previous code to load each sub-ROI's data with noise ceiling filtration
                f_train, f_test = load_fmri_roi_data(
                    subject=subject, 
                    hemisphere=hemisphere, 
                    roi=sub_roi, 
                    nc_threshold=nc_threshold
                )
                
                # Check to prevent adding empty arrays if an entire ROI fails the noise ceiling check
                if f_train.shape[1] > 0:
                    train_vertex_pool.append(f_train)
                    test_vertex_pool.append(f_test)
                    
            except KeyError:
                # Catches instances where a sub-ROI might only exist in one hemisphere profile
                print(f"Warning: {sub_roi} not found in {hemisphere} metadata index grid. Skipping.")
                continue

    # 3. Handle combining the pools across the spatial vertex dimension (Axis 1)
    if len(train_vertex_pool) > 0:
        combined_train = np.concatenate(train_vertex_pool, axis=1)
        combined_test = np.concatenate(test_vertex_pool, axis=1)
    else:
        # Fallback security if absolutely zero vertices in the entire ROI survive the noise ceiling filter
        print(f"CRITICAL: Zero vertices passed NCSNR >= {nc_threshold} for macro-ROI: {roi}")
        combined_train = np.empty((0, 0), dtype=np.float32)
        combined_test = np.empty((0, 0), dtype=np.float32)
        
    return combined_train, combined_test



def get_roi_noise_ceiling_corr(roi, subject):
    """
    Computes the average correlation noise ceiling for a given macro-ROI 
    by pooling ALL vertices across both hemispheres and all sub-ROIs.
    
    Parameters:
    - roi: str (e.g., 'V1', 'FFA', 'PPA')
    - subject: int (e.g., 1, 4, 5)
    
    Returns:
    - float: The average correlation noise ceiling value across all vertices in the ROI.
    """
    # 1. Structural mapping of macro-ROIs to underlying sub-ROIs
    roi_groups = {
        'V1': ['V1v', 'V1d'],
        'V2': ['V2v', 'V2d'],
        'V3': ['V3v', 'V3d'],
        'hV4': ['hV4'],
        'FFA': ['FFA-1', 'FFA-2'],
        'OFA': ['OFA'],
        'EBA': ['EBA'],
        'PPA': ['PPA']
    }
    
    if roi not in roi_groups:
        raise ValueError(f"Requested ROI '{roi}' is not defined in structural roi_groups mapping.")
        
    sub_rois = roi_groups[roi]
    
    # Initialize your BERG instance (matching your local path structure)
    berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
    
    # Pool to accumulate individual vertex-level correlation ceilings across parts
    pooled_vertex_ceilings = []
    
    # 2. Cycle systematically through both hemispheres and all sub-components
    for hemisphere in ['lh', 'rh']:
        # Fetch metadata once per hemisphere to extract ROI indicators and ncsnr metrics
        metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
        wb_noise_ceilings = metadata['fmri'][f'{hemisphere}_ncsnr']
        
        for sub_roi in sub_rois:
            try:
                # Isolate target sub-ROI vertex indexes
                roi_idx = metadata['fmri'][f'{hemisphere}_fsaverage_rois'][sub_roi]
                
                # Convert the index list/array into valid vertex positions
                valid_vertices = np.array(roi_idx, dtype=int)
                
                if len(valid_vertices) > 0:
                    # Extract the raw SNR (ncsnr) values for all vertices in the ROI
                    vertex_snrs = wb_noise_ceilings[valid_vertices]
                    
                    # Ensure negative SNR estimations from background noise are clipped to 0
                    vertex_snrs = np.clip(vertex_snrs, 0.0, None)
                    
                    # Compute the variance noise ceiling: R^2 = snr / (snr + 1)
                    # Taking the square root gives us the correlation noise ceiling (r) directly
                    vertex_corr_ceilings = np.sqrt(vertex_snrs)
                    
                    # Append these calculations to the overarching spatial pool
                    pooled_vertex_ceilings.extend(vertex_corr_ceilings)
                    
            except KeyError:
                # Gracefully skip if a sub-ROI mapping doesn't exist for the current hemisphere
                continue
                
    # 3. Aggregate spatial calculations into a single macro scalar output
    if len(pooled_vertex_ceilings) > 0:
        roi_noise_ceiling_corr = np.mean(pooled_vertex_ceilings)
    else:
        print(f"Warning: Zero vertices found matching criteria for {roi}. Returning 0.0")
        roi_noise_ceiling_corr = 0.0
        
    return float(roi_noise_ceiling_corr)

def get_eeg_times():
    # Get the time points # !!! Use official time points
    n_times = 615
    times = np.round(np.linspace(-200, 1000, n_times)).astype(int)
    # Account for the 50ms shift in the EEG responses # !!!
    shift = -50
    times = times + shift
    # Only select time points between -100ms and 600ms
    t_start = np.where(times == -100)[0][0]
    t_end = np.where(times == 600)[0][0]
    times = times[t_start:t_end+1]
    return times



def get_significance_mask(correlation_data, alpha=0.05, method='fdr_bh'):
    """
    Performs Fisher z-transform, 1-sample t-test, and Multiple Comparison Correction.
    
    Parameters:
    - correlation_data: array (n_subjects, n_timepoints)
    - alpha: significance level
    - method: 'fdr_bh' for Benjamini-Hochberg, 'bonferroni' for the strict way.
    """
    # 1. Fisher z-transform (handles the r-distribution skew)
    # Note: We use np.clip to avoid infinity if r is exactly 1.0
    z_data = np.arctanh(np.clip(correlation_data, -0.999, 0.999))
    
    # 2. Perform 1-sample t-test across subjects at each timepoint
    t_stats, p_values = stats.ttest_1samp(z_data, 0, axis=0)
    
    # 3. Correct for Multiple Comparisons
    # reject is a boolean mask, pvals_corrected are the adjusted p-values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    
    return pvals_corrected < alpha


def sign_permutation_cluster_test(corr_timecourses, n_permutations=10000, p_thresh=0.05, alpha=0.05):
    """
    Perform a sign-permutation cluster test across subjects' correlation time courses.

    Parameters
    ----------
    corr_timecourses : list of np.ndarray
        Each array has shape (n_timepoints,)
    n_permutations : int
        Number of permutations for the null distribution.
    p_thresh : float
        Cluster-defining threshold (pointwise).
    alpha : float
        Cluster-level corrected significance threshold.

    Returns
    -------
    results : dict
        Contains:
        - 'observed_clusters': list of (indices, cluster_sum, p_value)
        - 'significant_clusters': same, filtered by p < alpha
        - 'average_onset_index': mean index of significant cluster onsets (or None)
    """
    corr_timecourses = np.array(corr_timecourses)  # shape: (n_subjects, n_timepoints)
    n_subj, n_time = corr_timecourses.shape

    # -------------------------
    # 1. Compute observed group mean
    # -------------------------
    mean_corr = np.mean(corr_timecourses, axis=0)

    # -------------------------
    # 2. Build null distribution via sign-flipping
    # -------------------------
    null_distrib = np.zeros((n_permutations, n_time))

    for i in range(n_permutations):
        signs = np.random.choice([-1, 1], size=(n_subj, 1))
        permuted = corr_timecourses * signs
        null_distrib[i, :] = np.mean(permuted, axis=0)

    # -------------------------
    # 3. Define cluster threshold (P < p_thresh)
    # -------------------------
    upper_thr = np.percentile(null_distrib, 100 * (1 - p_thresh / 2))
    lower_thr = np.percentile(null_distrib, 100 * (p_thresh / 2))

    # -------------------------
    # 4. Find clusters in observed data
    # -------------------------
    suprathreshold = (mean_corr > upper_thr) | (mean_corr < lower_thr)
    labeled, n_clusters = label(suprathreshold)

    clusters = []
    for i in range(1, n_clusters + 1):
        idx = np.where(labeled == i)[0]
        cluster_sum = np.sum(mean_corr[idx])
        clusters.append((idx, cluster_sum))

    # -------------------------
    # 5. Null distribution of cluster sums (max per permutation)
    # -------------------------
    max_cluster_sums = np.zeros(n_permutations)
    for i in range(n_permutations):
        suprathreshold_null = (null_distrib[i, :] > upper_thr) | (null_distrib[i, :] < lower_thr)
        labeled_null, n_null = label(suprathreshold_null)
        if n_null > 0:
            cluster_sums_null = [np.sum(null_distrib[i, np.where(labeled_null == j)[0]]) for j in range(1, n_null + 1)]
            max_cluster_sums[i] = np.max(cluster_sums_null)
        else:
            max_cluster_sums[i] = 0

    # -------------------------
    # 6. Compute corrected p-values for observed clusters
    # -------------------------
    results = []
    for idx, cluster_sum in clusters:
        p_val = (np.sum(max_cluster_sums >= np.abs(cluster_sum)) + 1) / (n_permutations + 1)
        results.append((idx, cluster_sum, p_val))

    # Filter significant clusters
    significant_clusters = [r for r in results if r[2] < alpha]

    # -------------------------
    # 7. Compute average onset index
    # -------------------------
    if significant_clusters:
        onsets = [r[0][0] for r in significant_clusters]  # first index of each cluster
        avg_onset = int(np.mean(onsets))
    else:
        avg_onset = None
    
    # -------------------------
    # 8. Compute per-subject onset list
    # -------------------------
    subject_onsets = []
    for subj_corr in corr_timecourses:
        subj_suprathreshold = (subj_corr > upper_thr) | (subj_corr < lower_thr)
        if np.any(subj_suprathreshold):
            onset_idx = int(np.argmax(subj_suprathreshold))  # first True
        else:
            onset_idx = None
        subject_onsets.append(onset_idx)

    return {
        "observed_clusters": results,
        "significant_clusters": significant_clusters,
        "onsets": subject_onsets,  # <--- list of 6 onset indices
        "mean_corr": mean_corr,
        "upper_thr": upper_thr,
        "lower_thr": lower_thr
    }

