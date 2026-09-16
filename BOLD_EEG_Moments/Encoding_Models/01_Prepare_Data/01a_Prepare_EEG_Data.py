"""
Prepares the preprocessed EMD (EEG Moments Dataset) responses: per session
z-scoring, concatenation across sessions, then repeat-averaging per video
condition. Saves each EEG subject separately plus a version appended across
subjects along the channel dimension. The training set is saved three times --
averaged across all repeats, across even repeats only, and across odd repeats
only (for the downstream SFEF analysis); the test set is averaged across all
repeats. In addition, a non-averaged ("single_trial_") version of both train
and test is saved, keeping every individual repeat rather than collapsing it.

Even/odd refers to a video's repeat presentations in acquisition order: even
takes the 1st, 3rd, 5th ... presentation, odd takes the 2nd, 4th, 6th ...

Single-trial data are stored as (n_videos, n_repeats, n_channels, n_time)
arrays. If a video condition has fewer repeats than the max observed for its
split (train/test), the missing repeats are padded with NaN (then zeroed, same
convention as the trial-averaged arrays below) so all conditions share one
rectangular array.

Parameters
----------
eeg_subjects : EEG participants to process.
save_dir : where the prepared .npy files are written.
"""

import os
import argparse
import time

import numpy as np
import h5py
from scipy.stats import zscore
from tqdm import tqdm

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--eeg_subjects', type=int, nargs='+',
                     default=[1, 2, 3, 4, 5, 6])
parser.add_argument('--emd_dir', type=str, default='/scratch/giffordale95/projects/eeg_moments_dataset')
parser.add_argument('--save_dir', type=str, default='/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_data')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

print('>>> Preparing BOLD EMD (EEG Moments Dataset) <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

TRAIN_VIDEO_IDS = np.arange(1, 1001)
TEST_VIDEO_IDS = np.arange(1001, 1103)
N_SESSIONS = 8
# Accumulators for the appended-across-subjects version (channel axis)
appended = {
    'train_all': [], 'train_even': [], 'train_odd': [], 'test': [],
    'train_single_trial': [], 'test_single_trial': [],
}


def average_repeats(eeg, stimulus_id, video_ids, repeat_selector=None):
    """
    Average the EEG responses of each video condition across its repeats.
    repeat_selector optionally subsets a video's repeats (e.g. every other one).
    """
    out = np.zeros((len(video_ids), eeg.shape[1], eeg.shape[2]), dtype=np.float32)
    for v, video in enumerate(video_ids):
        idx = np.where(stimulus_id == video)[0]
        if repeat_selector is not None:
            idx = repeat_selector(idx)
        out[v] = np.nanmean(eeg[idx], 0)
    return out


def collect_single_trials(eeg, stimulus_id, video_ids):
    """
    Collect each video condition's individual (non-averaged) repeat responses.
    Conditions with fewer repeats than the max observed across video_ids are
    padded with NaN so the result is one rectangular
    (n_videos, n_repeats, n_channels, n_time) array.
    """
    repeat_counts = [int(np.sum(stimulus_id == video)) for video in video_ids]
    n_repeats_max = max(repeat_counts)
    out = np.full((len(video_ids), n_repeats_max, eeg.shape[1], eeg.shape[2]),
                  np.nan, dtype=np.float32)
    for v, video in enumerate(video_ids):
        idx = np.where(stimulus_id == video)[0]
        out[v, :len(idx)] = eeg[idx]
    return out


# =============================================================================
# Load the EEG responses for train and test videos
# =============================================================================
for es, esub in enumerate(tqdm(args.eeg_subjects, desc='EEG subjects')):

    data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg', f'sub-{esub:02}')
    metadata = np.load(os.path.join(data_dir, f'sub-{esub:02}_eeg_metadata.npy'),
                       allow_pickle=True).item()

    eeg_sub = []
    stimulus_id_sub = []
    for ses in tqdm(range(1, N_SESSIONS + 1), desc=f'sub-{esub:02} sessions', leave=False):
        file_name = f'sub-{esub:02}_ses-{ses:02}_preprocessed_eeg.h5'
        eeg_ses = h5py.File(os.path.join(data_dir, file_name), 'r')['eeg']
        # Z-score the EEG responses at each session (all time points kept)
        eeg_ses = zscore(eeg_ses[:], axis=0, nan_policy='omit')
        eeg_sub.append(eeg_ses)
        stimulus_id_sub.append(metadata['stimulus_id'][f'ses-{ses:02}'])
        del eeg_ses
    eeg_sub = np.concatenate(eeg_sub, 0)
    stimulus_id_sub = np.concatenate(stimulus_id_sub, 0)

    print(f"\n[sub-{esub:02}] concatenated EEG: {eeg_sub.shape}")

    # =========================================================================
    # Average the EEG responses of each video condition across repeats, and
    # separately collect the non-averaged (single-trial) responses
    # =========================================================================
    splits = {
        'train_all': average_repeats(eeg_sub, stimulus_id_sub, TRAIN_VIDEO_IDS),
        'train_even': average_repeats(eeg_sub, stimulus_id_sub, TRAIN_VIDEO_IDS,
                                      lambda idx: idx[0::2]),
        'train_odd': average_repeats(eeg_sub, stimulus_id_sub, TRAIN_VIDEO_IDS,
                                     lambda idx: idx[1::2]),
        'test': average_repeats(eeg_sub, stimulus_id_sub, TEST_VIDEO_IDS),
        'train_single_trial': collect_single_trials(eeg_sub, stimulus_id_sub, TRAIN_VIDEO_IDS),
        'test_single_trial': collect_single_trials(eeg_sub, stimulus_id_sub, TEST_VIDEO_IDS),
    }
    del eeg_sub, stimulus_id_sub

    # Set NaN values to 0 (both the trial-averaged NaNs and the single-trial
    # padding NaNs from conditions with fewer-than-max repeats)
    for key in splits:
        splits[key] = np.nan_to_num(splits[key], nan=0)

    print(f"[sub-{esub:02}] train {splits['train_all'].shape}, "
          f"train_even {splits['train_even'].shape}, "
          f"train_odd {splits['train_odd'].shape}, test {splits['test'].shape}, "
          f"train_single_trial {splits['train_single_trial'].shape}, "
          f"test_single_trial {splits['test_single_trial'].shape}")

    # =========================================================================
    # Save this subject's prepared data
    # =========================================================================
    extras = {}
    if 'times' in metadata:
        extras['times'] = metadata['times']
    if 'ch_names' in metadata:
        extras['ch_names'] = metadata['ch_names']

    for trial_avg, key in [('all', 'train_all'), ('even', 'train_even'), ('odd', 'train_odd')]:
        save_path = os.path.join(
            args.save_dir, f'eeg_train_sub-{esub:02d}_trial_avg-{trial_avg}.npy')
        np.save(save_path, {'eeg_train': splits[key], 'video_ids': TRAIN_VIDEO_IDS, **extras})
        print(f"Saved: {save_path}")

    save_path = os.path.join(args.save_dir, f'eeg_test_sub-{esub:02d}.npy')
    np.save(save_path, {'eeg_test': splits['test'], 'video_ids': TEST_VIDEO_IDS, **extras})
    print(f"Saved: {save_path}")

    save_path = os.path.join(args.save_dir, f'eeg_train_single_trial_sub-{esub:02d}.npy')
    np.save(save_path, {'eeg_train': splits['train_single_trial'], 'video_ids': TRAIN_VIDEO_IDS,
                        **extras})
    print(f"Saved: {save_path}")

    save_path = os.path.join(args.save_dir, f'eeg_test_single_trial_sub-{esub:02d}.npy')
    np.save(save_path, {'eeg_test': splits['test_single_trial'], 'video_ids': TEST_VIDEO_IDS,
                        **extras})
    print(f"Saved: {save_path}")

    for key in appended:
        appended[key].append(splits[key])
    del splits

# =============================================================================
# Append the EEG across subjects along the channel dimension and save
# =============================================================================
print('\n>>> Appending across subjects (channel dimension) <<<')

# Trial-averaged arrays are (n_videos, n_channels, n_time) -> channel axis = 1.
# Single-trial arrays are (n_videos, n_repeats, n_channels, n_time) -> channel
# axis = 2.
channel_axis = {
    'train_all': 1, 'train_even': 1, 'train_odd': 1, 'test': 1,
    'train_single_trial': 2, 'test_single_trial': 2,
}
for key in appended:
    appended[key] = np.concatenate(appended[key], axis=channel_axis[key])

print(f"train {appended['train_all'].shape}, train_even {appended['train_even'].shape}, "
      f"train_odd {appended['train_odd'].shape}, test {appended['test'].shape}, "
      f"train_single_trial {appended['train_single_trial'].shape}, "
      f"test_single_trial {appended['test_single_trial'].shape}")

for trial_avg, key in [('all', 'train_all'), ('even', 'train_even'), ('odd', 'train_odd')]:
    save_path = os.path.join(
        args.save_dir, f'eeg_train_multi_subject_trial_avg-{trial_avg}.npy')
    np.save(save_path, {'eeg_train': appended[key], 'video_ids': TRAIN_VIDEO_IDS,
                        'eeg_subjects': args.eeg_subjects})
    print(f"Saved: {save_path}")

save_path = os.path.join(args.save_dir, 'eeg_test_multi_subject.npy')
np.save(save_path, {'eeg_test': appended['test'], 'video_ids': TEST_VIDEO_IDS,
                    'eeg_subjects': args.eeg_subjects})
print(f"Saved: {save_path}")

save_path = os.path.join(args.save_dir, 'eeg_train_single_trial_multi_subject.npy')
np.save(save_path, {'eeg_train': appended['train_single_trial'], 'video_ids': TRAIN_VIDEO_IDS,
                    'eeg_subjects': args.eeg_subjects})
print(f"Saved: {save_path}")

save_path = os.path.join(args.save_dir, 'eeg_test_single_trial_multi_subject.npy')
np.save(save_path, {'eeg_test': appended['test_single_trial'], 'video_ids': TEST_VIDEO_IDS,
                    'eeg_subjects': args.eeg_subjects})
print(f"Saved: {save_path}")

print(f"\nDone! Total Time: {time.time() - start_time:.2f} seconds.")