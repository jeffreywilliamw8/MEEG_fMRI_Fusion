import numpy as np
import pickle


def load_fmri_data(subject, hemisphere, roi, threshold=0.0):

    if hemisphere == 'both':

        fmri_file_train_left = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-train_hemi-left_normalized.pkl'
        fmri_file_test_left = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-test_hemi-left_normalized.pkl'
        fmri_file_train_right = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-train_hemi-right_normalized.pkl'
        fmri_file_test_right = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-test_hemi-right_normalized.pkl'

        rois_masks_dir_l = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_roi_masks_hemi-left.npy'
        noise_ceiling_file_l = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_noiseceiling_space-fsaverage_task-test_hemi-left_n-10.pkl'
        rois_masks_dir_r = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_roi_masks_hemi-right.npy'
        noise_ceiling_file_r = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_noiseceiling_space-fsaverage_task-test_hemi-right_n-10.pkl'



        f_train_l = open(fmri_file_train_left, 'rb')
        f_test_l = open(fmri_file_test_left, 'rb')
        f_train_r = open(fmri_file_train_right, 'rb')
        f_test_r = open(fmri_file_test_right, 'rb')



        fmri_T_l = np.float32(pickle.load(f_train_l)[0])
        fmri_t_l = np.float32(pickle.load(f_test_l)[0])
        fmri_T_r = np.float32(pickle.load(f_train_r)[0])
        fmri_t_r = np.float32(pickle.load(f_test_r)[0])



        noise_ceiling_l = pickle.load(open(noise_ceiling_file_l.format(subject, subject, 'left'), 'rb'))[1]
        noise_ceiling_r = pickle.load(open(noise_ceiling_file_r.format(subject, subject, 'right'), 'rb'))[1]

        if roi=='WB':
            fmri_train_l = np.mean(fmri_T_l, axis=1, dtype=np.float32)[:, noise_ceiling_l >= threshold]
            fmri_test_l = np.mean(fmri_t_l, axis=1, dtype=np.float32)[:, noise_ceiling_l >= threshold]
            fmri_train_r = np.mean(fmri_T_r, axis=1, dtype=np.float32)[:, noise_ceiling_r >= threshold]
            fmri_test_r = np.mean(fmri_t_r, axis=1, dtype=np.float32)[:, noise_ceiling_r >= threshold]
            del fmri_T_l, fmri_t_l, fmri_T_r, fmri_t_r
            return np.concatenate([fmri_train_l, fmri_test_l], axis=0), np.concatenate([fmri_train_r, fmri_test_r], axis=0)
        
        else:
            mask_l = np.load(rois_masks_dir_l.format(subject, subject, 'left'), allow_pickle=True).item()[roi]
            mask_r = np.load(rois_masks_dir_r.format(subject, subject, 'right'), allow_pickle=True).item()[roi]
            # Selecting vertices from the whole-brain surface based on 2 conditions: belonging to the ROI and noise ceiling above the threshold
            # Condition 1: Non-zero values of the mask (selecting vertices belonging to the ROI)
            condition1_l = mask_l != 0
            condition1_r = mask_r != 0


            # Condition 2: Vertices with a noise ceiling greater than the threshold
            condition2_l = noise_ceiling_l >= threshold
            condition2_r = noise_ceiling_r >= threshold

            # Combined conditions
            combined_condition_l = condition1_l & condition2_l
            combined_condition_r = condition1_r & condition2_r

            # Selecting the vertices that satisfy both conditions and Averaging the fMRI responses across repetitions
            fmri_train_l = np.mean(fmri_T_l, axis=1, dtype=np.float32)[:, combined_condition_l]
            fmri_test_l = np.mean(fmri_t_l, axis=1, dtype=np.float32)[:, combined_condition_l]
            del fmri_T_l, fmri_t_l
            fmri_train_r = np.mean(fmri_T_r, axis=1, dtype=np.float32)[:, combined_condition_r]
            fmri_test_r = np.mean(fmri_t_r, axis=1, dtype=np.float32)[:, combined_condition_r]
            del fmri_T_r, fmri_t_r

            print("Number of selected vertices - left hemisphere: (train, test)", fmri_train_l.shape[1], fmri_test_l.shape[1])
            print("Number of selected vertices - right hemisphere: (train, test)", fmri_train_r.shape[1], fmri_test_r.shape[1])
            return np.concatenate([fmri_train_l, fmri_test_l], axis=0), np.concatenate([fmri_train_r, fmri_test_r], axis=0)
    
    else:

        fmri_file_train = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-train_hemi-'+hemisphere+'_normalized.pkl'
        fmri_file_test = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-test_hemi-'+hemisphere+'_normalized.pkl'

        rois_masks_dir = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_roi_masks_hemi-{}.npy'
        noise_ceiling_file = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_noiseceiling_space-fsaverage_task-test_hemi-{}_n-10.pkl'

        f_train = open(fmri_file_train, 'rb')
        f_test = open(fmri_file_test, 'rb')

        fmri_T = np.float32(pickle.load(f_train)[0])
        fmri_t = np.float32(pickle.load(f_test)[0])

        noise_ceiling = pickle.load(open(noise_ceiling_file.format(subject, subject, hemisphere), 'rb'))[1]

        if roi == 'WB':
            fmri_train = np.mean(fmri_T, axis=1, dtype=np.float32)[:, noise_ceiling >= threshold]
            fmri_test = np.mean(fmri_t, axis=1, dtype=np.float32)[:, noise_ceiling >= threshold]
            del fmri_T, fmri_t
            return fmri_train, fmri_test

        else:

            mask = np.load(rois_masks_dir.format(subject, subject, hemisphere), allow_pickle=True).item()[roi]
            # Selecting vertices from the whole-brain surface based on 2 conditions: belonging to the ROI and noise ceiling above the threshold
            # Condition 1: Non-zero values of the mask (selecting vertices belonging to the ROI)
            condition1 = mask != 0


            # Condition 2: Vertices with a noise ceiling greater than the threshold
            condition2 = noise_ceiling >= threshold

            # Combined conditions
            combined_condition = condition1 & condition2

            # Selecting the vertices that satisfy both conditions and Averaging the fMRI responses across repetitions
            fmri_train = np.mean(fmri_T, axis=1, dtype=np.float32)[:, combined_condition]
            fmri_test = np.mean(fmri_t, axis=1, dtype=np.float32)[:, combined_condition]
            del fmri_T, fmri_t

            print("Number of selected vertices : (train, test)", fmri_train.shape[1], fmri_test.shape[1])

            return fmri_train, fmri_test