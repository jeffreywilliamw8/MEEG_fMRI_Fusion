import numpy as np
import os
import random
import argparse
from sklearn.linear_model import LinearRegression, RidgeCV
import tqdm
from utils import load_fmri_roi_data
import h5py
import time

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

#======================================
# Input arguments
#======================================

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh') # lh -> left hemisphere, rh -> right hemisphere
parser.add_argument('--roi', type=str, default='V1v')
parser.add_argument('--cv_split', type=str, default='odd') # Even/odd cross-validation split
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--dnn_type', type=str, default='both', choices=['vdnn', 'llm', 'both'],
                    help='Type of DNN features to use for the joint encoding fusion: "vdnn" for vision DNN features, "llm" for language model features.')
args = parser.parse_args()

print(f'>>> Joint EEG-Features Encoding Fusion Phase 2 (ROI-wise) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#=====================================================
# Loading the EEG responses (odd repeats for phase 2)
#======================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
cv_dict = {
        'even': 'odd', # If 'even' was used for phase 1, 'odd' will be used for phase 2, and vice-versa
        'odd': 'even'
    }
eeg_train = np.load(os.path.join(data_path, f'eeg_train_sub-{args.subject:02d}_trial_avg-{args.cv_split}.npy'), allow_pickle=True).item()['eeg_train'] # Shape: (9000, 160, 359)
print('Shape of the EEG data (train):', eeg_train.shape)


# =============================================================================
# Load the fMRI responses (only the test set are necessary for phase 2)
# =============================================================================
_, fmri_test = load_fmri_roi_data(args.subject, args.hemisphere, args.roi, nc_threshold=0.2) # Shape: (9000, n_vertices), (515, n_vertices)
print('Shape of the fMRI data (test):', fmri_test.shape)
if fmri_test.shape[1]>0:

    #=======================================================================
    # Loading the pre-trained EEG-to-fMRI encoder's weights (from phase 1)
    #=======================================================================
    
    phase_1_weights_path = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/joint_eeg_feature_encoding/roi/phase_1/subject-{args.subject}'
    phase_1_weights = np.load(os.path.join(phase_1_weights_path, f'{args.roi}_{args.hemisphere}_cv_split-{cv_dict[args.cv_split]}.npy'), allow_pickle=True).item()
    print("Loaded pre-trained EEG-to-fMRI encoder's weights")

    #=======================================================================
    # Loading the features data
    #=======================================================================
    if args.dnn_type == 'vdnn':
        features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/vit_b_32'
        features_train = np.load(os.path.join(features_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_train']
        features_test = np.load(os.path.join(features_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_test']
    elif args.dnn_type == 'llm':
        features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/llms'
        features_train = np.load(os.path.join(features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_train']
        features_test = np.load(os.path.join(features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_test']

    elif args.dnn_type == 'both':
        vision_features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/vit_b_32'
        vision_features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/vit_b_32'
        vision_features_train = np.load(os.path.join(vision_features_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_train']
        vision_features_test = np.load(os.path.join(vision_features_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_test']

        language_features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/llms'
        language_features_train = np.load(os.path.join(language_features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_train']
        language_features_test = np.load(os.path.join(language_features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_test']

        features_train = np.concatenate((vision_features_train, language_features_train), axis=1)
        features_test = np.concatenate((vision_features_test, language_features_test), axis=1)

    print("Shape of the features data (train, test):", features_train.shape, features_test.shape) # Should be (9000, 250), (515, 250) or  (9000, 500), (515, 500)

    #=========================================================================
    # Settings for saving the correlation coefficients and regression weights
    # The weights will be used for variance partitioning
    #========================================================================= 
    file_name = f'{args.roi}_{args.hemisphere}_cv_split-{args.cv_split}.npy'

    scores_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/r2_scores/jefe_phase_2/roi/vision_language_models/dnn_type-{args.dnn_type}/subject-{args.subject}'
    if os.path.isdir(scores_save_dir) == False:
        os.makedirs(scores_save_dir)


    weights_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/jefe_phase_2/roi/dnn_type-{args.dnn_type}/subject-{args.subject}'
    if os.path.isdir(weights_save_dir) == False:
        os.makedirs(weights_save_dir)


    #===========================================================================
    # Joint EEG-Feature Encoding Fusion: Predict fMRI from EEG (t-fMRI), 
    # and then train model to predict t-fMRI from the features
    # Testing is done using the test fMRI responses
    #===========================================================================
    encoding_models_weights = {}
    encoding_models_weights['coef_'] = []
    encoding_models_weights['intercept_'] = []
    scores = []
    alphas = np.logspace(-6, 3, 20) # List of alphas for Ridge regression
    print("Starting Joint EEG-Feature Encoding Fusion...")
    for t in tqdm.tqdm(range(eeg_train.shape[2])):
        # Loading the pre-trained EEG-to-fMRI encoder's weights for the current time point
        eeg2fmri = LinearRegression()
        # Loading the linear regression weights
        eeg2fmri.coef_ = phase_1_weights['coef_'][t]
        eeg2fmri.intercept_ = phase_1_weights['intercept_'][t]
        t_fmri = eeg2fmri.predict(eeg_train[:,:,t])

        # Fitting a new linear regression model using the trained predicted t-fMRI as target
        #encoding_model = RidgeCV(alphas=alphas, store_cv_results=True)
        encoding_model = LinearRegression()
        encoding_model.fit(features_train, t_fmri)
        # Store the encoding fusion model weights (will be used later for variance partitioning)
        encoding_models_weights['coef_'].append(encoding_model.coef_.astype(np.float32))
        encoding_models_weights['intercept_'].append(encoding_model.intercept_.astype(np.float32))
        np.save(os.path.join(weights_save_dir, file_name), encoding_models_weights)

        # Evaluating the encoding model and saving the correlation coefficients
        #pred_fmri = encoding_model.predict(features_test)
        scores.append(encoding_model.score(features_test, fmri_test))

        np.save(os.path.join(scores_save_dir, file_name), np.array(scores, dtype=np.float32))
    print("Joint EEG-Feature Encoding Fusion complete!")

else:
     print("No vertices above noise ceiling threshold found in this ROI. Terminating...")


# End time
end_time = time.time()
execution_time = end_time - start_time

print("JEFE Phase 2 complete!")
print(f"Execution time: {execution_time:.2f} seconds.")