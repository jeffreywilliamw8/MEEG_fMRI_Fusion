# M/EEG-fMRI Fusion

This is the accompanying GitHub repository of the paper "M/EEG-fMRI fusion using encoding models and Representational Similarity Analysis: a tutorial. The paper presents two methods to combine the high temporal resolution of M/EEG with the high spatial resolution of fMRI, resolving visual processing in both space and time: **encoding-based fusion**, which predicts fMRI responses from M/EEG responses with regression models, and **RSA-based fusion**, which compares the representational geometries (pairwise stimulus dissimilarities) of the two modalities. Both approaches link M/EEG and fMRI through their responses to the same stimuli, so the two modalities can come from separate sessions and different participants.

<!--
<p align="center">
  <img src="figure_1.png" alt="Figure 1" width="750"><br>
  <em><strong>Figure 1.</strong> <CAPTION></em>
</p>
-->



## 🗂️ Datasets

* **Natural Scenes Dataset (NSD).** [NSD](https://doi.org/10.1038/s41593-021-00962-x) consists of 7T fMRI responses of 8 participants, each viewing up to 10,000 natural scene images from the COCO database. Here it is paired with EEG responses to the NSD images. The NSD EEG data is not yet publicly available, and will be released together with its dedicated data paper.
* **THINGS.** [THINGS-data](https://doi.org/10.7554/eLife.82580) is a multimodal collection of large-scale datasets of responses to images of objects from the THINGS database. Here we use its MEG (4 participants) and 3T fMRI (3 participants) datasets, through the 8,640 training images and 100 test images shared by both.
* **BOLD Moments and EEG Moments (BMD/EMD).** The [BOLD Moments Dataset (BMD)](https://doi.org/10.1038/s41467-024-50310-3) consists of fMRI responses of 10 participants viewing 1,102 short naturalistic videos (3-second long; 1,000 training and 102 test videos). Its companion dataset, the [EEG Moments Dataset (EMD)](https://doi.org/10.48550/arXiv.2608.28768), consists of 128-channel EEG responses of 6 participants to the same 1,102 videos.



## 🔬 Analyses

All analyses follow the same two-part structure across datasets.

### Encoding-based fusion

* **M/EEG-to-fMRI encoding fusion:** At each M/EEG time point, a ridge regression predicts fMRI responses from M/EEG sensor patterns, and the predicted and actual test fMRI responses are correlated. This gives a fusion time course for every fMRI vertex/voxel, analyzed at the ROI and whole-brain level.
* **Stimulus feature encoding fusion (SFEF):** Adds a model of the stimuli, to reveal the representational format of the brain responses captured by M/EEG and fMRI. In phase 1, M/EEG-to-fMRI encoding models are trained on one half of the training data. In phase 2, they are applied to the M/EEG responses of the other half to obtain time-resolved predicted fMRI (t-fMRI), and a second encoding model predicts the t-fMRI from stimulus features, evaluated on the actual test fMRI. The results are averaged over both assignments of the two halves. Stimulus features include layerwise deep neural network activations (AlexNet for images, r3d_18 for videos), and vision (VDNN) and language (LLM) model features.
* **Unique VDNN vs LLM contributions:** Partial correlation between the actual test fMRI and the VDNN-based (or LLM-based) SFEF prediction, controlling for the other model's prediction.

### RSA-based fusion

* **RSA fusion:** At each M/EEG time point, the M/EEG RDM is correlated (Spearman) with the fMRI RDM of an ROI or of a searchlight neighbourhood around every vertex/voxel. M/EEG RDMs are computed per participant and then averaged, using Pearson correlation distance, cross-validated Mahalanobis distance (crossnobis), or pairwise decoding accuracy.
* **Commonality analysis:** The variance of the fMRI RDM explained jointly by the M/EEG RDM and a model RDM, computed layerwise for AlexNet.
* **VDNN vs LLM variance partitioning:** Commonality analysis after regressing out the other model's RDM, isolating the variance shared by M/EEG and fMRI that is uniquely explained by the VDNN or by the LLM.



## 🚀 Fusion tutorial

Through [this interactive Colab tutorial](https://colab.research.google.com/drive/1armXrKWxQnwzQQR7kD8rxRkT8MfM0SX9#scrollTo=ab41fa17) in Python, you will learn how to run the encoding-based and RSA-based fusion analyses on the THINGS MEG and fMRI data, restricted to three ROIs (V1, V4, IT) of one fMRI participant.



## ♻️ Reproducibility

This repository contains code to reproduce all of the paper results.



### ⚙️ Installation

To run the code, you first need to install the libraries in the [requirements.txt](<REPO_LINK>/blob/main/requirements.txt) file within an Anaconda environment. 


### 📦 Code description

The code is organized in one folder per dataset. Within each folder, scripts are numbered by analysis stage: data preparation first, then the encoding-based and RSA-based fusion analyses, each with separate compute and plotting scripts.

* **[`NSD`](<REPO_LINK>/tree/main/NSD):** Fusion analyses between NSD's fMRI responses and the NSD EEG responses.
* **[`THINGS`](<REPO_LINK>/tree/main/THINGS):** Fusion analyses between THINGS-MEG and THINGS-fMRI responses.
* **[`BMD_EMD`](<REPO_LINK>/tree/main/BMD_EMD):** Fusion analyses between EMD's EEG responses and BMD's fMRI responses.



## ❗ Issues

If you experience problems with the code, or for any other questions, please address your enquiries to jeffreywilliamw8@gmail.com..



## 📜 Citation

If you use any of our code, please cite:

> * <FUSION PAPER CITATION>

If you use the data, please also cite the corresponding dataset papers:

> * Allen EJ, St-Yves G, Wu Y, Breedlove JL, Prince JS, Dowdle LT, Nau M, Caron B, Pestilli F, Charest I, Hutchinson JB, Naselaris T, Kay K. 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. _Nature Neuroscience_, 25(1), 116-126. DOI: [https://doi.org/10.1038/s41593-021-00962-x](https://doi.org/10.1038/s41593-021-00962-x)
> * Hebart MN, Contier O, Teichmann L, Rockter AH, Zheng CY, Kidder A, Corriveau A, Vaziri-Pashkam M, Baker CI. 2023. THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. _eLife_, 12, e82580. DOI: [https://doi.org/10.7554/eLife.82580](https://doi.org/10.7554/eLife.82580)
> * Lahner B, Dwivedi K, Iamshchinina P, Graumann M, Lascelles A, Roig G, Gifford AT, Pan B, Jin S, Murty AR, Kay K, Oliva A, Cichy RM. 2024. Modeling short visual events through the BOLD moments video fMRI dataset and metadata. _Nature Communications_, 15(1), 6241. DOI: [https://doi.org/10.1038/s41467-024-50310-3](https://doi.org/10.1038/s41467-024-50310-3)
> * Gifford AT, Oyarzo P, Zonneveld AW, Sartzetaki C, Groen IIA, Cichy RM. 2026. A large dataset of human EEG responses to short naturalistic videos for studying dynamic visual event processing. _arXiv_. DOI: [https://doi.org/10.48550/arXiv.2608.28768](https://doi.org/10.48550/arXiv.2608.28768)
