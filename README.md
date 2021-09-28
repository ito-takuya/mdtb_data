# Multi-domain task battery data set (King et al. 2019, Nat. Neurosci)

#### This code repository contains preprocessing and basic code to load and access preprocessed code from the MDTB data set 
#### For usage on Yale's CRC Grace cluster (http://cluster.ycrc.yale.edu/grace/)
#### E-mail for questions: taku [dot] ito1 [at] gmail [dot] com
#### Last updated: 9/27/21


# Basic data directory structure:
#### Note that files will be read-only access, so it's recommended to copy template code to one's own directory on Grace and read-in data from your local project repo

Base directory: `/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/`

Raw BIDS formatted data: `/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/bids/`

QuNex preprocessed data: `/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/qunex_mdtb/sessions`

Fully preprocessed data: Includes QuNex preprocessed + post-processed nuisance regression (including removal of white matter, ventricle and motion time series). This includes processed data for both task and rest data at parcellated and vertex-wise time series: `/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/derivatives/postprocessing/`

Local code repo (which is synced to this GitHub repo): `/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/docs/`

# Demo code to load in processed resting-state and task-state fMRI
#### Note that processed task-state fMRI data has been processed using finite impulse response modeling (FIR) for task-state correlation/functional connectivity analyses, following procedures in Cole et al. (2019) NeuroImage http://www.sciencedirect.com/science/article/pii/S1053811918322043

Task conditions that are modeled in FIR are at the block-level, and so there are only 26 conditions for each task-specific time series.

Processed data is in parcellated time series (vertex-wise data to come) using the Glasser et al. (2016) parcellation scheme (http://www.nature.com/doifinder/10.1038/nature18933)

Demo code (Grace): `/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/docs/scripts/rest_task_fMRI_demo.ipynb`

Demo code (This repo): `scripts/rest_task_fMRI_demo.ipynb` (https://github.com/ito-takuya/mdtb_data/blob/main/scripts/rest_task_fMRI_demo.ipynb)

# Useful python dependencies (python version 3.8)
#### Helpful to start with anaconda python environment

Code requires nibabel, nipy, h5py

# For advanced users
#### For those who wish to preprocess and run GLMs themselves, the following code can be used as templates

#### Preprocessing from BIDS was implemented using QuNex. 

Qunex shell script: `scripts/preproc_qunex_turnkey_v2.sh` 

#### Postprocessing (i.e., nuisance regression for resting-state data and task GLMs) were implemented with custom python code. 

Resting-state nuisance regression: `scripts/glm_scripts/postproc_rest.py`

Task-state FIR regression (for FC and timescale analyses): `scripts/glm_scripts/postproc_taskFIR.py`

Task-state GLM activation estimation (uses a beta series type model, see Rissman et al. (2004), NeuroImage 10.1016/j.neuroimage.2004.06.035: `scripts/glm_scripts/postproc_taskbetaseries.py`

Generic post-processing tools: `scripts/glm_scripts/postproc_tools.py`

