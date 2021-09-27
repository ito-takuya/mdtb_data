# Multi-domain task battery data set (King et al. 2019, Nat. Neurosci)

#### This code repository contains preprocessing and basic code to load and access preprocessed code from the MDTB data set 
#### For usage on Yale's CRC Grace cluster (http://cluster.ycrc.yale.edu/grace/)
#### E-mail for questions: taku [dot] ito1 [at] gmail [dot] com
#### Last updated: 9/27/21

___

# Basic data directory structure:
#### Note that files will be read-only access, so it's recommended to copy template code to one's own directory on Grace and read-in data from your local project repo

Base directory: `/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/`
Raw BIDS formatted data: `/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/bids/`
QuNex preprocessed data: `/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/qunex_mdtb/sessions`
Fully preprocessed data: Includes QuNex preprocessed + post-processed nuisance regression (including removal of white matter, ventricle and motion time series). This includes processed data for both task and rest data at parcellated and vertex-wise time series.
`/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/derivatives/postprocessing/`
Local code repo (which is synced to this GitHub repo): `/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/mdtb_data/docs/`

