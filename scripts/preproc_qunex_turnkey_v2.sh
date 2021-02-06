#!/bin/bash
# Taku Ito
# Preprocess MDTB data set with Qunex (Legacy variant -- exclude T2 since it's low-res)

# FSL Setup
FSLDIR=/gpfs/loomis/project/fas/n3/software/fsl/fsl/
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
. ${FSLDIR}/etc/fslconf/fsl.sh

## STUDY PARAMETERS
my_study=qunexMultiTaskVAE
training_folder=/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE
my_study_folder=$training_folder/$my_study
bids_data_folder=/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/bids

qunex_container=/gpfs/project/fas/n3/software/Singularity/qunex_suite_0.61.17.sif
#qunex_container=/gpfs/project/fas/n3/software/Singularity/qunex_suite-latest.sif

# batch files
batch_file_template=$my_study_folder/sessions/specs/batch_mdtb_legacy.txt
batch_directory=$my_study_folder/processing 

# session IDs
#sessions="sub-02/ses-a1,sub-02/ses-a2,sub-02/ses-b1,sub-02/ses-b2"
subjects="sub-02 sub-03 sub-04 sub-06 sub-09 sub-12 sub-15 sub-18 sub-20 sub-22 sub-25 sub-27 sub-29 sub-31 sub-02 sub-04 sub-08 sub-10 sub-14 sub-17 sub-19 sub-21 sub-24 sub-26 sub-28 sub-30"
session_suffixes="a1 a2 b1 b2"
sessions="02_a1,02_a2,02_b1,02_b2,03_a1,03_a2,03_b1,03_b2,04_a1,04_a2,04_b1,04_b2,06_a1,06_a2,06_b1,06_b2,09_a1,09_a2,09_b1,09_b2,12_a1,12_a2,12_b1,12_b2,15_a1,15_a2,15_b1,15_b2,18_a1,18_a2,18_b1,18_b2,20_a1,20_a2,20_b1,20_b2,22_a1,22_a2,22_b1,22_b2,25_a1,25_a2,25_b1,25_b2,27_a1,27_a2,27_b1,27_b2,29_a1,29_a2,29_b1,29_b2,31_a1,31_a2,31_b1,31_b2,02_a1,02_a2,02_b1,02_b2,04_a1,04_a2,04_b1,04_b2,08_a1,08_a2,08_b1,08_b2,10_a1,10_a2,10_b1,10_b2,14_a1,14_a2,14_b1,14_b2,17_a1,17_a2,17_b1,17_b2,19_a1,19_a2,19_b1,19_b2,21_a1,21_a2,21_b1,21_b2,24_a1,24_a2,24_b1,24_b2,26_a1,26_a2,26_b1,26_b2,28_a1,28_a2,28_b1,28_b2,30_a1,30_a2,30_b1,30_b2"
sessions_arr="02_a1 02_a2 02_b1 02_b2 03_a1 03_a2 03_b1 03_b2 04_a1 04_a2 04_b1 04_b2 06_a1 06_a2 06_b1 06_b2 09_a1 09_a2 09_b1 09_b2 12_a1 12_a2 12_b1 12_b2 15_a1 15_a2 15_b1 15_b2 18_a1 18_a2 18_b1 18_b2 20_a1 20_a2 20_b1 20_b2 22_a1 22_a2 22_b1 22_b2 25_a1 25_a2 25_b1 25_b2 27_a1 27_a2 27_b1 27_b2 29_a1 29_a2 29_b1 29_b2 31_a1 31_a2 31_b1 31_b2 02_a1 02_a2 02_b1 02_b2 04_a1 04_a2 04_b1 04_b2 08_a1 08_a2 08_b1 08_b2 10_a1 10_a2 10_b1 10_b2 14_a1 14_a2 14_b1 14_b2 17_a1 17_a2 17_b1 17_b2 19_a1 19_a2 19_b1 19_b2 21_a1 21_a2 21_b1 21_b2 24_a1 24_a2 24_b1 24_b2 26_a1 26_a2 26_b1 26_b2 28_a1 28_a2 28_b1 28_b2 30_a1 30_a2 30_b1 30_b2"
sessions="08_a1,17_a2"
sessions_arr="08_a1 17_a2"

# Session names, excluding 08_a1 and 17_a2
sessions="02_a1,02_a2,02_b1,02_b2,03_a1,03_a2,03_b1,03_b2,04_a1,04_a2,04_b1,04_b2,06_a1,06_a2,06_b1,06_b2,09_a1,09_a2,09_b1,09_b2,12_a1,12_a2,12_b1,12_b2,15_a1,15_a2,15_b1,15_b2,18_a1,18_a2,18_b1,18_b2,20_a1,20_a2,20_b1,20_b2,22_a1,22_a2,22_b1,22_b2,25_a1,25_a2,25_b1,25_b2,27_a1,27_a2,27_b1,27_b2,29_a1,29_a2,29_b1,29_b2,31_a1,31_a2,31_b1,31_b2,02_a1,02_a2,02_b1,02_b2,04_a1,04_a2,04_b1,04_b2,08_a2,08_b1,08_b2,10_a1,10_a2,10_b1,10_b2,14_a1,14_a2,14_b1,14_b2,17_a1,17_b1,17_b2,19_a1,19_a2,19_b1,19_b2,21_a1,21_a2,21_b1,21_b2,24_a1,24_a2,24_b1,24_b2,26_a1,26_a2,26_b1,26_b2,28_a1,28_a2,28_b1,28_b2,30_a1,30_a2,30_b1,30_b2"
sessions_arr="02_a1 02_a2 02_b1 02_b2 03_a1 03_a2 03_b1 03_b2 04_a1 04_a2 04_b1 04_b2 06_a1 06_a2 06_b1 06_b2 09_a1 09_a2 09_b1 09_b2 12_a1 12_a2 12_b1 12_b2 15_a1 15_a2 15_b1 15_b2 18_a1 18_a2 18_b1 18_b2 20_a1 20_a2 20_b1 20_b2 22_a1 22_a2 22_b1 22_b2 25_a1 25_a2 25_b1 25_b2 27_a1 27_a2 27_b1 27_b2 29_a1 29_a2 29_b1 29_b2 31_a1 31_a2 31_b1 31_b2 02_a1 02_a2 02_b1 02_b2 04_a1 04_a2 04_b1 04_b2 08_a2 08_b1 08_b2 10_a1 10_a2 10_b1 10_b2 14_a1 14_a2 14_b1 14_b2 17_a1 17_b1 17_b2 19_a1 19_a2 19_b1 19_b2 21_a1 21_a2 21_b1 21_b2 24_a1 24_a2 24_b1 24_b2 26_a1 26_a2 26_b1 26_b2 28_a1 28_a2 28_b1 28_b2 30_a1 30_a2 30_b1 30_b2"


# - Type of QuNex Turnkey run
RUNTURNKEY_TYPE="local"


## BEGIN CODE BLOCKS

# Create a study folder
execute=0
if [ $execute -eq 1 ]; then

qunexContainer createStudy "$my_study_folder" \
    --container="$qunex_container" 
    #--scheduler="SLURM,time=0-01:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=day" 
fi

# Copy rest functional data into session b2 -- rest data doesn't have its own anat/fmaps so we'll process it with data from session b2
execute=0
if [ $execute -eq 1 ]; then

    for subj in $subjects
    do
        # Copy the NIFTIS
        cp -v $bids_data_folder/$subj/ses-rest/func/${subj}_ses-rest_task-rest_run-1_bold.nii.gz $bids_data_folder/$subj/ses-b2/func/${subj}_ses-b2_task-rest_run-1_bold.nii.gz 
        cp -v $bids_data_folder/$subj/ses-rest/func/${subj}_ses-rest_task-rest_run-2_bold.nii.gz $bids_data_folder/$subj/ses-b2/func/${subj}_ses-b2_task-rest_run-2_bold.nii.gz 
        # Copy the JSONs
        cp -v $bids_data_folder/$subj/ses-rest/func/${subj}_ses-rest_task-rest_run-1_bold.json $bids_data_folder/$subj/ses-b2/func/${subj}_ses-b2_task-rest_run-1_bold.json
        cp -v $bids_data_folder/$subj/ses-rest/func/${subj}_ses-rest_task-rest_run-2_bold.json $bids_data_folder/$subj/ses-b2/func/${subj}_ses-b2_task-rest_run-2_bold.json 
    done

fi

### Import BIDS
execute=0
if [ $execute -eq 1 ]; then

qunexContainer importBIDS \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$sessions" \
    --container="$qunex_container" \
    --inbox="$bids_data_folder" \
    --action="copy" \
    --archive="leave" \
    --fileinfo="full"

tail -f `ls -Art | tail -n 1`

fi


# - QuNex Turnkey steps -- HCP SET UP
runturnkey_setup="createSessionInfo,createBatch,setupHCP"
execute=0
if [ $execute -eq 1 ]; then

qunexContainer runTurnkey \
    --batchfile="$batch_file_template" \
    --turnkeytype="${RUNTURNKEY_TYPE}" \
    --path="$my_study_folder" \
    --workingdir="$training_folder" \
    --sessions="$sessions" \
    --sessionsfoldername="sessions" \
    --mappingfile="$my_study_folder/sessions/specs/mdtb2hcp_mapping_2fmapmag.txt" \
    --projectname="$my_study" \
    --turnkeysteps="$runturnkey_setup" \
    --overwritestep="no" \
    --acceptancetest="no" \
    --container="$qunex_container"

fi


#### Remap FieldMaps (CUSTOM)
execute=0
if [ $execute -eq 1 ]; then
for session in $sessions_arr
do
    bidsdir=$my_study_folder/sessions/$session/bids/fmap/
    outputdir=$my_study_folder/sessions/$session/hcp/$session/unprocessed/FieldMap
    mv -v ${outputdir}1 ${outputdir}
    chmod 777 $outputdir/${session}_FieldMap_Magnitude.nii.gz
    fslmerge -a $outputdir/${session}_FieldMap_Magnitude.nii.gz $bidsdir/*_magnitude1.nii.gz $bidsdir/*_magnitude2.nii.gz
done
fi

#### One time edit: two sessions don't have a fieldmap. This is a hack -- just copy over a fieldmap from a previous session (same subject)
# sessions without a fieldmap: 29_b2, 08_b1
execute=0
if [ $execute -eq 1 ]; then
    outputdir=$my_study_folder/sessions/$session/hcp/$session/unprocessed/FieldMap
    # copy 29_b1 to 29_b2
    cp -v -r $my_study_folder/sessions/29_b1/hcp/29_b1/unprocessed/FieldMap $my_study_folder/sessions/29_b2/hcp/29_b2/unprocessed/
    # Artificially rename to correct session name
    mv -v $my_study_folder/sessions/29_b2/hcp/29_b2/unprocessed/FieldMap/29_b1_FieldMap_Magnitude.nii.gz $my_study_folder/sessions/29_b2/hcp/29_b2/unprocessed/FieldMap/29_b2_FieldMap_Magnitude.nii.gz
    mv -v $my_study_folder/sessions/29_b2/hcp/29_b2/unprocessed/FieldMap/29_b1_FieldMap_Phase.nii.gz $my_study_folder/sessions/29_b2/hcp/29_b2/unprocessed/FieldMap/29_b2_FieldMap_Phase.nii.gz
    # copy 08_b2 to 08_b1
    cp -v -r $my_study_folder/sessions/08_b2/hcp/08_b2/unprocessed/FieldMap $my_study_folder/sessions/08_b1/hcp/08_b1/unprocessed/
    # Artificially rename to correct session name
    mv -v $my_study_folder/sessions/08_b1/hcp/08_b1/unprocessed/FieldMap/08_b2_FieldMap_Magnitude.nii.gz $my_study_folder/sessions/08_b1/hcp/08_b1/unprocessed/FieldMap/08_b1_FieldMap_Magnitude.nii.gz
    mv -v $my_study_folder/sessions/08_b1/hcp/08_b1/unprocessed/FieldMap/08_b2_FieldMap_Phase.nii.gz $my_study_folder/sessions/08_b1/hcp/08_b1/unprocessed/FieldMap/08_b1_FieldMap_Phase.nii.gz

fi


#### Now, create a batch file for every session separately
execute=0
if [ $execute -eq 1 ]; then

for sess in $sessions_arr
do
    qunexContainer createBatch \
        --sessionsfolder="$my_study_folder/sessions" \
        --sessions="$sess" \
        --sourcefiles="session_hcp.txt" \
        --paramfile="$my_study_folder/sessions/specs/batch_mdtb_legacy.txt" \
        --targetfile="$batch_directory/${sess}_batch.txt" \
        --overwrite="yes" \
        --container="$qunex_container" 
done

fi

# MOST IMPORTANT STEPS - RUN PREPROCESSING VIA TURNKEY on scheduler
# - QuNex Turnkey steps
runturnkey_hcp="hcp1,hcp2,hcp3,hcp4,hcp5"
execute=0
if [ $execute -eq 1 ]; then

for sess in $sessions_arr
do
    qunexContainer runTurnkey \
        --batchfile="$batch_directory/${sess}_batch.txt" \
        --local_batchfile="$batch_directory/${sess}_batch.txt" \
        --turnkeytype="${RUNTURNKEY_TYPE}" \
        --path="$my_study_folder" \
        --workingdir="$training_folder" \
        --sessions="$sess" \
        --sessionsid="$sess" \
        --sessionsfoldername="sessions" \
        --mappingfile="$my_study_folder/sessions/specs/mdtb2hcp_mapping_2fmapmag.txt" \
        --projectname="$my_study" \
        --turnkeysteps="$runturnkey_hcp" \
        --overwritestep="no" \
        --container="$qunex_container" \
        --scheduler="SLURM,time=3-00:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=pi_anticevic,partition=week,account=anticevic" 

done
fi

# run QC turnkey 
# - QuNex Turnkey steps
#runturnkey_hcp="runQC_T1w,runQC_T2w,runQC_Myelin,runQC_BOLD,mapHCPData,createBOLDBrainMasks,createBOLDStats"
runturnkey_hcp="runQC_T1w,runQC_BOLD,mapHCPData,createBOLDBrainMasks,createBOLDStats"
#runturnkey_hcp="runQC_BOLD"
execute=0
if [ $execute -eq 1 ]; then

for sess in $sessions_arr
do
    qunexContainer runTurnkey \
        --batchfile="$batch_directory/${sess}_batch.txt" \
        --local_batchfile="$batch_directory/${sess}_batch.txt" \
        --turnkeytype="${RUNTURNKEY_TYPE}" \
        --path="$my_study_folder" \
        --workingdir="$training_folder" \
        --sessions="$sess" \
        --sessionsid="$sess" \
        --sessionsfoldername="sessions" \
        --mappingfile="$my_study_folder/sessions/specs/mdtb2hcp_mapping_2fmapmag.txt" \
        --projectname="$my_study" \
        --turnkeysteps="$runturnkey_hcp" \
        --overwritestep="no" \
        --container="$qunex_container" \
        --scheduler="SLURM,time=1-00:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=pi_anticevic,partition=day,account=anticevic" 

done
fi


## Extra custom code -- need to resample aparc+aseg to appropriate create masks and extract nuisance signals
execute=0
if [ $execute -eq 1 ]; then

    for session in $sessions_arr
    do
        echo "Resampling freesurfer output for session $session"
        3dresample -input $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold.nii.gz -prefix $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold_3mm.nii.gz -master $my_study_folder/sessions/$session/images/segmentation/boldmasks/bold8_frame1_brain_mask.nii.gz
        mv -v $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold.nii.gz $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold_orig.nii.gz
        mv -v $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold_3mm.nii.gz $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold.nii.gz
        echo "DONE for session $session"
    done

fi

execute=1
if [ $execute -eq 1 ]; then

    for session in $sessions_arr
    do  
        qunexContainer extractNuisanceSignal \
            --sessionsfolder="$my_study_folder/sessions" \
            --sessions="${batch_directory}/${session}_batch.txt" \
            --overwrite='yes' \
            --container="$qunex_container" \
            --scheduler="SLURM,time=1-00:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=pi_anticevic,partition=day,account=anticevic" 
    done
fi


