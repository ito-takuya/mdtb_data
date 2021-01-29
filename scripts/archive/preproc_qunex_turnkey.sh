#!/bin/bash
# FSL Setup
FSLDIR=/gpfs/loomis/project/fas/n3/software/fsl/fsl/
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
. ${FSLDIR}/etc/fslconf/fsl.sh

# --- Some parameter definition to use in the script

# -> the name of the study & study folder

my_study=qunexMultiTaskVAE

# -> the paths to the training and study folder

#training_folder=/gpfs/loomis/pi/n3/Studies/Training
training_folder=/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE
my_study_folder=$training_folder/$my_study

# -> the paths to the raw data and prepared spec files

#raw_data_folder=/gpfs/loomis/pi/n3/Studies/QuNexAcceptTest/RawData/QuNex_acceptance_MB_Yale_BIC_Prisma
raw_data_folder=/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/bids
spec_files_folder=/gpfs/loomis/pi/n3/software/qunexsdk/qunexaccept/SpecFiles/QuNex_acceptance_MB_Yale_BIC_Prisma

# -> the path to the container to be used

#qunex_container=/gpfs/project/fas/n3/software/Singularity/qunex_suite_0.61.17.sif
qunex_container=/gpfs/project/fas/n3/software/Singularity/qunex_suite-latest.sif

# -> the expected path of where the batch.txt folder for the study will be generated

batch_file_template=$my_study_folder/sessions/specs/batch_legacy.txt
batch_file=$my_study_folder/processing/batch_turnkey_2fmapmag.txt
sessions="04_b1"
# - Type of QuNex Turnkey run
RUNTURNKEY_TYPE="local"

# --- create a study folder
execute=0
if [ $execute -eq 1 ]; then

qunexContainer createStudy "$my_study_folder" \
    --container="$qunex_container" 
    #--scheduler="SLURM,time=0-01:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=day" 
fi


execute=0
if [ $execute -eq 1 ]; then

qunexContainer importBIDS \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$sessions" \
    --container="$qunex_container" \
    --inbox="$raw_data_folder" \
    --action="copy" \
    --archive="leave" \
    --fileinfo="full"

tail -f `ls -Art | tail -n 1`

fi


# - QuNex Turnkey steps
runturnkey_setup="createSessionInfo,createBatch,setupHCP"
execute=0
if [ $execute -eq 1 ]; then

qunexContainer runTurnkey \
    --batchfile="$batch_file_template" \
    --turnkeytype="${RUNTURNKEY_TYPE}" \
    --path="$my_study_folder" \
    --workingdir="$training_folder" \
    --sessions="04_b1" \
    --sessionsfoldername="sessions" \
    --mappingfile="$my_study_folder/sessions/specs/mdtb2hcp_mapping_2fmapmag.txt" \
    --projectname="$my_study" \
    --turnkeysteps="$runturnkey_setup" \
    --overwritestep="no" \
    --container="$qunex_container"

fi


#### Remap FieldMaps (CUSTOM)
execute=0
if [ $execute -eq 1 ]; then
for session in $sessions
do
    bidsdir=$my_study_folder/sessions/$session/bids/fmap/
    outputdir=$my_study_folder/sessions/$session/hcp/$session/unprocessed/FieldMap
    mv -v ${outputdir}1 ${outputdir}
    chmod 777 $outputdir/${session}_FieldMap_Magnitude.nii.gz
    fslmerge -a $outputdir/${session}_FieldMap_Magnitude.nii.gz $bidsdir/*_magnitude1.nii.gz $bidsdir/*_magnitude2.nii.gz
done
fi

# --- run HCP steps turnkey
# - QuNex Turnkey steps
runturnkey_hcp="hcp1,hcp2,hcp3,hcp4,hcp5,runQC_T1w,runQC_T2w,runQC_Myelin,runQC_BOLD,mapHCPData,createBOLDBrainMasks,createBOLDStats"
execute=1
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
    --turnkeysteps="$runturnkey_hcp" \
    --overwritestep="no" \
    --container="$qunex_container"

fi

## Extra custom code -- need to resample aparc+aseg to appropriate create masks and extract nuisance signals
execute=0
if [ $execute -eq 1 ]; then

    for session in $sessions
    do
        3dresample -input $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold.nii.gz -prefix $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold_3mm.nii.gz -master $my_study_folder/sessions/$session/images/segmentation/boldmasks/bold8_frame1_brain_mask.nii.gz
        mv -v $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold.nii.gz $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold_orig.nii.gz
        mv -v $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold_3mm.nii.gz $my_study_folder/sessions/$session/images/segmentation/freesurfer/mri/aparc+aseg_bold.nii.gz
    done

fi

execute=0
if [ $execute -eq 1 ]; then

qunexContainer extractNuisanceSignal \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --overwrite='yes' \
    --container="$qunex_container" 
fi


