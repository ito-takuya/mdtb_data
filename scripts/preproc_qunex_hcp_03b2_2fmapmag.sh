## FSL Setup
#FSLDIR=/gpfs/loomis/project/fas/n3/software/fsl/fsl/
#PATH=${FSLDIR}/bin:${PATH}
#export FSLDIR PATH
#. ${FSLDIR}/etc/fslconf/fsl.sh

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

qunex_container=/gpfs/project/fas/n3/software/Singularity/qunex_suite_0.61.17.sif
#qunex_container=/gpfs/project/fas/n3/software/Singularity/qunex_suite-latest.sif

# -> the expected path of where the batch.txt folder for the study will be generated

batch_file=$my_study_folder/processing/batch_03b2_hcp_2fmapmag.txt

# --- create a study folder
execute=0
if [ $execute -eq 1 ]; then

qunexContainer createStudy "$my_study_folder" \
    --container="$qunex_container" 
    #--scheduler="SLURM,time=0-01:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=day" 


fi
# --- import data

#cp $raw_data_folder/*.zip $my_study_folder/sessions/inbox/MR

execute=0
if [ $execute -eq 1 ]; then


qunexContainer importBIDS \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="03_b2" \
    --inbox="$raw_data_folder" \
    --action="copy" \
    --archive="leave" \
    --fileinfo="full" \
    --container="$qunex_container" 

#qunex importDICOM \
#    --sessionsfolder="$my_study_folder/sessions" \
#    --container="$qunex_container" \
#    --check=any \
#    --options="addImageType:1"
#    --scheduler="SLURM,time=0-01:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=day" 

# -- note, we have added option "addImageType:1" so that different images generated from the 
#    same sequence have different name, which enables differating between them in the mapping file.

# --> use  this command to track the stdout of the running job

#tail -f `ls -Art | tail -n 1`

fi



# --- prepare spec files
#cp $spec_files_folder/*.txt $my_study_folder/sessions/specs

execute=0
if [ $execute -eq 1 ]; then

#cp $spec_files_folder/*.txt $my_study_folder/sessions/specs

qunexContainer createSessionInfo \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="03_b2" \
    --mapping="$my_study_folder/sessions/specs/mdtb2hcp_mapping_2fmapmag.txt" \
    --overwrite="yes" \
    --container="$qunex_container"
#    --scheduler="SLURM,time=0-01:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=day" 

fi

execute=0
if [ $execute -eq 1 ]; then

execute=0
qunexContainer createBatch \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="03_b2" \
    --sourcefiles="session_hcp.txt" \
    --paramfile="$my_study_folder/sessions/specs/batch_hcp2.txt" \
    --targetfile="$batch_file" \
    --overwrite="yes" \
    --container="$qunex_container"

    #--sessions="[0-9][0-9]_*" \
    #--targetfile="$my_study_folder/processing/batch.txt" \

fi


# --- prepare data
execute=0
if [ $execute -eq 1 ]; then

qunexContainer setupHCP \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --existing="clear" \
    --folderstructure="hcpls" \
    --container="$qunex_container"
    #--filename="standard" \

#qunexContainer setupHCP \
#    --filename="original" \
#    --container="$qunex_container" 


fi


#### Remap FieldMaps (CUSTOM)
execute=0
if [ $execute -eq 1 ]; then
sessions="03_b2"
for session in $sessions
do
    bidsdir=$my_study_folder/sessions/$session/bids/fmap/
    outputdir=$my_study_folder/sessions/$session/hcp/$session/unprocessed/FieldMap
    mv ${outputdir}1 ${outputdir}
    chmod 777 $outputdir/${session}_FieldMap_Magnitude.nii.gz
    fslmerge -a $outputdir/${session}_FieldMap_Magnitude.nii.gz $bidsdir/*_magnitude2.nii.gz $bidsdir/*_magnitude1.nii.gz
done
fi

# --- run HCP steps

execute=0
if [ $execute -eq 1 ]; then

qunexContainer hcp1 \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --overwrite="yes" \
    --container="$qunex_container"
    #--parsessions="4" \
#    --scheduler="SLURM,time=1-00:00:00,ntasks=1,cpus-per-task=2,mem-per-cpu=8000,partition=day"

fi

execute=0
if [ $execute -eq 1 ]; then

qunexContainer hcp2 \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --overwrite="no" \
    --container="$qunex_container" 
    #--scheduler="SLURM,time=2-00:00:00,ntasks=1,cpus-per-task=2,mem-per-cpu=8000,partition=week"

fi

execute=0
if [ $execute -eq 1 ]; then

qunexContainer hcp3 \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --overwrite="no" \
    --container="$qunex_container" 
    #--scheduler="SLURM,time=2-00:00:00,ntasks=1,cpus-per-task=2,mem-per-cpu=8000,partition=week"
fi

execute=0
if [ $execute -eq 1 ]; then

qunexContainer hcp4 \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --overwrite="no" \
    --container="$qunex_container" 
    #--scheduler="SLURM,time=2-00:00:00,ntasks=1,cpus-per-task=2,mem-per-cpu=8000,partition=week"
fi

execute=0
if [ $execute -eq 1 ]; then

qunexContainer hcp5 \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --overwrite="yes" \
    --container="$qunex_container" 
    #--scheduler="SLURM,time=2-00:00:00,ntasks=1,cpus-per-task=2,mem-per-cpu=8000,partition=week"

fi

execute=0
if [ $execute -eq 1 ]; then

qunexContainer runQC \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --modality='T1w' \
    --overwrite='no' \
    --container="$qunex_container" 

fi

execute=1
if [ $execute -eq 1 ]; then

qunexContainer runQC \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --modality='BOLD' \
    --bolddata='1' \
    --boldsuffix='Atlas' \
    --overwrite='no' \
    --container="$qunex_container" 
fi
