# --- Some parameter definition to use in the script

# -> the name of the study & study folder

my_study=grega

# -> the paths to the training and study folder

training_folder=/gpfs/loomis/pi/n3/Studies/Training
my_study_folder=$training_folder/$my_study

# -> the paths to the raw data and prepared spec files

raw_data_folder=/gpfs/loomis/pi/n3/Studies/QuNexAcceptTest/RawData/QuNex_acceptance_MB_Yale_BIC_Prisma
spec_files_folder=/gpfs/loomis/pi/n3/software/qunexsdk/qunexaccept/SpecFiles/QuNex_acceptance_MB_Yale_BIC_Prisma

# -> the path to the container to be used

qunex_container=/gpfs/project/fas/n3/software/Singularity/qunex_suite_0.61.17.sif

# -> the expected path of where the batch.txt folder for the study will be generated

batch_file=$my_study_folder/processing/batch.txt


# --- create a study folder

qunexContainer createStudy "$training_folder/$my_study" \
    --container="$qunex_container" \
    --scheduler="SLURM,time=0-01:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=day" 


# --- import data

cp $raw_data_folder/*.zip $my_study_folder/sessions/inbox/MR

qunexContainer importDICOM \
    --sessionsfolder="$my_study_folder/sessions" \
    --container="$qunex_container" \
    --check=any \
    --options="addImageType:1" \
    --scheduler="SLURM,time=0-01:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=day" 

# -- note, we have added option "addImageType:1" so that different images generated from the 
#    same sequence have different name, which enables differating between them in the mapping file.

# --> use  this command to track the stdout of the running job

tail -f `ls -Art | tail -n 1`



# --- prepare spec files

cp $spec_files_folder/*.txt $my_study_folder/sessions/specs

qunexContainer createSessionInfo \
    --sessionsfolder="$my_study_folder/sessions" \
    --container="$qunex_container" \
    --scheduler="SLURM,time=0-01:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=day" 

qunexContainer createBatch \
    --sessionsfolder="$my_study_folder/sessions" \
    --container="$qunex_container" \
    --scheduler="SLURM,time=0-01:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=day" 


# --- prepare data

qunexContainer setupHCP \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --container="$qunex_container" \
    --scheduler="SLURM,time=0-01:00:00,ntasks=1,cpus-per-task=1,mem-per-cpu=8000,partition=day" 


# --- run HCP steps

qunexContainer hcp1 \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --container="$qunex_container" \
    --scheduler="SLURM,time=1-00:00:00,ntasks=1,cpus-per-task=2,mem-per-cpu=8000,partition=day"

qunexContainer hcp2 \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --container="$qunex_container" \
    --scheduler="SLURM,time=2-00:00:00,ntasks=1,cpus-per-task=2,mem-per-cpu=8000,partition=week"

qunexContainer hcp3 \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --container="$qunex_container" \
    --scheduler="SLURM,time=1-00:00:00,ntasks=1,cpus-per-task=2,mem-per-cpu=8000,partition=day"

qunexContainer hcp4 \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --container="$qunex_container" \
    --scheduler="SLURM,time=1-00:00:00,ntasks=1,cpus-per-task=2,mem-per-cpu=8000,partition=day"

qunexContainer hcp5 \
    --sessionsfolder="$my_study_folder/sessions" \
    --sessions="$batch_file" \
    --container="$qunex_container" \
    --scheduler="SLURM,time=1-00:00:00,ntasks=1,cpus-per-task=2,mem-per-cpu=8000,partition=day"