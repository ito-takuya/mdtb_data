/gpfs/loomis/project/fas/n3/software/HCP/HCPpipelines/PreFreeSurfer/PreFreeSurferPipeline.sh \
    --path="/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/03_a1/hcp" \
    --subject="03_a1" \
    --t1="/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/03_a1/hcp/03_a1/unprocessed/T1w/03_a1_T1w_MPR1.nii.gz" \
    --t2=None \
    --t1template="/opt/HCP/HCPpipelines/global/templates/MNI152_T1_0.7mm.nii.gz" \
    --t1templatebrain="/opt/HCP/HCPpipelines/global/templates/MNI152_T1_0.7mm_brain.nii.gz" \
    --t1template2mm="/opt/HCP/HCPpipelines/global/templates/MNI152_T1_2mm.nii.gz" \
    --t2template="/opt/HCP/HCPpipelines/global/templates/MNI152_T2_0.7mm.nii.gz" \
    --t2templatebrain="/opt/HCP/HCPpipelines/global/templates/MNI152_T2_0.7mm_brain.nii.gz" \
    --t2template2mm="/opt/HCP/HCPpipelines/global/templates/MNI152_T2_2mm.nii.gz" \
    --templatemask="/opt/HCP/HCPpipelines/global/templates/MNI152_T1_0.7mm_brain_mask.nii.gz" \
    --template2mmmask="/opt/HCP/HCPpipelines/global/templates/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
    --brainsize="150" \
    --fnirtconfig="/opt/HCP/HCPpipelines/global/config/T1_2_MNI152_2mm.cnf" \
    --fmapmag="/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/03_a1/hcp/03_a1/unprocessed/FieldMap/03_a1_FieldMap_Magnitude.nii.gz" \
    --fmapphase="/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/03_a1/hcp/03_a1/unprocessed/FieldMap/03_a1_FieldMap_Phase.nii.gz" \
    --echodiff="2.46" \
    --seechospacing="NONE" \
    --seunwarpdir="NONE" \
    --t1samplespacing="0.0000082" \
    --t2samplespacing="0.0000089" \
    --unwarpdir="z" \
    --gdcoeffs="NONE" \
    --avgrdcmethod="SiemensFieldMap" \
    --processing-mode="LegacyStyleData"
#/gpfs/loomis/project/fas/n3/software/HCP/HCPpipelines/PreFreeSurfer/PreFreeSurferPipeline.sh \
#    --path="/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/03_a1/hcp" \
#    --subject="03_a1" \
#    --t1="/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/03_a1/hcp/03_a1/unprocessed/T1w/03_a1_T1w_MPR1.nii.gz" \
#    --t2="/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/03_a1/hcp/03_a1/unprocessed/T2w/03_a1_T2w_SPC1.nii.gz" \
#    --t1template="/opt/HCP/HCPpipelines/global/templates/MNI152_T1_0.7mm.nii.gz" \
#    --t1templatebrain="/opt/HCP/HCPpipelines/global/templates/MNI152_T1_0.7mm_brain.nii.gz" \
#    --t1template2mm="/opt/HCP/HCPpipelines/global/templates/MNI152_T1_2mm.nii.gz" \
#    --t2template="/opt/HCP/HCPpipelines/global/templates/MNI152_T2_0.7mm.nii.gz" \
#    --t2templatebrain="/opt/HCP/HCPpipelines/global/templates/MNI152_T2_0.7mm_brain.nii.gz" \
#    --t2template2mm="/opt/HCP/HCPpipelines/global/templates/MNI152_T2_2mm.nii.gz" \
#    --templatemask="/opt/HCP/HCPpipelines/global/templates/MNI152_T1_0.7mm_brain_mask.nii.gz" \
#    --template2mmmask="/opt/HCP/HCPpipelines/global/templates/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
#    --brainsize="150" \
#    --fnirtconfig="/opt/HCP/HCPpipelines/global/config/T1_2_MNI152_2mm.cnf" \
#    --fmapmag="/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/03_a1/hcp/03_a1/unprocessed/FieldMap/03_a1_FieldMap_Magnitude.nii.gz" \
#    --fmapphase="/gpfs/loomis/pi/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/03_a1/hcp/03_a1/unprocessed/FieldMap/03_a1_FieldMap_Phase.nii.gz" \
#    --echodiff="2.46" \
#    --seechospacing="NONE" \
#    --seunwarpdir="NONE" \
#    --t1samplespacing="0.0000082" \
#    --t2samplespacing="0.0000089" \
#    --unwarpdir="z" \
#    --gdcoeffs="NONE" \
#    --avgrdcmethod="SiemensFieldMap" \
#    --processing-mode="LegacyStyleData"
