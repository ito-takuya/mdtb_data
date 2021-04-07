# Taku Ito
# 10/09/2018

# Script requires workbench (wb_command), in addition to the below python packages
# Load dependencies
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

#Setting the parcel files to be the 360 Glasser2016 cortical parcels
cabnp_dir='/home/ti236/AnalysisTools/ColeAnticevicNetPartition/'
L_parcelCIFTIFile=cabnp_dir + 'SeparateHemispheres/Q1-Q6_RelatedValidation210.L.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'
R_parcelCIFTIFile=cabnp_dir + 'SeparateHemispheres/Q1-Q6_RelatedValidation210.R.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'
glasser=cabnp_dir + '../Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
#glasser = cabnp_dir + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'


sessions=['02_a1','02_a2','02_b1','02_b2','03_a1','03_a2','03_b1','03_b2','04_a1','04_a2','04_b1','04_b2','06_a1','06_a2','06_b1','06_b2','09_a1','09_a2','09_b1','09_b2','12_a1','12_a2','12_b1','12_b2','15_a1','15_a2','15_b1','15_b2','18_a1','18_a2','18_b1','18_b2','20_a1','20_a2','20_b1','20_b2','22_a1','22_a2','22_b1','22_b2','25_a1','25_a2','25_b1','25_b2','27_a1','27_a2','27_b1','27_b2','29_a1','29_a2','29_b1','29_b2','31_a1','31_a2','31_b1','31_b2','08_a1','08_a2','08_b1','08_b2','10_a1','10_a2','10_b1','10_b2','14_a1','14_a2','14_b1','14_b2','17_a1','17_a2','17_b1','17_b2','19_a1','19_a2','19_b1','19_b2','21_a1','21_a2','21_b1','21_b2','24_a1','24_a2','24_b1','24_b2','26_a1','26_a2','26_b1','26_b2','28_a1','28_a2','28_b1','28_b2','30_a1','30_a2','30_b1','30_b2']

runnames=['bold1','bold2','bold3','bold4','bold5','bold6','bold7','bold8','bold9','bold10']

datadir = '/gpfs/loomis/project/n3/Studies/MurrayLab/taku/mdtb_data/qunex_mdtb/'

for sess in sessions:
    for run in runnames:
        # Skip this session if it's session _b2 and bold9 or bold10, since these are rest runs exclusive to that session
        if sess[-2:]!='b2':
            if run=='bold9' or run=='bold10':
                continue 

        print('Parcellating session:', sess, '| Run :', run)
        # Set session directory
        sessiondir=datadir + 'sessions/' + sess + '/images/functional/'

        #Set this to be your input fMRI data CIFTI file
        inputFile=sessiondir + run + '_Atlas.dtseries.nii'

        # Specify output files 
        #L_parcelTSFilename=sessiondir + run + '_Atlas.L.Parcels.32k_fs_LR.ptseries.nii'
        #R_parcelTSFilename=sessiondir + run + '_Atlas.R.Parcels.32k_fs_LR.ptseries.nii'
        all_parcelTSFilename=sessiondir + run + '_Atlas.LR.Parcels.32k_fs_LR.ptseries.nii'

        ## This approach seems off - RH indices and LH indices appear to be flipped (LH indices [181-360], RH indices are [1-380]
        # Parcellate dense time series using wb_command for left and right hemispheres
        #os.system('wb_command -cifti-parcellate ' + inputFile + ' ' + L_parcelCIFTIFile + ' COLUMN ' + L_parcelTSFilename + ' -method MEAN')
        #os.system('wb_command -cifti-parcellate ' + inputFile + ' ' + R_parcelCIFTIFile + ' COLUMN ' + R_parcelTSFilename + ' -method MEAN')
        #

        ### This is OKAY - Glasser file is R->L, but structure information is preserved within cifti, and L regions are still indexed 1-180, while R regions are indexed 181-360
        os.system('wb_command -cifti-parcellate ' + inputFile + ' ' + glasser + ' COLUMN ' + all_parcelTSFilename + ' -method MEAN')

        #try:
        #    dtseries = np.squeeze(nib.load(inputFile).get_data())
        #    mat = []
        #    for parcel in range(360):
        #        roi_ind = np.where(glasser==parcel+1)[0]
        #        mat.append(np.nanmean(dtseries[:,roi_ind],axis=1))

        #    mat = np.asarray(mat)
        #    np.savetxt(sessiondir + run + '_parcellated.LR.csv',mat)
        #except:
        #    print('Skipping session--- not found')





