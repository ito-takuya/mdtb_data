# Takuya Ito
# 2/5/20201

# Post-processing nuisance regression using Ciric et al. 2017 inspired best-practices
## OVERVIEW
# There are two main parts to this script/set of functions
# 1. "step1_createNuisanceRegressors" 
#   Generates a variety of nuisance regressors, such as motionSpikes, aCompCor regressors, etc. that are essential to a subset of Ciric-style models, with the addition of some new combinations (e.g., aCompCor + spikeReg + movement parameters) 
#   This is actually the bulk of the script, and takes quite a while to compute, largely due to the fact that we need to load in 4D time series from the raw fMRI data (in order to compute regressors such as global signal)
# 2. "step2_nuisanceRegression"
#   This is the function that actually performs the nuisance regression, using regressors obtained from step1. There are a variety of models to choose from, including:
#       The best model from Ciric et al. (2017) (e.g., 36p + spikeReg)
#       What I call the "legacy Cole Lab models", which are the traditional 6 motion parameters, gsr, wm and ventricle time series and all their derivatives (e.g., 18p)
#       There is also 16pNoGSR, which is the above, but without gsr and its derivative.
#   Ultimately, read below for other combinations; what I would consider the best option that does NOT include GSR is the default, called "24pXaCompCorXVolterra" - read below for what it entails...

# IMPORTANT: In general, only functions step1, step2 and the parameters preceding that will need to be edited. There are many helper functions below, but in theory, they should not be edited.
# Currently, this script is defaulted to create the nuisance regressors in your current working directory (in a sub directory), and the glm output in your current working directory
# The default is set to use data from the HCP352 QC'd data set, so will need to be updated accordingly. 
# For now, this only includes extensive nuisance regression. Any task regression will need to be performed independently after this.

## EXAMPLE USAGE:
# import nuisanceRegressionPipeline as nrp
# nrp.step1_createNuisanceRegressors(nproc=8)
# nrp.step2_nuisanceRegression(nproc=5, model='24pXaCompCorXVolterra',spikeReg=False,zscore=False)

## DISCLAIMER: This is a first draft, so... keep that in mind.

import numpy as np
import os
import glob
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
import multiprocessing as mp
import h5py
import scipy.stats as stats
from scipy import signal
import nibabel as nib
import scipy
import pandas as pd
import time
import warnings
warnings.simplefilter('ignore', np.ComplexWarning)
import regression
from sklearn.linear_model import LinearRegression
import glob

## Define GLOBAL variables (variables accessible to all functions
# Define base data directory
datadir = '/gpfs/loomis/project/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/'
# Define number of frames to skip
framesToSkip = 5
# Define all runs you want to preprocess
allRuns = ['rfMRI_REST1_RL', 'rfMRI_REST1_LR','rfMRI_REST2_RL', 'rfMRI_REST2_LR','tfMRI_EMOTION_RL','tfMRI_EMOTION_LR','tfMRI_GAMBLING_RL','tfMRI_GAMBLING_LR','tfMRI_LANGUAGE_RL','tfMRI_LANGUAGE_LR','tfMRI_MOTOR_RL','tfMRI_MOTOR_LR','tfMRI_RELATIONAL_RL','tfMRI_RELATIONAL_LR','tfMRI_SOCIAL_RL','tfMRI_SOCIAL_LR','tfMRI_WM_RL','tfMRI_WM_LR']
# Define the *output* directory for nuisance regressors
nuis_reg_dir = datadir + 'nuisanceRegressors/'
# Create directory if it doesn't exist
if not os.path.exists(nuis_reg_dir): os.makedirs(nuis_reg_dir)
# Define the *output* directory for preprocessed data
#

# All subjects
sessions=['02_a1','02_a2','02_b1','02_b2','03_a1','03_a2','03_b1','03_b2','04_a1','04_a2','04_b1','04_b2','06_a1','06_a2','06_b1','06_b2','09_a1','09_a2','09_b1','09_b2','12_a1','12_a2','12_b1','12_b2','15_a1','15_a2','15_b1','15_b2','18_a1','18_a2','18_b1','18_b2','20_a1','20_a2','20_b1','20_b2','22_a1','22_a2','22_b1','22_b2','25_a1','25_a2','25_b1','25_b2','27_a1','27_a2','27_b1','27_b2','29_a1','29_a2','29_b1','29_b2','31_a1','31_a2','31_b1','31_b2','02_a1','02_a2','02_b1','02_b2','04_a1','04_a2','04_b1','04_b2','08_a1','08_a2','08_b1','08_b2','10_a1','10_a2','10_b1','10_b2','14_a1','14_a2','14_b1','14_b2','17_a1','17_a2','17_b1','17_b2','19_a1','19_a2','19_b1','19_b2','21_a1','21_a2','21_b1','21_b2','24_a1','24_a2','24_b1','24_b2','26_a1','26_a2','26_b1','26_b2','28_a1','28_a2','28_b1','28_b2','30_a1','30_a2','30_b1','30_b2']

subIDs=['02','03','04','06','08','09','10','12','14','15','18','20','22','25','27','29','31','17','19','21','24','26','28','30']
sessionIDs=['_a1','_a2','_b1','_b2']

def taskGLMforFIR(taskmodel='canonical',model='qunex',space='parcellated',spikeReg=False,zscore=False):
    """
    Function to perform a task GLM (in conjunction with nuisance regression on each rest run separately)
    
    Input parameters:
        taskmodel :
            Type of model: 'canonical' or 'fir'
        model :
            qunex (similar to 16p no gsr, but a variant) -- uses qunex output time series 
                Includes (32 regressors total):
                    - Movement parameters (6 directions), their derivatives, and all quadratics (24 regressors)
                    - White matter, white matter derivatives, and their quadratics (4 regressors)
                    - Ventricles, ventricle derivatives, and their quadratics (4 regressors)
        spikeReg : 
            spike regression (Satterthwaite et al. 2013) [True/False]
            Note, inclusion of this will add additional set of regressors, which is custom for each subject/run
        zscore   : 
            Normalize data (across time) prior to fitting regression
    """
    outputdir = datadir + '../derivatives/postprocessing/'
    # Iterate through each subject
    runs = range(1,9)
    sess_suffix = ['a1','a2','b1','b2']

    for subj in subIDs:
        # Note all Rest fMRI runs are in ${sub}_b2
        # Note rest runs are bold9 and bold10.nii.gz
        # Iterate through each run
        for sess_id in [1,2]:
            sess_list = []
            sess_list.append(subj + '_a' + sess_id)
            sess_list.append(subj + '_b' + sess_id)
            print ('Running task regression on session | set', sess_id)
            print ('\tModel:', model, '| task model:', taskmodel, '| spikeReg:', spikeReg, '| zscore:', zscore)
            # Run nuisance regression for this subject's run, using a helper function defined below
            # Data will be output in 'outputdir', defined above
            outputfilename = outputdir + 'sessionset' + str(sess_id) + '_tfMRI_' + space + '_' + taskmodel + '_' + model + '_bold' + str(run)
            #try: 
            _postProcRegression(sess_list, runs, outputfilename, space=space, taskmodel=taskmodel, nuisModel=model)
            #except:
            #    print('Looks like no files')

def taskGLM(taskmodel='canonical',model='qunex',space='parcellated',spikeReg=False,zscore=False):
    """
    Function to perform a task GLM (in conjunction with nuisance regression on each rest run separately)
    
    Input parameters:
        taskmodel :
            Type of model: 'canonical' or 'fir'
        model :
            qunex (similar to 16p no gsr, but a variant) -- uses qunex output time series 
                Includes (32 regressors total):
                    - Movement parameters (6 directions), their derivatives, and all quadratics (24 regressors)
                    - White matter, white matter derivatives, and their quadratics (4 regressors)
                    - Ventricles, ventricle derivatives, and their quadratics (4 regressors)
        spikeReg : 
            spike regression (Satterthwaite et al. 2013) [True/False]
            Note, inclusion of this will add additional set of regressors, which is custom for each subject/run
        zscore   : 
            Normalize data (across time) prior to fitting regression
    """
    outputdir = datadir + '../derivatives/postprocessing/'
    # Iterate through each subject
    runs = range(1,9)
    sess_suffix = ['a1','a2','b1','b2']

    for subj in subIDs:
        # Note all Rest fMRI runs are in ${sub}_b2
        # Note rest runs are bold9 and bold10.nii.gz
        # Iterate through each run
        for sess_id in sess_suffix:
            sess = subj + '_' + sess_id
            for run in runs:
                print ('Running task regression on session', sess, '| run', run)
                print ('\tModel:', model, '| task model:', taskmodel, '| spikeReg:', spikeReg, '| zscore:', zscore)
                # Run nuisance regression for this subject's run, using a helper function defined below
                # Data will be output in 'outputdir', defined above
                outputfilename = outputdir + sess + '_tfMRI_' + space + '_' + taskmodel + '_' + model + '_bold' + str(run)
                #try: 
                _postProcRegression([sess], [run], outputfilename, space=space, taskmodel=taskmodel, nuisModel=model)
                #except:
                #    print('Looks like no files')

def rsNuisRegression(model='qunex',space='parcellated',spikeReg=False,zscore=False):
    """
    Function to perform nuisance regression on each rest run separately
    
    Input parameters:
        model   : model choices for linear regression. Models include:
                    1. 24pXaCompCorXVolterra [default]
                        Variant from Ciric et al. 2017. 
                        Includes (64 regressors total):
                            - Movement parameters (6 directions; x, y, z displacement, and 3 rotations) and their derivatives, and their quadratics (24 regressors)
                            - aCompCor (5 white matter and 5 ventricle components) and their derivatives, and their quadratics (40 regressors)
                    2. 18p (the lab's legacy default)
                        Includes (18 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - Global signal and its derivative (2 regressors)
                            - White matter signal and its derivative (2 regressors)
                            - Ventricles signal and its derivative (2 regressors)
                    3. 16pNoGSR (the legacy default, without GSR)
                        Includes (16 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - White matter signal and its derivative (2 regressors)
                            - Ventricles signal and its derivative (2 regressors)
                    4. 12pXaCompCor (Typical motion regression, but using CompCor (noGSR))
                        Includes (32 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - aCompCor (5 white matter and 5 ventricle components) and their derivatives (no quadratics; 20 regressors)
                    5. 36p (State-of-the-art, according to Ciric et al. 2017)
                        Includes (36 regressors total - same as legacy, but with quadratics):
                            - Movement parameters (6 directions) and their derivatives and quadratics (24 regressors)
                            - Global signal and its derivative and both quadratics (4 regressors)
                            - White matter signal and its derivative and both quadratics (4 regressors)
                            - Ventricles signal and its derivative (4 regressors)
                    6. qunex (similar to 16p no gsr, but a variant) -- uses qunex output time series 
                        Includes (32 regressors total):
                            - Movement parameters (6 directions), their derivatives, and all quadratics (24 regressors)
                            - White matter, white matter derivatives, and their quadratics (4 regressors)
                            - Ventricles, ventricle derivatives, and their quadratics (4 regressors)
        spikeReg : spike regression (Satterthwaite et al. 2013) [True/False]
                        Note, inclusion of this will add additional set of regressors, which is custom for each subject/run
        zscore   : Normalize data (across time) prior to fitting regression
    """
    outputdir = datadir + '../derivatives/postprocessing/'
    # Iterate through each subject
    #runs = ['bold9','bold10']
    runs = range(9,11)

    for subj in subIDs:
        # Note all Rest fMRI runs are in ${sub}_b2
        # Note rest runs are bold9 and bold10.nii.gz
        # Iterate through each run
        sess = subj + '_b2'
        for run in runs:
            print ('Running regression on session', sess, '| run', run)
            print ('\tModel:', model, 'with spikeReg:', spikeReg, '| zscore:', zscore)
            # Run nuisance regression for this subject's run, using a helper function defined below
            # Data will be output in 'outputdir', defined above
            outputfilename = outputdir + sess + '_rsfMRI_' + space + '_' + model + '_bold' + str(run)
            try: 
                _postProcRegression([sess], [run], outputfilename, space=space, taskmodel=None, nuisModel=model)
            except:
                print('Looks like no files')


#########################################
# Helper functions

def _postProcRegression(sessions,runs,outputfilename,space='parcellated',taskmodel=None, nuisModel='qunex'):
    """
    This function runs a post processing regression on a single subject
    Will only regress out task-timing, using either a canonical HRF model or FIR model
    Input parameters:
        sessions    :   a list/array of strings specifying session IDs
        runs        :   a list/array of strings specifying run IDs
        taskmodel        :  task model (Default: None (rs nuisance regression); [None, 'canonical','FIR'] 
    """

    data = []
    nuisanceRegressors = []
    constant = []
    lineartrend = []
    if taskmodel is not None: 
        task_regs = []
        regression_index = []

    df_run_params = {}
    df_run_params['Session'] = []
    df_run_params['Run'] = []
    df_run_params['SkippedFrames'] = []
    df_run_params['NumTimepoints'] = []
    for sess in sessions:
        for run in runs:
            run_id = 'bold' + str(run)

            if space=='parcellated':
                rundata = loadRawParcellatedData(sess,run_id)
            elif space=='vertex':
               rundata = loadRawVertexData(sess,run_id)

            num_timepoints = rundata.shape[0]
            df_run_params['Session'].append(sess)
            df_run_params['Run'].append(run)
            df_run_params['SkippedFrames'].append(framesToSkip)
            df_run_params['NumTimepoints'].append(num_timepoints)

            tMask = np.ones((num_timepoints,))
            tMask[:framesToSkip] = 0
            tMask = np.asarray(tMask,dtype=bool)

            # Skip frames
            rundata = rundata[tMask,:]
            
#            # Demean each run
#            rundata = signal.detrend(rundata,axis=0,type='constant')
#            # Detrend each run
#            rundata = signal.detrend(rundata,axis=0,type='linear')
            
            # Load in nuisance regressors
            nuisregs = loadNuisanceRegressors(sess,run_id,num_timepoints)
            # Skip nuis regs frames
            nuisregs = nuisregs[tMask,:]

            nuisanceRegressors.extend(nuisregs)
            data.extend(rundata)

    df_run_params = pd.DataFrame(df_run_params)

    if taskmodel is not None:
        #  Load in task timing file
        tasktiming = loadTaskTiming(sessions, runs, df_run_params, taskModel=taskmodel)
        task_regs.extend(tasktiming['taskRegressors'][tMask,:])
        regression_index.extend(tasktiming['stimIndex'])


    data = np.asarray(data)
    nuisanceRegressors = np.asarray(nuisanceRegressors)
    nROIs = data.shape[1]

    return data, tasktiming

#    # Load regressors for data
#    if taskmodel==None:
#        print('Running standard nuisance regression')
#        #X = loadTaskTiming(subj, task, taskModel=taskModel, nRegsFIR=25)
#
#        allRegressors = nuisanceRegressors
#    else: 
#        task_regs = np.asarray(task_regs)
#        regression_index = np.asarray(regression_index)
#        allRegressors = np.hstack((task_regs,nuisanceRegressors))
#
#
#    #betas, resid = regression.regression(data, allRegressors, constant=True)
#
#    reg = LinearRegression().fit(allRegressors,data)
#    betas = reg.coef_
#    y_pred = reg.predict(allRegressors)
#    resid = data - y_pred
#    resid = data
#    
#    betas = betas.T # Exclude nuisance regressors
#    residual_ts = resid.T
#
#    # For task data, only include task-related regressors
#    if taskmodel is not None:
#        betas = betas[:len(regression_index),:].T
#
#    h5f = h5py.File(outputfilename + '.h5','a')
#    outname1 = 'residuals'
#    outname2 = 'betas'
#    if taskmodel is not None: np.savetxt(outputfilename + '_taskIndex.csv', regression_index,delimiter=',',fmt ='% s')
#    try:
#        h5f.create_dataset(outname1,data=residual_ts)
#        h5f.create_dataset(outname2,data=betas)
#    except:
#        del h5f[outname1], h5f[outname2]
#        h5f.create_dataset(outname1,data=residual_ts)
#        h5f.create_dataset(outname2,data=betas)
#    h5f.close()

def loadTaskTiming(sessions, runs, df_run_params, taskModel='canonical', nRegsFIR=20):
    """
    Loads task timings for each run separately
    """
    trLength = 1.0
    df_all = pd.DataFrame()
    conditions = []
    trcount = 0
    for sess in sessions:
        for run in runs:
            subj = sess[:2] # first 2 characters form the subject ID
            sess_id = sess[-2:] # last 2 characters form the session
            tasktime_dir = datadir + 'sessions/' + sess + '/bids/func/'
            stimfile = glob.glob(tasktime_dir + 'sub-' + subj + '_ses-' + sess_id + '*' + str(run) + '_events.tsv')[0]
            stimdf = pd.read_csv(stimfile,sep='\t') 
            # Add number of timepoints from last run and subtract the number of skipped frames
            skipped_frames = df_run_params.SkippedFrames[(df_run_params.Session==sess) & (df_run_params.Run==run)].values
            stimdf.startTRreal = stimdf.startTRreal.values + trcount - skipped_frames
            trcount += df_run_params.NumTimepoints[(df_run_params.Session==sess) & (df_run_params.Run==run)].values - skipped_frames
            if stimdf.startTRreal.min()<0:
                raise Exception("Start TR CANNOT be less than 0")
            df_all = df_all.append(stimdf)
            conditions.extend(list(np.unique(stimdf.trial_type.values)))

    df_all = df_all.reset_index(drop=True) # reset indices
    trcount = int(trcount)
    conditions = list(np.unique(conditions))
    conditions.remove('Instruct') # Remove this - not a condition (and no timing information)
    # Note that the event files don't have a distinction between 0-back nd 2-back conditions for both object recognition and verbal recognition tasks
    # conditions.remove('Rest')
    tasks = np.unique(df_all.taskName.values)

    stim_mat = np.zeros((trcount,len(conditions)))
    stim_index = []

    stimcount = 0
    for cond in conditions:
        print('Condition', cond)
        conddf = df_all.loc[df_all.trial_type==cond]
        for ind in conddf.index:
            trstart = int(df_all.startTRreal[ind])
            duration = df_all.duration[ind]
            print('\tduration:', duration)
            trend = int(np.ceil(trstart + duration))
            stim_mat[trstart:trend,stimcount] = 1.0

        stim_index.append(cond)
        stimcount += 1 # go to next condition

    if taskModel=='betaseries':

        beta_mat = []
        stimcount = 0
        for cond in conditions:
            print('Condition', cond)
            conddf = df_all.loc[df_all.trial_type==cond]
            for ind in conddf.index:
                trstart = int(df_all.startTRreal[ind])
                duration = df_all.duration[ind]
                print('\tduration:', duration)
                trend = int(np.ceil(trstart + duration))
                stim_mat[trstart:trend,stimcount] = 1.0

            stim_index.append(cond)
            stimcount += 1 # go to next condition


    ## 
    if taskModel=='FIR':
        ## TODO
        taskRegressors = convertDesignMat2FIR(stim_mat,hrf_lag=nRegsFIR)
#        ## First set up FIR design matrix
#        stim_index = []
#        taskStims_FIR = [] 
#        for stim in range(stim_mat.shape[1]):
#            taskStims_FIR.append([])
#            time_ind = np.where(stim_mat[:,stim]==1)[0]
#            blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
#            # Identify the longest block - set FIR duration to longest block
#            maxRegsForBlocks = 0
#            for block in blocks:
#                if len(block) > maxRegsForBlocks: maxRegsForBlocks = len(block)
#            taskStims_FIR[stim] = np.zeros((stim_mat.shape[0],maxRegsForBlocks+nRegsFIR)) # Task timing for this condition is TR x length of block + FIR lag
#            stim_index.extend(np.repeat(stim,maxRegsForBlocks+nRegsFIR))
#        stim_index = np.asarray(stim_index)
#
#        ## Now fill in FIR design matrix
#        # Make sure to cut-off FIR models for each run separately
#        trcount = 0
#
#        for run in range(nRunsPerTask):
#            trstart = trcount
#            trend = trstart + nTRsPerRun
#                
#            for stim in range(stim_mat.shape[1]):
#                time_ind = np.where(stim_mat[:,stim]==1)[0]
#                blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
#                for block in blocks:
#                    reg = 0
#                    for tr in block:
#                        # Set impulses for this run/task only
#                        if trstart < tr < trend:
#                            taskStims_FIR[stim][tr,reg] = 1
#                            reg += 1
#
#                        if not trstart < tr < trend: continue # If TR not in this run, skip this block
#
#                    # If TR is not in this run, skip this block
#                    if not trstart < tr < trend: continue
#
#                    # Set lag due to HRF
#                    for lag in range(1,nRegsFIR+1):
#                        # Set impulses for this run/task only
#                        if trstart < tr+lag < trend:
#                            taskStims_FIR[stim][tr+lag,reg] = 1
#                            reg += 1
#            trcount += nTRsPerRun
#        
#
#        taskStims_FIR2 = np.zeros((stim_mat.shape[0],1))
#        task_index = []
#        for stim in range(stim_mat.shape[1]):
#            task_index.extend(np.repeat(stim,taskStims_FIR[stim].shape[1]))
#            taskStims_FIR2 = np.hstack((taskStims_FIR2,taskStims_FIR[stim]))
#
#        taskStims_FIR2 = np.delete(taskStims_FIR2,0,axis=1)
#
#        #taskRegressors = np.asarray(taskStims_FIR)
#        taskRegressors = taskStims_FIR2
#    
#        # To prevent SVD does not converge error, make sure there are no columns with 0s
#        zero_cols = np.where(np.sum(taskRegressors,axis=0)==0)[0]
#        taskRegressors = np.delete(taskRegressors, zero_cols, axis=1)
#        stim_index = np.delete(stim_index, zero_cols)

    elif taskModel=='canonical':
        ## 
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
        taskStims_HRF = np.zeros(stim_mat.shape)
        spm_hrfTS = spm_hrf(trLength,oversampling=1)
       

        for stim in range(stim_mat.shape[1]):

            # Perform convolution
            tmpconvolve = np.convolve(stim_mat[:,stim],spm_hrfTS)
            taskStims_HRF[:,stim] = tmpconvolve[:num_timepoints]


        taskRegressors = taskStims_HRF.copy()
    

    # Create temporal mask (skipping which frames?)
    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors
    output['taskDesignMat'] = stim_mat
    output['stimIndex'] = stim_index

    return output

def loadNuisanceRegressors(sess, run, num_timepoints, model='qunex', spikeReg=False, zscore=False):
    """
    This function runs nuisance regression on the Glasser Parcels (360) on a single sessects run
    Will only regress out noise parameters given the model choice (see below for model options)
    Input parameters:
        sess    : sess number as a string
        run     : task run
        model   : model choices for linear regression. Models include:
                    1. 24pXaCompCorXVolterra [default]
                        Variant from Ciric et al. 2017. 
                        Includes (64 regressors total):
                            - Movement parameters (6 directions; x, y, z displacement, and 3 rotations) and their derivatives, and their quadratics (24 regressors)
                            - aCompCor (5 white matter and 5 ventricle components) and their derivatives, and their quadratics (40 regressors)
                    2. 18p (the legacy default)
                        Includes (18 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - Global signal and its derivative (2 regressors)
                            - White matter signal and its derivative (2 regressors)
                            - Ventricles signal and its derivative (2 regressors)
                    3. 16pNoGSR (the legacy default, without GSR)
                        Includes (16 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - White matter signal and its derivative (2 regressors)
                            - Ventricles signal and its derivative (2 regressors)
                    4. 12pXaCompCor (Typical motion regression, but using CompCor (noGSR))
                        Includes (32 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - aCompCor (5 white matter and 5 ventricle components) and their derivatives (no quadratics; 20 regressors)
                    5. 36p (State-of-the-art, according to Ciric et al. 2017)
                        Includes (36 regressors total - same as legacy, but with quadratics):
                            - Movement parameters (6 directions) and their derivatives and quadratics (24 regressors)
                            - Global signal and its derivative and both quadratics (4 regressors)
                            - White matter signal and its derivative and both quadratics (4 regressors)
                            - Ventricles signal and its derivative (4 regressors)
                    6. qunex (similar to 16p no gsr, but a variant) -- uses qunex output time series 
                        Includes (32 regressors total):
                            - Movement parameters (6 directions), their derivatives, and all quadratics (24 regressors)
                            - White matter, white matter derivatives, and their quadratics (4 regressors)
                            - Ventricles, ventricle derivatives, and their quadratics (4 regressors)
        spikeReg : spike regression (Satterthwaite et al. 2013) [True/False]
                        Note, inclusion of this will add additional set of regressors, which is custom for each session/run
        zscore   : Normalize data (across time) prior to fitting regression
    """

    # Load nuisance regressors for this data
    if model=='qunex':
        nuisdir = datadir + 'sessions/' + sess + '/images/functional/movement/'
        # Load physiological signals
        data = pd.read_csv(nuisdir + run + '.nuisance',sep='\s+')
        ventricles_signal = data.V.values[:-2]
        ventricles_signal_deriv = np.zeros(ventricles_signal.shape)
        ventricles_signal_deriv[1:] = ventricles_signal[1:] - ventricles_signal[:-1] 
        ventricles_signal_deriv[0] = np.mean(ventricles_signal_deriv[1:])
        #
        wm_signal = data.WM.values[:-2]
        wm_signal_deriv = np.zeros(wm_signal.shape)
        wm_signal_deriv[1:] = wm_signal[1:] - wm_signal[:-1] 
        wm_signal_deriv[0] = np.mean(wm_signal_deriv[1:])
        #
        global_signal = data.WB.values[:-2]
        global_signal_deriv = np.zeros(global_signal.shape)
        global_signal_deriv[1:] = global_signal[1:] - global_signal[:-1] 
        global_signal_deriv[0] = np.mean(global_signal_deriv[1:])
        #
        motiondat = pd.read_csv(nuisdir + run + '_mov.dat',sep='\s+')
        motionparams = np.zeros((len(motiondat),6)) # time x num params
        motionparams[:,0] = motiondat['dx(mm)'].values
        motionparams[:,1] = motiondat['dy(mm)'].values
        motionparams[:,2] = motiondat['dz(mm)'].values
        motionparams[:,3] = motiondat['X(deg)'].values
        motionparams[:,4] = motiondat['Y(deg)'].values
        motionparams[:,5] = motiondat['Z(deg)'].values
        motionparams_deriv = np.zeros((len(motiondat),6)) # time x num params
        motionparams_deriv[1:,:] = motionparams[1:,:] - motionparams[:-1,:] 
        motionparams_deriv[0,:] = np.mean(motionparams[1:,:],axis=0)
        ## Include quadratics
        motionparams_quadratics = motionparams**2
        motionparams_deriv_quadratics = motionparams_deriv**2

        ## EXCLUDE GLOBAL SIGNAL - my philosophical preference
        physiological_params = np.vstack((wm_signal,wm_signal_deriv,ventricles_signal,ventricles_signal_deriv)).T
        physiological_params_quadratics = physiological_params**2
        nuisanceRegressors = np.hstack((motionparams,motionparams_quadratics,motionparams_deriv,motionparams_deriv_quadratics,physiological_params,physiological_params_quadratics))

        nuisanceRegressors = np.random.random((len(global_signal),3))

    else:
        # load all nuisance regressors for all other regression models
        h5f = h5py.File(nuis_reg_dir + sess + '_nuisanceRegressors.h5','r') 
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # Global signal + derivatives
        global_signal = h5f[run]['global_signal'][:].copy()
        global_signal_deriv = h5f[run]['global_signal_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[run]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))

    if model=='24pXaCompCorXVolterra':
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # WM aCompCor + derivatives
        aCompCor_WM = h5f[run]['aCompCor_WM'][:].copy()
        aCompCor_WM_deriv = h5f[run]['aCompCor_WM_deriv'][:].copy()
        # Ventricles aCompCor + derivatives
        aCompCor_ventricles = h5f[run]['aCompCor_ventricles'][:].copy()
        aCompCor_ventricles_deriv = h5f[run]['aCompCor_ventricles_deriv'][:].copy()
        # Create nuisance regressors design matrix
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, aCompCor_WM, aCompCor_WM_deriv, aCompCor_ventricles, aCompCor_ventricles_deriv))
        quadraticRegressors = nuisanceRegressors**2
        nuisanceRegressors = np.hstack((nuisanceRegressors,quadraticRegressors))
    
    elif model=='18p':
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # Global signal + derivatives
        global_signal = h5f[run]['global_signal'][:].copy()
        global_signal_deriv = h5f[run]['global_signal_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[run]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))

    elif model=='16pNoGSR':
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[run]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))
    
    elif model=='12pXaCompCor':
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # WM aCompCor + derivatives
        aCompCor_WM = h5f[run]['aCompCor_WM'][:].copy()
        aCompCor_WM_deriv = h5f[run]['aCompCor_WM_deriv'][:].copy()
        # Ventricles aCompCor + derivatives
        aCompCor_ventricles = h5f[run]['aCompCor_ventricles'][:].copy()
        aCompCor_ventricles_deriv = h5f[run]['aCompCor_ventricles_deriv'][:].copy()
        # Create nuisance regressors design matrix
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, aCompCor_WM, aCompCor_WM_deriv, aCompCor_ventricles, aCompCor_ventricles_deriv))
    
    elif model=='36p':
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # Global signal + derivatives
        global_signal = h5f[run]['global_signal'][:].copy()
        global_signal_deriv = h5f[run]['global_signal_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[run]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))
        quadraticRegressors = nuisanceRegressors**2
        nuisanceRegressors = np.hstack((nuisanceRegressors,quadraticRegressors))


    if spikeReg:
        # Obtain motion spikes
        try:
            motion_spikes = h5f[run]['motionSpikes'][:].copy()
            nuisanceRegressors = np.hstack((nuisanceRegressors,motion_spikes))
        except:
            print('Spike regression option was chosen... but no motion spikes for sess', sess, '| run', run, '!')
        # Update the model name - to keep track of different model types for output naming
        model = model + '_spikeReg' 

    if zscore:
        model = model + '_zscore'

    if model!='qunex':
        h5f.close()
    
    return nuisanceRegressors

def loadRawParcellatedData(sess,run,datadir='/gpfs/loomis/project/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/'):
    """
    Load in parcellated data for given session and run
    """
    datafile = datadir + sess + '/images/functional/' + run + '_Atlas.LR.Parcels.32k_fs_LR.ptseries.nii'
    data = nib.load(datafile).get_data()
    return data

def loadRawVertexData(sess,run,datadir='/gpfs/loomis/project/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/'):
    """
    Load in surface vertex data for given session and run
    """
    datafile = datadir + sess + '/images/functional/' + run + '_Atlas.dtseries.nii'
    data = nib.load(datafile).get_data()
    return data

def convertDesignMat2FIR(design_mat,hrf_lag=20):
    """
    design_mat
        2d matrix (with time/observations X features/regressors)

    hrf_lag
        Number of time points to include after FIR/task offset to account for HRF lag
    """
    nfeatures = design_mat.shape[1]
    max_tps = design_mat.shape[0]
    firdesign = []
    for i in range(nfeatures):
        tmp = design_mat[:,i].copy()
        # get onsets and offsets
        tmp_deriv = np.diff(tmp)
        onsets = np.where(tmp_deriv==1)[0] + 1 # add 1 to onset since derivative is calculated at the first time point
        offsets = np.where(tmp_deriv==-1)[0]
        #print(onsets,offsets)
        print(np.asarray(onsets)-np.asarray(offsets))
        n_blocks = len(onsets)
        block_duration = offsets[0] - onsets[0]
        # Create FIR design matrix for this condition/feature
#        fir_arr = np.zeros((len(tmp),block_duration+hrf_lag)) # Number of time points for each FIR block
#        for block in range(n_blocks):
#            reg = 0
#            for tp in range(onsets[block],offsets[block]):
#                fir_arr[tp,reg] = 1
#                reg += 1
#            tp += 1 # next time point
#
#            # Now include lag after offset (to account for slow HRF lag)
#            for lag in range(hrf_lag):
#                if tp>=max_tps: continue
#                fir_arr[tp,reg] = 1
#                reg += 1
#                tp += 1
#
#        firdesign.extend(fir_arr.T)
#
#    firdesign = np.asarray(firdesign)
#    return firdesign

        



