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
import time
import warnings
warnings.simplefilter('ignore', np.ComplexWarning)
from regression import regression

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
outputdir = datadir + 'hcpPostProcCiric/' 
#

# All subjects
sessions=['02_a1','02_a2','02_b1','02_b2','03_a1','03_a2','03_b1','03_b2','04_a1','04_a2','04_b1','04_b2','06_a1','06_a2','06_b1','06_b2','09_a1','09_a2','09_b1','09_b2','12_a1','12_a2','12_b1','12_b2','15_a1','15_a2','15_b1','15_b2','18_a1','18_a2','18_b1','18_b2','20_a1','20_a2','20_b1','20_b2','22_a1','22_a2','22_b1','22_b2','25_a1','25_a2','25_b1','25_b2','27_a1','27_a2','27_b1','27_b2','29_a1','29_a2','29_b1','29_b2','31_a1','31_a2','31_b1','31_b2','02_a1','02_a2','02_b1','02_b2','04_a1','04_a2','04_b1','04_b2','08_a1','08_a2','08_b1','08_b2','10_a1','10_a2','10_b1','10_b2','14_a1','14_a2','14_b1','14_b2','17_a1','17_a2','17_b1','17_b2','19_a1','19_a2','19_b1','19_b2','21_a1','21_a2','21_b1','21_b2','24_a1','24_a2','24_b1','24_b2','26_a1','26_a2','26_b1','26_b2','28_a1','28_a2','28_b1','28_b2','30_a1','30_a2','30_b1','30_b2']
sessions=['08_a1','17_a2']

# Session names, excluding 08_a1 and 17_a2
sessions=['02_a1','02_a2','02_b1','02_b2','03_a1','03_a2','03_b1','03_b2','04_a1','04_a2','04_b1','04_b2','06_a1','06_a2','06_b1','06_b2','09_a1','09_a2','09_b1','09_b2','12_a1','12_a2','12_b1','12_b2','15_a1','15_a2','15_b1','15_b2','18_a1','18_a2','18_b1','18_b2','20_a1','20_a2','20_b1','20_b2','22_a1','22_a2','22_b1','22_b2','25_a1','25_a2','25_b1','25_b2','27_a1','27_a2','27_b1','27_b2','29_a1','29_a2','29_b1','29_b2','31_a1','31_a2','31_b1','31_b2','02_a1','02_a2','02_b1','02_b2','04_a1','04_a2','04_b1','04_b2','08_a2','08_b1','08_b2','10_a1','10_a2','10_b1','10_b2','14_a1','14_a2','14_b1','14_b2','17_a1','17_b1','17_b2','19_a1','19_a2','19_b1','19_b2','21_a1','21_a2','21_b1','21_b2','24_a1','24_a2','24_b1','24_b2','26_a1','26_a2','26_b1','26_b2','28_a1','28_a2','28_b1','28_b2','30_a1','30_a2','30_b1','30_b2']


def postProcRegression(nproc=5, model='24pXaCompCorXVolterra',spikeReg=False,zscore=False):
    """
    Function to perform nuisance regression on each run separately
    This uses parallel processing, but parallelization occurs within each subject
    Each subject runs regression on each region/voxel in parallel, thus iterating subjects and runs serially
    
    Input parameters:
        subj    : subject number as a string
        run     : task run
        outputdir: Directory for GLM output, as an h5 file (each run will be contained within each h5)
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
        spikeReg : spike regression (Satterthwaite et al. 2013) [True/False]
                        Note, inclusion of this will add additional set of regressors, which is custom for each subject/run
        zscore   : Normalize data (across time) prior to fitting regression
        nproc = number of processes to use via multiprocessing
    """
    # Iterate through each subject
    for subj in subjNums:
        # Iterate through each run
        for run in allRuns:
            print 'Running regression on subject', subj, '| run', run
            print '\tModel:', model, 'with spikeReg:', spikeReg, '| zscore:', zscore
            ## Load in data to be preprocessed - This needs to be a space x time 2d array
            inputfile = datadir + '/hcpPreprocessedData/' + subj + '_GlasserParcellated_' + run + '_LR.csv'
            # Load data
            data = np.loadtxt(inputfile,delimiter=',')
            # Run nuisance regression for this subject's run, using a helper function defined below
            # Data will be output in 'outputdir', defined above
            _nuisanceRegression(subj, run, data, outputdir, model=model,spikeReg=spikeReg,zscore=zscore,nproc=nproc)


#########################################
# Helper functions

def _postProcRegression(sessions,runs,task=None, taskModel=None, nuisModel='qunex', nproc=8):
    """
    This function runs a post processing regression on a single subject
    Will only regress out task-timing, using either a canonical HRF model or FIR model
    Input parameters:
        sessions    :   a list/array of strings specifying session IDs
        runs        :   a list/array of strings specifying run IDs
        task        :   task or session name (?) 
        taskModel   :   regression model (default: None (nuisance only) ['canonicalTask',firTask'])
        nproc       :   number of processes to use via multiprocessing
    """

    data = []
    nuisanceRegressors = []
    for sess in sessions:
        for run in runs:
            tmp = loadRawParcellatedData(sess,run)
            num_timepoints = tmp.shape[0]
            data.extend(tmp)
            nuisanceRegressors.extend(loadNuisanceRegressors(sess,run,num_timepoints)

    

    tMask = np.ones((data.shape[1],))
    tMask[:framesToSkip] = 0

    # Skip frames
    data = data[:,framesToSkip:]
    
    # Demean each run
    data = signal.detrend(data,axis=1,type='constant')
    # Detrend each run
    data = signal.detrend(data,axis=1,type='linear')
    tMask = np.asarray(tMask,dtype=bool)
    
    nROIs = data.shape[0]



    # Identify number of ROIs
    nROIs = data.shape[0]
    # Identify number of TRs
    nTRs = data.shape[1]

    # Load regressors for data
    if task!=None:
        print('Running standard nuisance regression')
        #X = loadTaskTiming(subj, task, taskModel=taskModel, nRegsFIR=25)

        taskRegs = X['taskRegressors'] # These include the two binary regressors
        nuisanceRegressors = loadNuisanceRegressors(subj,
        allregressors = 

    betas, resid = regression.regression(data.T, taskRegs, constant=True)
    
    betas = betas.T # Exclude nuisance regressors
    residual_ts = resid.T

    h5f = h5py.File(outputdir + subj + '_glmOutput_data.h5','a')
    outname1 = 'taskRegression/' + task + '_' + nuisModel + '_taskReg_resid_' + taskModel
    outname2 = 'taskRegression/' + task + '_' + nuisModel + '_taskReg_betas_' + taskModel
    try:
        h5f.create_dataset(outname1,data=residual_ts)
        h5f.create_dataset(outname2,data=betas)
    except:
        del h5f[outname1], h5f[outname2]
        h5f.create_dataset(outname1,data=residual_ts)
        h5f.create_dataset(outname2,data=betas)
    h5f.close()

def loadTaskTiming(subj, task, taskModel='canonical', nRegsFIR=25):
    nRunsPerTask = 2

    taskkey = task[6:] # Define string identifier for tasks
    taskEVs = taskEV_Identifier[taskkey]
    stimMat = np.zeros((taskLength[taskkey]*nRunsPerTask,len(taskEV_Identifier[taskkey])))
    stimdir = basedir + 'timingfiles3/'
    stimfiles = glob.glob(stimdir + subj + '*EV*' + taskkey + '*1D')
    
    for stimcount in range(len(taskEVs)):
        ev = taskEVs[stimcount] + 1
        stimfile = glob.glob(stimdir + subj + '*EV' + str(ev) + '_' + taskkey + '*1D')
        stimMat[:,stimcount] = np.loadtxt(stimfile[0])

    nTRsPerRun = int(stimMat.shape[0]/2.0)

    ## 
    if taskModel=='FIR':
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)

        ## First set up FIR design matrix
        stim_index = []
        taskStims_FIR = [] 
        for stim in range(stimMat.shape[1]):
            taskStims_FIR.append([])
            time_ind = np.where(stimMat[:,stim]==1)[0]
            blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
            # Identify the longest block - set FIR duration to longest block
            maxRegsForBlocks = 0
            for block in blocks:
                if len(block) > maxRegsForBlocks: maxRegsForBlocks = len(block)
            taskStims_FIR[stim] = np.zeros((stimMat.shape[0],maxRegsForBlocks+nRegsFIR)) # Task timing for this condition is TR x length of block + FIR lag
            stim_index.extend(np.repeat(stim,maxRegsForBlocks+nRegsFIR))
        stim_index = np.asarray(stim_index)

        ## Now fill in FIR design matrix
        # Make sure to cut-off FIR models for each run separately
        trcount = 0

        for run in range(nRunsPerTask):
            trstart = trcount
            trend = trstart + nTRsPerRun
                
            for stim in range(stimMat.shape[1]):
                time_ind = np.where(stimMat[:,stim]==1)[0]
                blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
                for block in blocks:
                    reg = 0
                    for tr in block:
                        # Set impulses for this run/task only
                        if trstart < tr < trend:
                            taskStims_FIR[stim][tr,reg] = 1
                            reg += 1

                        if not trstart < tr < trend: continue # If TR not in this run, skip this block

                    # If TR is not in this run, skip this block
                    if not trstart < tr < trend: continue

                    # Set lag due to HRF
                    for lag in range(1,nRegsFIR+1):
                        # Set impulses for this run/task only
                        if trstart < tr+lag < trend:
                            taskStims_FIR[stim][tr+lag,reg] = 1
                            reg += 1
            trcount += nTRsPerRun
        

        taskStims_FIR2 = np.zeros((stimMat.shape[0],1))
        task_index = []
        for stim in range(stimMat.shape[1]):
            task_index.extend(np.repeat(stim,taskStims_FIR[stim].shape[1]))
            taskStims_FIR2 = np.hstack((taskStims_FIR2,taskStims_FIR[stim]))

        taskStims_FIR2 = np.delete(taskStims_FIR2,0,axis=1)

        #taskRegressors = np.asarray(taskStims_FIR)
        taskRegressors = taskStims_FIR2
    
        # To prevent SVD does not converge error, make sure there are no columns with 0s
        zero_cols = np.where(np.sum(taskRegressors,axis=0)==0)[0]
        taskRegressors = np.delete(taskRegressors, zero_cols, axis=1)
        stim_index = np.delete(stim_index, zero_cols)

    elif taskModel=='canonical':
        ## 
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
        taskStims_HRF = np.zeros(stimMat.shape)
        spm_hrfTS = spm_hrf(trLength,oversampling=1)
       
        trcount = 0
        for run in range(nRunsPerTask):
            trstart = trcount
            trend = trstart + nTRsPerRun

            for stim in range(stimMat.shape[1]):

                # Perform convolution
                tmpconvolve = np.convolve(stimMat[trstart:trend,stim],spm_hrfTS)
                tmpconvolve_run = tmpconvolve[:nTRsPerRun] # Make sure to cut off at the end of the run
                taskStims_HRF[trstart:trend,stim] = tmpconvolve_run

            trcount += nTRsPerRun

        taskRegressors = taskStims_HRF.copy()
    
        stim_index = []
        for stim in range(stimMat.shape[1]):
            stim_index.append(stim)
        stim_index = np.asarray(stim_index)


    # Create temporal mask (skipping which frames?)
    tMask = []
    tmp = np.ones((nTRsPerRun,), dtype=bool)
    tmp[:framesToSkip] = False
    tMask.extend(tmp)
    tMask.extend(tmp)
    tMask = np.asarray(tMask,dtype=bool)

    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors[tMask,:]
    output['taskDesignMat'] = stimMat[tMask,:]
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
        nproc = number of processes to use via multiprocessing
    """

    # Load nuisance regressors for this data
    if model=='qunex':
        nuisdir = datadir + 'sessions/' + sess + '/images/functional/movement'
        # Load physiological signals
        data = pd.read_csv(nuisdir + run + '.nuisance',sep='\s+')
        ventricles_signal = data.V.values[:-2]
        ventricle_signal_deriv = np.zeros(ventricle_signal.shape)
        ventricle_signal_deriv[1:] = ventricle_signal[1:] - ventricle_signal[:-1] 
        ventricle_signal_deriv[0] = np.mean(ventricle_signal_deriv[1:])
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
        motionparams = np.zeros((len(mov),6)) # time x num params
        motionparams[:,0] = motiondat.['dx(mm)'].values
        motionparams[:,1] = motiondat.['dy(mm)'].values
        motionparams[:,2] = motiondat.['dz(mm)'].values
        motionparams[:,3] = motiondat.['X(deg)'].values
        motionparams[:,4] = motiondat.['Y(deg)'].values
        motionparams[:,5] = motiondat.['Z(deg)'].values
        motionparams_deriv = np.zeros((len(mov),6)) # time x num params
        motionparams_deriv[1:,:] = motionparams[1:,:] - motionparams[:-1,:] 
        motionparams_deriv[0,:] = np.mean(motionparams[1:,:],axis=0)
        ## Include quadratics
        motionparams_quadratics = motionparams**2
        motionparams_deriv_quadratics = motionparams_deriv**2

        ## EXCLUDE GLOBAL SIGNAL - my philosophical preference
        physiological_params = hp.vstack((wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T
        physiological_params_quadratics = physiological_params**2
        nuisanceRegressors = hp.hstack((motionparams,motionparams_quadratics,motionparams_deriv,motionparams_deriv_quadratics,physiological_params,physiological_params_quadratics))

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
            print 'Spike regression option was chosen... but no motion spikes for sess', sess, '| run', run, '!'
        # Update the model name - to keep track of different model types for output naming
        model = model + '_spikeReg' 

    if zscore:
        model = model + '_zscore'

    if model!='qunex':
        h5f.close()

    # Skip first 5 frames of nuisanceRegressors, too
    nuisanceRegressors = nuisanceRegressors[framesToSkip:,:].copy()
    
    return nuisanceRegressors

def loadRawParcellatedData(sess,run,datadir='/gpfs/loomis/project/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/sessions/'):
    """
    Load in parcellated data for given session and run
    """
    datafile = datadir + sess + '/images/functional/' + run + '_Atlas.LR.Parcels.32k_fs_LR.ptseries.nii'
    data = nib.load(datafile).get_data()
    return data
