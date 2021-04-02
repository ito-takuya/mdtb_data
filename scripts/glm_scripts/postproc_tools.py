# Takuya Ito

# Post-processing nuisance regression using Ciric et al. 2017 inspired best-practices
# Takes output from qunex (up to hcp5 + extracting nuisance signals)

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
from sklearn.linear_model import LinearRegression
import glob

## Define GLOBAL variables (variables accessible to all functions
# Define base data directory
datadir = '/gpfs/loomis/project/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/'
# Define number of frames to skip
framesToSkip = 5
# Define the *output* directory for nuisance regressors
nuis_reg_dir = datadir + 'nuisanceRegressors/'
# Define the *output* directory for preprocessed data
#
#########################################

def loadTaskTimingCanonical(sess, run, num_timepoints, nRegsFIR=20):
    """
    Loads task timings for each run separately
    """
    trLength = 1.0
    subj = sess[:2] # first 2 characters form the subject ID
    sess_id = sess[-2:] # last 2 characters form the session
    tasktime_dir = datadir + 'sessions/' + sess + '/bids/func/'
    stimfile = glob.glob(tasktime_dir + 'sub-' + subj + '_ses-' + sess_id + '*' + str(run) + '_events.tsv')[0]
    stimdf = pd.read_csv(stimfile,sep='\t') 
    conditions = np.unique(stimdf.trial_type.values)
    conditions = list(conditions)
    conditions.remove('Instruct') # Remove this - not a condition (and no timing information)
    # Note that the event files don't have a distinction between 0-back nd 2-back conditions for both object recognition and verbal recognition tasks
    # conditions.remove('Rest')
    tasks = np.unique(stimdf.taskName.values)

    stim_mat = np.zeros((num_timepoints,len(conditions)))
    stim_index = []

    stimcount = 0
    for cond in conditions:
        conddf = stimdf.loc[stimdf.trial_type==cond]
        for ind in conddf.index:
            trstart = int(conddf.startTRreal[ind])
            duration = conddf.duration[ind]
            trend = int(trstart + duration)
            stim_mat[trstart:trend,stimcount] = 1.0

        stim_index.append(cond)
        stimcount += 1 # go to next condition


    ## 
    if taskModel=='FIR':
        ## TODO
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)

        ## First set up FIR design matrix
        stim_index = []
        taskStims_FIR = [] 
        for stim in range(stim_mat.shape[1]):
            taskStims_FIR.append([])
            time_ind = np.where(stim_mat[:,stim]==1)[0]
            blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
            # Identify the longest block - set FIR duration to longest block
            maxRegsForBlocks = 0
            for block in blocks:
                if len(block) > maxRegsForBlocks: maxRegsForBlocks = len(block)
            taskStims_FIR[stim] = np.zeros((stim_mat.shape[0],maxRegsForBlocks+nRegsFIR)) # Task timing for this condition is TR x length of block + FIR lag
            stim_index.extend(np.repeat(stim,maxRegsForBlocks+nRegsFIR))
        stim_index = np.asarray(stim_index)

        ## Now fill in FIR design matrix
        # Make sure to cut-off FIR models for each run separately
        trcount = 0

        for run in range(nRunsPerTask):
            trstart = trcount
            trend = trstart + nTRsPerRun
                
            for stim in range(stim_mat.shape[1]):
                time_ind = np.where(stim_mat[:,stim]==1)[0]
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
        

        taskStims_FIR2 = np.zeros((stim_mat.shape[0],1))
        task_index = []
        for stim in range(stim_mat.shape[1]):
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

def loadTaskTimingBetaSeries(sess, run, num_timepoints):
    """
    Loads task timings for each run separately
    """
    trLength = 1.0
    subj = sess[:2] # first 2 characters form the subject ID
    sess_id = sess[-2:] # last 2 characters form the session
    tasktime_dir = datadir + 'sessions/' + sess + '/bids/func/'
    stimfile = glob.glob(tasktime_dir + 'sub-' + subj + '_ses-' + sess_id + '*' + str(run) + '_events.tsv')[0]
    stimdf = pd.read_csv(stimfile,sep='\t') 

    # number of betas correspond to all task conditions "excluding" the instruction screen condition for each task
    n_betas = np.sum(stimdf.trial_type!="Instruct")
    index_loc = np.where(stimdf.trial_type!="Instruct")[0]
    
    stim_mat = np.zeros((num_timepoints,n_betas))
    stim_index = []

    col_count = 0 
    for i in index_loc:
        trstart = int(np.floor(stimdf.startTRreal[i]))
        duration = stimdf.duration[i]
        trend = int(np.ceil(trstart + duration))
        stim_mat[trstart:trend,col_count] = 1.0

        stim_index.append(stimdf.trial_type[i])
        col_count += 1 # go to next matrix_col

    conditions = np.unique(stimdf.trial_type.values)
    conditions = list(conditions)
    conditions.remove('Instruct') # Remove this - not a condition (and no timing information)

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

def loadTaskTimingFIR(sess, run, num_timepoints, nRegsFIR=20):
    """
    Loads task timings for each run separately
    """
    trLength = 1.0
    subj = sess[:2] # first 2 characters form the subject ID
    sess_id = sess[-2:] # last 2 characters form the session
    tasktime_dir = datadir + 'sessions/' + sess + '/bids/func/'
    stimfile = glob.glob(tasktime_dir + 'sub-' + subj + '_ses-' + sess_id + '*' + str(run) + '_events.tsv')[0]
    stimdf = pd.read_csv(stimfile,sep='\t') 
    conditions = np.unique(stimdf.trial_type.values)
    conditions = list(conditions)
    conditions.remove('Instruct') # Remove this - not a condition (and no timing information)
    # Note that the event files don't have a distinction between 0-back nd 2-back conditions for both object recognition and verbal recognition tasks
    # conditions.remove('Rest')
    tasks = np.unique(stimdf.taskName.values)

    stim_mat = np.zeros((num_timepoints,len(conditions)))
    stim_index = []

    stimcount = 0
    for cond in conditions:
        conddf = stimdf.loc[stimdf.trial_type==cond]
        for ind in conddf.index:
            trstart = int(conddf.startTRreal[ind])
            duration = conddf.duration[ind]
            trend = int(trstart + duration)
            stim_mat[trstart:trend,stimcount] = 1.0

        stim_index.append(cond)
        stimcount += 1 # go to next condition


    ## 
    if taskModel=='FIR':
        ## TODO
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)

        ## First set up FIR design matrix
        stim_index = []
        taskStims_FIR = [] 
        for stim in range(stim_mat.shape[1]):
            taskStims_FIR.append([])
            time_ind = np.where(stim_mat[:,stim]==1)[0]
            blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
            # Identify the longest block - set FIR duration to longest block
            maxRegsForBlocks = 0
            for block in blocks:
                if len(block) > maxRegsForBlocks: maxRegsForBlocks = len(block)
            taskStims_FIR[stim] = np.zeros((stim_mat.shape[0],maxRegsForBlocks+nRegsFIR)) # Task timing for this condition is TR x length of block + FIR lag
            stim_index.extend(np.repeat(stim,maxRegsForBlocks+nRegsFIR))
        stim_index = np.asarray(stim_index)

        ## Now fill in FIR design matrix
        # Make sure to cut-off FIR models for each run separately
        trcount = 0

        for run in range(nRunsPerTask):
            trstart = trcount
            trend = trstart + nTRsPerRun
                
            for stim in range(stim_mat.shape[1]):
                time_ind = np.where(stim_mat[:,stim]==1)[0]
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
        

        taskStims_FIR2 = np.zeros((stim_mat.shape[0],1))
        task_index = []
        for stim in range(stim_mat.shape[1]):
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
