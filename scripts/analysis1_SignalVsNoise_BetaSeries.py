import numpy as np
import nibabel as nib
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import tools
import pandas as pd
import matplotlib.image as img 
import statsmodels.sandbox.stats.multicomp as mc
import argparse
import os

basedir = '/home/ti236/taku/multiTaskVAE/'
outdir = basedir + 'derivatives/results/analysis1/'
if not os.path.exists(outdir): os.makedirs(outdir)

networkdef = np.loadtxt('/home/ti236/AnalysisTools/ColeAnticevicNetPartition/cortex_parcel_network_assignments.txt')
# need to subtract one to make it compatible for python indices
indsort = np.loadtxt('/home/ti236/AnalysisTools/ColeAnticevicNetPartition/cortex_community_order.txt',dtype=int) - 1 
indsort.shape = (len(indsort),1)
# network mappings for final partition set
networkmappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8, 'lan':6, 'dan':5, 'con':4, 'dmn':9, 
                   'pmulti':10, 'none1':11, 'none2':12}
networks = networkmappings.keys()

## General parameters/variables
nParcels = 360
glasserfilename = '/home/ti236/AnalysisTools/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser = np.squeeze(nib.load(glasserfilename).get_fdata())
 
# Set task ordering
unique_tasks = ['NoGo','Go','TheoryOfMind','VideoActions','VideoKnots','Math',
                'DigitJudgement','Objects','MotorImagery','FingerSimple','FingerSeq',
                'Verbal2Back','SpatialImagery','VerbGen','WordRead','Rest',
                'PermutedRules','SpatialMapEasy','SpatialMapMed','SpatialMapHard',
                'NatureMovie','AnimatedMovie','LandscapeMovie','UnpleasantScenes','PleasantScenes',
                'SadFaces','HappyFaces','Object2Back','IntervalTiming',
                'Prediction','PredictViol','PredictScram','VisualSearchEasy','VisualSearchMed','VisualSearchHard',
                'StroopIncon','StroopCon','MentalRotEasy','MentalRotMed','MentalRotHard',
                'BiologicalMotion','ScrambledMotion','RespAltEasy','RespAltMed','RespAltHard']

task_passivity = ['left','left','left','passive','passive','right',
                  'right','passive','passive','both','both',
                  'left','passive','passive','passive','passive',
                  'both','both','both','both',
                  'passive','passive','passive','left','left',
                  'right','right','right','right',
                  'left','left','left','left','left','left',
                  'both','both','right','right','right',
                  'right','right','both','both','both']

# sort tasks by passivity
unique_tasks = np.asarray(unique_tasks)
task_passivity = np.asarray(task_passivity)
unique_tasks2 = []
passivity_order = ['passive','left','right','both']
for i in passivity_order:
    ind = np.where(task_passivity==i)[0]
    unique_tasks2.extend(unique_tasks[ind])
unique_tasks = np.asarray(unique_tasks2)


subIDs=['02','03','04','06','08','09','10','12','14','15','18','20','22','25','27','29','31','17','19','21','24','26','28','30']
sessIDs = ['a1','a2','b1','b2']
runs = range(1,9)

parser = argparse.ArgumentParser('./main.py', description='Run a set of simulations/models')
parser.add_argument('--outfilename', type=str, default="analysis1", help='Prefix output filenames (Default: analysis1')

def run(args):
    args 
    outfilename = args.outfilename

    #### Set subject parameters
    triu_ind = np.triu_indices(nParcels,k=1)


    #### Load in data
    print('LOADING DATA')
    data, task_index = loadData(subIDs,sessIDs,runs)
    conditions = np.unique(task_index[subIDs[0]])

    #### Measure signal correlations
    print('MEASURING SIGNAL CORRELATIONS')
    signal_corr = measureSignalCorr(data,task_index)
    signal_corr = np.arctanh(signal_corr)
    avg_sigcorr = np.mean(signal_corr,axis=2)
    np.savetxt(outdir + outfilename + '_signalcorr_groupaverage.txt',np.mean(signal_corr,axis=2),delimiter=',')

    #### Measure noise correlations
    print('MEASURING NOISE CORRELATIONS')
    noise_corr = measureNoiseCorr(data,task_index)
    tmp = np.mean(np.mean(noise_corr,axis=2),axis=2)
    np.savetxt(outdir + outfilename + '_noisecorr_groupconditionaverage.txt',tmp,delimiter=',')
    rest_ind = np.where(conditions=='Rest')[0][0]
    tmp = np.mean(noise_corr[:,:,rest_ind,:],axis=2)
    avg_noisecorr = np.mean(noise_corr,axis=3)
    np.savetxt(outdir + outfilename + '_noisecorr_rest_groupaverage.txt',tmp,delimiter=',')
    
    #### Count number of trials per condition
    print("COUNT NUMBER OF TRIALS PER CONDITION")
    df_numtrialspercond = countNumberTrials(data,task_index)
    df_numtrialspercond.to_csv(outdir + outfilename + '_numTrialsPerCond.csv')
            
    #### Compute average correlation change from rest to condition (Ito et al. 2020 replication)
    print("COMPUTE AVERAGE CORRELATION CHANGE FROM REST TO TASK CONDITION (ITO ET AL. 2020 REPLICATION)")
    df_fcchange = measureRestTaskFCChange(avg_noisecorr,task_index)
    df_fcchange.to_csv(outdir + outfilename + '_AvgCorrelationChangeRestToTask.csv')


    #### Compute signal vs noise correlations (categorized by signal correlation signs) 
    print("COMPUTE SIGNAL VS NOISE CORRELATION CHANGES AS A FUNCTION OF SIGNAL CORR")
    df_sig = computeSignalNoiseCorrContrast(avg_sigcorr, avg_noisecorr, task_index, control='Signal')
    df_sig.to_csv(outdir + outfilename + '_SignalNoiseCorrDifferences_BasedOnPosNegSignalCorr.csv')


    #### Compute signal vs noise correlations (categorized by noise correlation signs) 
    print("COMPUTE SIGNAL VS NOISE CORRELATION CHANGES AS A FUNCTION OF NOISE CORR")
    df_noise = computeSignalNoiseCorrContrast(avg_sigcorr, avg_noisecorr, task_index, control='Noise')
    df_noise.to_csv(outdir + outfilename + '_SignalNoiseCorrDifferences_NoiseCorrChanges.csv')


def loadData(subIDs,sessIDs,runs):
    data = {}
    task_index = {}
    for sub in subIDs:
        data[sub] = []
        task_index[sub] = []
        for sess in sessIDs:
            session = sub + '_' + sess
            for run in runs:
                tmpdat, tmpind = tools.loadTaskActivations(session,run,space='parcellated',model='betaseries')
                data[sub].extend(tmpdat.T)
                task_index[sub].extend(tmpind)

        data[sub] = np.asarray(data[sub])
        task_index[sub] = np.asarray(task_index[sub])
    return data, task_index

def measureSignalCorr(data,task_index):
    triu_ind = np.triu_indices(nParcels,k=1)
    conditions = np.unique(task_index[subIDs[0]])
    signal_corr = np.zeros((nParcels,nParcels,len(subIDs)))
    i = 0
    for sub in data:
        parcel_signals = []
        j = 0
        for cond in conditions:
            tmp_taskind = np.where(task_index[sub]==cond)[0]
            parcel_signals.append(np.mean(data[sub][tmp_taskind,:],axis=0))        
            j += 1
        # Compute the 'average' of signals across task conditions for each parcel
        parcel_signals = np.asarray(parcel_signals).T
        # Compute signal correlation
        tmpmat = np.corrcoef(parcel_signals)
        # Compute t-values and p-values of correlations
        tmat = tmpmat*np.sqrt((parcel_signals.shape[1]-2)/(1-tmpmat**2))
        pval = stats.t.sf(np.abs(tmat), parcel_signals.shape[1]-2)*2
        h0,qs = mc.fdrcorrection0(pval[triu_ind])
        thresh_mat = np.zeros((nParcels,nParcels))
        thresh_mat[triu_ind] = np.multiply(tmpmat[triu_ind],h0)
        thresh_mat = thresh_mat + thresh_mat.T
        # Compute signal correlations, i.e., 'brain co-activations'
    #     signal_corr[:,:,i] = np.corrcoef(parcel_signals)
        signal_corr[:,:,i] = tmpmat
        np.fill_diagonal(signal_corr[:,:,i],0)

        i += 1

    return signal_corr

def measureNoiseCorr(data,task_index):
    conditions = np.unique(task_index[subIDs[0]])
    noise_corr = np.zeros((nParcels,nParcels,len(conditions),len(subIDs)))
    i = 0
    for sub in data:
        parcel_signals = []
        j = 0
        for cond in conditions:
            tmp_taskind = np.where(task_index[sub]==cond)[0]
            tmpmat = np.corrcoef(data[sub][tmp_taskind,:].T)
    #         tmat = tmpmat*np.sqrt((len(tmp_taskind)-2)/(1-tmpmat**2))
    #         pval = stats.t.sf(np.abs(tmat), len(tmp_taskind)-2)*2
    #         # Compute threshold
    #         h0,qs = mc.fdrcorrection0(pval[triu_ind])
    #         thresh_mat = np.zeros((nParcels,nParcels))
    #         thresh_mat[triu_ind] = np.multiply(tmpmat[triu_ind],h0)
    #         thresh_mat = thresh_mat + thresh_mat.T
            noise_corr[:,:,j,i] = tmpmat
            np.fill_diagonal(noise_corr[:,:,j,i],0)
            noise_corr[:,:,j,i] = np.arctanh(noise_corr[:,:,j,i])
            j += 1
        i += 1
    
    return noise_corr

def measureRestTaskFCChange(noisecorr_mat,task_index):
    conditions = np.unique(task_index[subIDs[0]])
    tmpdf = {}
    tmpdf['Condition'] = []
    tmpdf['CorrelationChange'] = []
    rest_ind = np.where(conditions=='Rest')[0][0]
    for i in range(len(conditions)):
        if conditions[i]=='Rest': continue
        tmpdf['CorrelationChange'].append(np.nanmean(noisecorr_mat[:,:,i] - noisecorr_mat[:,:,rest_ind]))
        tmpdf['Condition'].append(conditions[i])
    tmpdf = pd.DataFrame(tmpdf)
    return tmpdf

def countNumberTrials(data,task_index):
    conditions = np.unique(task_index[subIDs[0]])
    df_numtrialspercond = {}
    df_numtrialspercond['Condition'] = []
    df_numtrialspercond['Number of Trials'] = []
    df_numtrialspercond['Subject'] = []
    for cond in conditions:
        for sub in task_index:
            df_numtrialspercond['Number of Trials'].append(np.sum(task_index[sub]==cond))
            df_numtrialspercond['Condition'].append(cond)
            df_numtrialspercond['Subject'].append(sub)
    
    df_numtrialspercond = pd.DataFrame(df_numtrialspercond)
    return df_numtrialspercond

def computeSignalNoiseCorrContrast(sigcorr,noisecorr,task_index,control='Signal'):
    triu_ind = np.triu_indices(nParcels,k=1)
    conditions = np.unique(task_index[subIDs[0]])
    df_signoise = {}
    df_signoise['Correlation'] = []
    df_signoise['Percent'] = []
    df_signoise['Condition'] = []
    df_signoise['Type'] = []
    rest_ind = np.where(conditions=='Rest')[0][0]
    for i in range(len(conditions)):
        if conditions[i] == 'Rest': continue
        other_cond = np.where(np.arange(len(conditions))!=i)[0]

        noise_corr_diff = noisecorr[:,:,i] - noisecorr[:,:,rest_ind]
        tmp = np.mean(np.multiply(noise_corr_diff[triu_ind],sigcorr[triu_ind])<0)
        r, p = stats.spearmanr(noise_corr_diff[triu_ind],sigcorr[triu_ind])
        df_signoise['Type'].append('All')
        df_signoise['Percent'].append(tmp)
        df_signoise['Condition'].append(conditions[i])
        df_signoise['Correlation'].append(r)

        if control=='Signal': 
            pos_ind = sigcorr[triu_ind]>0
        elif control=='Noise':
            pos_ind = noise_corr_diff[triu_ind]>0
        tmp = np.mean(np.multiply(noise_corr_diff[triu_ind][pos_ind],sigcorr[triu_ind][pos_ind])<0)
        r, p = stats.spearmanr(noise_corr_diff[triu_ind][pos_ind],sigcorr[triu_ind][pos_ind])
        df_signoise['Type'].append('Positive' + control + 'Corr')
        df_signoise['Percent'].append(tmp)
        df_signoise['Condition'].append(conditions[i])
        df_signoise['Correlation'].append(r)


        if control=='Signal':
            neg_ind = sigcorr[triu_ind]<0
        elif control=='Noise':
            neg_ind = noise_corr_diff[triu_ind]<0
        tmp = np.mean(np.multiply(noise_corr_diff[triu_ind][neg_ind],sigcorr[triu_ind][neg_ind])<0)
        r, p = stats.spearmanr(noise_corr_diff[triu_ind][neg_ind],sigcorr[triu_ind][neg_ind])
        df_signoise['Type'].append('Negative' + control + 'Corr')
        df_signoise['Percent'].append(tmp)
        df_signoise['Condition'].append(conditions[i])
        df_signoise['Correlation'].append(r)

    df_signoise = pd.DataFrame(df_signoise)
    return df_signoise


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
