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
from sklearn.model_selection import StratifiedKFold
import multiprocessing as mp

basedir = '/home/ti236/taku/multiTaskVAE/'
outdir = basedir + 'derivatives/results/analysis2/'
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
parser.add_argument('--outfilename', type=str, default="analysis2", help='Prefix output filenames (Default: analysis2')
parser.add_argument('--nfeatures', type=int, default=20, help='Number of regions/features to use per decoding (Default: 20')

def run(args):
    args 
    outfilename = args.outfilename
    nfeatures = args.nfeatures
    outfilename = outfilename + '_' + str(nfeatures) + 'nfeatures'
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
    #np.savetxt(outdir + outfilename + '_signalcorr_groupaverage.txt',np.mean(signal_corr,axis=2),delimiter=',')

    #### Measure noise correlations
    print('MEASURING NOISE CORRELATIONS')
    noise_corr = measureNoiseCorr(data,task_index)
    tmp = np.mean(np.mean(noise_corr,axis=2),axis=2)
    #np.savetxt(outdir + outfilename + '_noisecorr_groupconditionaverage.txt',tmp,delimiter=',')
    rest_ind = np.where(conditions=='Rest')[0][0]
    tmp = np.mean(noise_corr[:,:,rest_ind,:],axis=2)
    avg_noisecorr = np.mean(noise_corr,axis=3)
    #np.savetxt(outdir + outfilename + '_noisecorr_rest_groupaverage.txt',tmp,delimiter=',')
    
    #### identify sets of regions with different 'coding' signatures
    print("IDENTIFY REGIONS WITH DIFFERENT CODING SIGNATURES")
    sigpos_noisepos, signeg_noisepos, sigpos_noiseneg, signeg_noiseneg = findSignalNoiseRegionSets(avg_sigcorr,avg_noisecorr,task_index)
    # Select regions for each decoding type
#    sigpos_noisepos = np.random.choice(sigpos_noisepos,nfeatures,replace=False) # not replacement; don't want redundant features
#    signeg_noisepos = np.random.choice(signeg_noisepos,nfeatures,replace=False) # not replacement; don't want redundant features
#    sigpos_noiseneg = np.random.choice(sigpos_noiseneg,nfeatures,replace=False) # not replacement; don't want redundant features
#    signeg_noiseneg = np.random.choice(signeg_noiseneg,nfeatures,replace=False) # not replacement; don't want redundant features
            
    #### Compute average correlation change from rest to condition (Ito et al. 2020 replication)
    print("RUN DECODING ON SIGNAL POS–NOISE POS REGIONS")
    nbootstraps = 20
    ncvs = 10
    tmpdf = {}
    tmpdf['Subject'] = []
    tmpdf['CodingType'] = []
    tmpdf['Accuracy'] = []
    inputs = []
    #for sub in subIDs[:2]:
    for sub in data:
        print('Decoding subject', sub)
        accuracy = runDecodingOnRegionSets(data[sub],task_index[sub],sigpos_noisepos,nfeatures=nfeatures,nbootstraps=20,ncvs=10)
        tmpdf['Subject'].append(sub)
        tmpdf['CodingType'].append('SignalPositive–NoisePositive')
        tmpdf['Accuracy'].append(np.mean(accuracy))
        #
        accuracy = runDecodingOnRegionSets(data[sub],task_index[sub],signeg_noisepos,nfeatures=nfeatures,nbootstraps=20,ncvs=10)
        tmpdf['Subject'].append(sub)
        tmpdf['CodingType'].append('SignalNegative–NoisePositive')
        tmpdf['Accuracy'].append(np.mean(accuracy))
        #
        accuracy = runDecodingOnRegionSets(data[sub],task_index[sub],sigpos_noiseneg,nfeatures=nfeatures,nbootstraps=20,ncvs=10)
        tmpdf['Subject'].append(sub)
        tmpdf['CodingType'].append('SignalPositive–NoiseNegative')
        tmpdf['Accuracy'].append(np.mean(accuracy))
        #
        accuracy = runDecodingOnRegionSets(data[sub],task_index[sub],signeg_noiseneg,nfeatures=nfeatures,nbootstraps=20,ncvs=10)
        tmpdf['Subject'].append(sub)
        tmpdf['CodingType'].append('SignalNegative–NoiseNegative')
        tmpdf['Accuracy'].append(np.mean(accuracy))
    tmpdf = pd.DataFrame(tmpdf)
    tmpdf.to_csv(outdir + outfilename + '_DecodingAccuraciesByCodingType.csv')

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

def measureSignalCorr(data,task_index,nbootstrap=50):
    triu_ind = np.triu_indices(nParcels,k=1)
    conditions = np.unique(task_index[subIDs[0]])
    signal_corr = np.zeros((nParcels,nParcels,len(subIDs)))
    i = 0
    for sub in data:
        parcel_signals = []
        j = 0
        for cond in conditions:
            tmp_taskind = np.where(task_index[sub]==cond)[0]
            bootstrap_ind = np.random.choice(tmp_taskind,size=nbootstrap,replace=True)
            parcel_signals.append(np.mean(data[sub][bootstrap_ind,:],axis=0))        
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

def measureNoiseCorr(data,task_index,nbootstrap=50):
    conditions = np.unique(task_index[subIDs[0]])
    noise_corr = np.zeros((nParcels,nParcels,len(conditions),len(subIDs)))
    i = 0
    for sub in data:
        parcel_signals = []
        j = 0
        for cond in conditions:
            tmp_taskind = np.where(task_index[sub]==cond)[0]
            bootstrap_ind = np.random.choice(tmp_taskind,size=nbootstrap,replace=True)
            tmpmat = np.corrcoef(data[sub][bootstrap_ind,:].T)
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

def findSignalNoiseRegionSets(sigcorr,noisecorr,task_index):
    conditions = np.unique(task_index[subIDs[0]])
    rest_ind = np.where(conditions=='Rest')[0][0]
    other_ind = np.where(conditions!='Rest')[0]

    noisecorr_diff = np.mean(noisecorr[:,:,other_ind],axis=2) - noisecorr[:,:,rest_ind]

    #noisecorr_pos = np.multiply(noisecorr_diff,noisecorr_diff>0)
    #noisecorr_neg = np.multiply(noisecorr_diff,noisecorr_diff<0)

    #sigcorr_pos = np.multiply(sigcorr,sigcorr>0)
    #sigcorr_neg = np.multiply(sigcorr,sigcorr<0)

    sigpos_noisepos = np.multiply(sigcorr>0,noisecorr_diff>0)
    signeg_noisepos = np.multiply(sigcorr<0,noisecorr_diff>0)
    sigpos_noiseneg = np.multiply(sigcorr>0,noisecorr_diff<0)
    signeg_noiseneg = np.multiply(sigcorr<0,noisecorr_diff<0)

    # Extract region sets
    sigpos_noisepos = np.where(sigpos_noisepos)[0]
    signeg_noisepos = np.where(signeg_noisepos)[0]
    sigpos_noiseneg = np.where(sigpos_noiseneg)[0]
    signeg_noiseneg = np.where(signeg_noiseneg)[0]

    return sigpos_noisepos, signeg_noisepos, sigpos_noiseneg, signeg_noiseneg

def runDecodingOnRegionSets(data,task_index,region_sets,nfeatures=20,nbootstraps=20,ncvs=10):
    """
    data
        conditions x regions
    task_index
        array of conditions
    region_sets
        array of ROIs
    ncvs
        number of cross validation folds
    """
    # Remove 'rest conditions'
    ind = np.where(task_index!='Rest')[0]
    task_index = task_index[ind]
    conditions = np.unique(task_index)
    
    # new 
    region_sets = np.random.choice(region_sets,nfeatures,replace=False) # not replacement; don't want redundant features
    
    #regions = np.random.choice(region_sets,nregions,replace=False) # not replacement; don't want redundant features
    data_mat = np.squeeze(data[ind][:,region_sets]).copy()
    # Get cross-validation folds
    accuracy = []
    skf = StratifiedKFold(n_splits=ncvs)
    training_folds = []
    testing_folds = []
    for train_index, test_index in skf.split(data_mat, task_index):
        tmp_new_train_ind = []
        tmp_new_test_ind = []
        for cond in conditions: 
            # Find samples for this condition in the trainset, and subsample nbootstrapped number of new samples for train set)
            train_cond_ind = np.where(task_index[train_index]==cond)[0]
            tmp_new_train_ind.extend(np.random.choice(train_index[train_cond_ind],nbootstraps,replace=True))
            # Find samples for this condition in the testset, and subsample nbootstrapped number of new samples for test set)
            test_cond_ind = np.where(task_index[test_index]==cond)[0]
            tmp_new_test_ind.extend(np.random.choice(test_index[test_cond_ind],nbootstraps,replace=True))

        training_folds.append(tmp_new_train_ind)
        testing_folds.append(tmp_new_test_ind)

    #for train_index, test_index in [training_folds, testing_folds]:
    for i in range(ncvs):
        train_index, test_index = training_folds[i], testing_folds[i]
        X_train, X_test = data_mat[train_index], data_mat[test_index]
        Y_train, Y_test = task_index[train_index], task_index[test_index]

        #Feature-wise normalization using trainset mean and std
        train_mean = np.mean(X_train,axis=0)
        train_std = np.std(X_train,axis=0)
        train_mean.shape = (1,len(train_mean))
        train_std.shape = (1,len(train_std))
        X_train = np.divide((X_train - train_mean),train_std)
        X_test = np.divide((X_test - train_mean),train_std)

        #print('fold')
        #for cond in np.unique(Y_train):
        #    print('\t',cond, np.sum(Y_train==cond), np.sum(Y_test==cond))
        accuracy.extend(tools.decoding(X_train,X_test,Y_train,Y_test,classifier='ridge',confusion=False))
    return accuracy



if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
