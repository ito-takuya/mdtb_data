# utility and tool functions

# Taku Ito
# 2/22/21

import numpy as np
import nibabel as nib
import scipy.stats as stats
import h5py
import sklearn
import sklearn.svm as svm

basedir = '/gpfs/loomis/project/n3/Studies/MurrayLab/taku/mdtb_data/'

networkdef = np.loadtxt('/home/ti236/AnalysisTools/ColeAnticevicNetPartition/cortex_parcel_network_assignments.txt')
# need to subtract one to make it compatible for python indices
indsort = np.loadtxt('/home/ti236/AnalysisTools/ColeAnticevicNetPartition/cortex_community_order.txt',dtype=int) - 1 
indsort.shape = (len(indsort),1)

# network mappings for final partition set
networkmappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8, 'lan':6, 'dan':5, 'con':4, 'dmn':9, 
                   'pmulti':10, 'none1':11, 'none2':12}
networks = networkmappings.keys()

xticks = {}
reorderednetworkaffil = networkdef[indsort]
for net in networks:
    netNum = networkmappings[net]
    netind = np.where(reorderednetworkaffil==netNum)[0]
    tick = np.max(netind)
    xticks[tick] = net

## General parameters/variables
nParcels = 360

sortednets = np.sort(list(xticks.keys()))
orderednetworks = []
for net in sortednets: orderednetworks.append(xticks[net])

networkpalette = ['royalblue','slateblue','paleturquoise','darkorchid','limegreen',
                  'lightseagreen','yellow','orchid','r','peru','orange','olivedrab']
networkpalette = np.asarray(networkpalette)

OrderedNetworks = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMM','VMM','ORA']

glasserfilename = '/home/ti236/AnalysisTools/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser = np.squeeze(nib.load(glasserfilename).get_data())




def loadTaskActivations(sess, run, space='vertex', model='canonical'):
    """
    Load task activation maps (canonical HRF)

    return: 
    data        :       Activation vector
    task_index  :       Index array with labels of all task conditions
    """

    taskdatadir = basedir  + 'derivatives/postprocessing/'
    filename = taskdatadir + sess + '_tfMRI_' + space + '_' + model + '_qunex_bold' + str(run)
    h5f = h5py.File(filename + '.h5','r')
    data = h5f['betas'][:].copy()
    #task_index = np.loadtxt(filename + '_taskIndex.csv')

    task_index = []
    # open file and read the content in a list
    with open(filename + '_taskIndex.csv', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            # add item to the list
            task_index.append(currentPlace)
            
    return data, task_index

def loadrsfMRI(subj,space='parcellated'):
    """
    Load in resting-state residuals
    """
    runs = ['bold9','bold10']

    data = []
    for run in runs:
        try:
            h5f = h5py.File(datadir + subj + '_b2_rsfMRI_parcellated_qunex_' + run + '.h5','r')
            ts = h5f['residuals'][:].T
            data.extend(ts)
            h5f.close()
        except:
            print('Subject', subj, '| run', run, ' does not exist... skipping')

    try:
        data = np.asarray(data).T
    except:
        print('\tError')

    return data

def computeSubjRSM(sub,space='vertex',glm='canonical', wholebrain=False,unique_tasks=None):
    """
    Computes a cross-validated RSM - cross-validated on A sessions versus B sessions (diagonals are therefore not ~ 1)
    Returns: a cross-validated, subject-specific RSM with corresponding index (ordering)
    """
    runs = range(1,9) # These are the run numbers for task data
    sess1 = ['a1', 'b1']
    sess2 = ['a2', 'b2']
    rsm_1 = []
    rsm_2 = []

    # Load in data for session a
    data_1 = []
    task_index_1 = []
    for sess in sess1:
        sess_id = sub + '_' + sess
        for run in runs:
            tmpdat, tmpind = loadTaskActivations(sess_id,run,space=space,model=glm)
            data_1.extend(tmpdat.T)
            task_index_1.extend(tmpind)
    data_1 = np.asarray(data_1)
    task_index_1 = np.asarray(task_index_1)

    # Load in data for session b
    data_2 = []
    task_index_2 = []
    for sess in sess2:
        sess_id = sub + '_' + sess
        for run in runs:
            tmpdat, tmpind = loadTaskActivations(sess_id,run,space=space)
            data_2.extend(tmpdat.T)
            task_index_2.extend(tmpind)
    data_2 = np.asarray(data_2)
    task_index_2 = np.asarray(task_index_2)
    
    if unique_tasks is None:
        unique_tasks = np.unique(task_index_1)
    # Ensure a and b have the same number of unique tasks
    if len(np.unique(task_index_1)) != len(np.unique(task_index_2)):
        raise Exception("Wait! Sessions 1 and 2 don't have the same number of tasks... Cannot generate cross-validated RSMs")
    n_tasks = len(unique_tasks)


    data_task_1 = []
    data_task_2 = []
    for task in unique_tasks:
        task_ind = np.where(task_index_1==task)[0]
        #data_task_1.append(stats.ttest_1samp(data_1[task_ind,:],0,axis=0)[0])
        data_task_1.append(np.mean(data_1[task_ind,:],axis=0))

        task_ind = np.where(task_index_2==task)[0]
        #data_task_2.append(stats.ttest_1samp(data_2[task_ind,:],0,axis=0)[0])
        data_task_2.append(np.mean(data_2[task_ind,:],axis=0))


    data_task_1 = np.asarray(data_task_1).T
    data_task_2 = np.asarray(data_task_2).T

    if space=='vertex':
        # Compute whole-brain RSM
        if wholebrain:
            tmpmat = np.zeros((n_tasks,n_tasks))
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i>j: continue
                    tmpmat[i,j] = stats.pearsonr(data_task_1[:,i],data_task_2[:,j])[0]

            # Now make symmetric
            tmpmat = tmpmat + tmpmat.T
            # double counting diagonal so divide by 2
            np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
            rsms = tmpmat
        else:

            # compute RSM for each parcel
            rsms = []
            for roi in range(1,361):
                roi_ind = np.where(glasser==roi)[0]
                roidat1 = data_task_1[roi_ind,:].T
                roidat2 = data_task_2[roi_ind,:].T
                tmpmat = np.zeros((n_tasks,n_tasks))
                for i in range(n_tasks):
                    for j in range(n_tasks):
                        if i>j: continue
                        tmpmat[i,j] = stats.pearsonr(roidat1[i,:],roidat2[j,:])[0]
                        #tmpmat[i,j] = np.mean(np.multiply(roidat1[i,:],roidat2[j,:]))

                # Now make symmetric
                tmpmat = tmpmat + tmpmat.T
                # double counting diagonal so divide by 2
                np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
                rsms.append(tmpmat)
            rsms = np.asarray(rsms)

    if space=='parcellated':
        # Compute whole-brain RSM
        if wholebrain:
            tmpmat = np.zeros((n_tasks,n_tasks))
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i>j: continue
                    tmpmat[i,j] = stats.pearsonr(data_task_1[:,i],data_task_2[:,j])[0]

            # Now make symmetric
            tmpmat = tmpmat + tmpmat.T
            # double counting diagonal so divide by 2
            np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
            rsms = tmpmat
        else:
            # compute rsm for each network
            rsms = {}
            for net in orderednetworks:
                net_ind = np.where(networkdef==networkmappings[net])[0]
                netdat1 = data_task_1[net_ind,:].T
                netdat2 = data_task_2[net_ind,:].T
                tmpmat = np.zeros((n_tasks,n_tasks))
                for i in range(n_tasks):
                    for j in range(n_tasks):
                        if i>j: continue
                        tmpmat[i,j] = stats.pearsonr(netdat1[i,:],netdat2[j,:])[0]
                        #tmpmat[i,j] = np.mean(np.multiply(netdat1[i,:],netdat2[j,:]))

                # Now make symmetric
                tmpmat = tmpmat + tmpmat.T
                # double counting diagonal so divide by 2
                np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
                rsms[net] = tmpmat

    return rsms, unique_tasks

    
def decoding(trainset,testset,trainlabels,testlabels,classifier='distance',confusion=False):
    unique_labels = np.unique(trainlabels)
    
    if classifier == 'distance':
        #### Create prototypes from trainset
        prototypes = {}
        for label in unique_labels:
            ind = np.where(trainlabels==label)[0]
            prototypes[label] = np.mean(trainset[ind,:],axis=0)

        #### Now classifiy each sample n the testset
        predictions = []
        for i in range(testset.shape[0]):
            # Correlate sampple with each prototype
            rs = []
            for label in prototypes:
                rs.append(stats.pearsonr(prototypes[label],testset[i,:])[0])
            
            # Find the closest prototype for sample
            max_ind = np.argmax(np.asarray(rs))
            predictions.append(unique_labels[max_ind])

        predictions = np.asarray(predictions)

    if classifier == 'logistic':

        #clf = sklearn.linear_model.LogisticRegression(solver='lbfgs',penalty='none',max_iter=1000)
        clf = sklearn.linear_model.LogisticRegression(solver='liblinear')
        clf.fit(trainset,trainlabels)
        predictions = clf.predict(testset)

    if classifier == 'ridge':

        clf = sklearn.linear_model.RidgeClassifier(solver='svd',max_iter=1000)
        clf.fit(trainset,trainlabels)
        predictions = clf.predict(testset)

    if classifier == 'svm':
        clf = svm.SVC(kernel='linear',probability=True)
        clf.fit(trainset,trainlabels)
        predictions = clf.predict(testset)

    accuracy = predictions == np.asarray(testlabels)
    #confusion_mat = sklearn.metrics.confusion_matrix(testlabels, predictions, labels=unique_labels)

    if classifier in ['svm','logistic']:
        decision_function = clf.predict_log_proba(testset)
        conditions = clf.classes_ 
        decision_on_test = []
        for i in range(len(testlabels)):
            ind = np.where(conditions==testlabels[i])[0]
            decision_on_test.append(decision_function[i,ind])
        return accuracy, decision_on_test 
    elif classifier in ['ridge']:
        decision_function = clf.decision_function(testset)
        conditions = clf.classes_ 
        decision_on_test = []
        for i in range(len(testlabels)):
            ind = np.where(conditions==testlabels[i])[0]
            decision_on_test.append(decision_function[i,ind])
        return accuracy, decision_on_test 
    else:
        return accuracy

def getDimensionality(data):
    """
    data needs to be a square, symmetric matrix
    """
    corrmat = data
    eigenvalues, eigenvectors = np.linalg.eig(corrmat)
    dimensionality_nom = 0
    dimensionality_denom = 0
    for eig in eigenvalues:
        dimensionality_nom += np.real(eig)
        dimensionality_denom += np.real(eig)**2

    dimensionality = dimensionality_nom**2/dimensionality_denom

    return dimensionality
