# Functions to access and load processed MDTB data

# Taku Ito
# 9/27/21

import numpy as np
import nibabel as nib
import scipy.stats as stats
import h5py
import sklearn
import sklearn.svm as svm

basedir = '/gpfs/loomis/project/n3/Studies/MurrayLab/taku/mdtb_data/'
datadir = basedir  + 'derivatives/postprocessing/'
analysistools = '/gpfs/loomis/project/n3/Studies/MurrayLab/taku/'

networkdef = np.loadtxt(analysistools + '/AnalysisTools/ColeAnticevicNetPartition/cortex_parcel_network_assignments.txt')
# need to subtract one to make it compatible for python indices
indsort = np.loadtxt(analysistools + 'AnalysisTools/ColeAnticevicNetPartition/cortex_community_order.txt',dtype=int) - 1 
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

glasserfilename = analysistools + '/AnalysisTools/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser = np.squeeze(nib.load(glasserfilename).get_fdata())


def loadTaskActivations(sess, run, space='vertex', model='canonical'):
    """
    Load task activation maps (canonical HRF)

    ## PARAMETERS
    sess
        Subject specific session, e.g., one of =['02_a1','02_a2','02_b1','02_b2','03_a1','03_a2','03_b1','03_b2','04_a1','04_a2','04_b1','04_b2','06_a1','06_a2','06_b1','06_b2','09_a1','09_a2','09_b1','09_b2','12_a1','12_a2','12_b1','12_b2','15_a1','15_a2','15_b1','15_b2','18_a1','18_a2','18_b1','18_b2','20_a1','20_a2','20_b1','20_b2','22_a1','22_a2','22_b1','22_b2','25_a1','25_a2','25_b1','25_b2','27_a1','27_a2','27_b1','27_b2','29_a1','29_a2','29_b1','29_b2','31_a1','31_a2','31_b1','31_b2','02_a1','02_a2','02_b1','02_b2','04_a1','04_a2','04_b1','04_b2','08_a1','08_a2','08_b1','08_b2','10_a1','10_a2','10_b1','10_b2','14_a1','14_a2','14_b1','14_b2','17_a1','17_a2','17_b1','17_b2','19_a1','19_a2','19_b1','19_b2','21_a1','21_a2','21_b1','21_b2','24_a1','24_a2','24_b1','24_b2','26_a1','26_a2','26_b1','26_b2','28_a1','28_a2','28_b1','28_b2','30_a1','30_a2','30_b1','30_b2']
    run
        Subject run, typically an int ranging from 1-8
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
    
    ## PARAMETERS
    subj
        Subject ID as a string. Subject list: ['02','03','04','06','08','09','10','12','14','15','18','20','22','25','27','29','31','17','19','21','24','26','28','30']
    space
        Spatial format to import data: ['parcellated', 'vertex'] # parcellated produces Glasser parcellated data; Vertex produces vertex-wise resting-state data

    ## RETURNS
    data
        space x time 2d matrix
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


def loadTaskfMRI_FIR_Timeseries(session,space='parcellated'):
    """
    Load in task-state residuals after FIR modeling on all tasks
    
    ## PARAMETERS
    session 
        Subject-specific session ID as a string. (Example, "02_a1")
    space
        Spatial format to import data: ['parcellated', 'vertex'] # parcellated produces Glasser parcellated data; Vertex produces vertex-wise resting-state data

    ## RETURNS
    data
        space x time 2d matrix
    tasktiming_labels
        Provides a 1d array that describes which task occurs during which timepoints ("None" indicates that it's either inter-trial interval or an encoding period)
    """
    h5f = h5py.File(datadir + '/fir/' + session + '_tfMRI_parcellated_TaskBlockFIR_qunex_allruns.h5','r')
    ts = h5f['residuals'][:].T
    h5f.close()

    tasktiming_labels = np.loadtxt(datadir + '/fir/' + session + '_tfMRI_parcellated_TaskBlockFIR_qunex_allruns_tasktiminglabels.csv',dtype=object)

    return ts.T, tasktiming_labels


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

