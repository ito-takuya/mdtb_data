# utility and tool functions

# Taku Ito
# 2/22/21

import numpy as np
import nibabel as nib
import scipy.stats as stats
import h5py

basedir = '/gpfs/loomis/project/n3/Studies/MurrayLab/taku/multiTaskVAE/'



def loadTaskActivations(sess, run, space='vertex'):
    """
    Load task activation maps (canonical HRF)

    return: 
    data        :       Activation vector
    task_index  :       Index array with labels of all task conditions
    """

    taskdatadir = basedir  + 'derivatives/postprocessing/'
    filename = taskdatadir + sess + '_tfMRI_' + space + '_canonical_qunex_bold' + str(run)
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
