import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch import zeros, ones
from os import listdir, path
from scipy import io

def errorplot(*args, **kwargs):
    """plot asymetric errorbars"""
    subjects = args[0]
    values = args[1].values
    
    unique_subjects = np.unique(subjects)
    nsub = len(unique_subjects)
    
    values = values.reshape(-1, nsub)
    
    quantiles = np.percentile(values, [5, 50, 95], axis=0)
    
    low_perc = quantiles[0]
    up_perc = quantiles[-1]
    
    x = unique_subjects
    y = quantiles[1]

    assert np.all(low_perc <= y)
    assert np.all(y <= up_perc)
    
    kwargs['yerr'] = [y-low_perc, up_perc-y]
    kwargs['linestyle'] = ''
    kwargs['marker'] = 'o'
    
    plt.errorbar(x, y, **kwargs)
    
def map_noise_to_values(strings):
    """mapping strings ('high', 'low') to numbers 0, 1"""
    for s in strings:
        if s[0] == 'high':
            yield 1
        elif s[0] == 'low':
            yield 0
        else:
            yield np.nan

def load_and_format_SAT_data(datadir, ids_to_exclude = []):
    '''
    Parameters
    ----------
    datadir : string
        Full os-compatible path to directory containing SAT result files.
        Filename format: *_ID*.mat, i.e. subject id must follow first underscore.
    ids_to_exclude: list or array of strings

    Returns
    -------
    stimuli:    dict(3) of numpy.ndarrays for task conditions, -configs, -states
    responses:  numpy.ndarray(sbj,mb,trial) of responses 
    mask:       numpy.ndarray(sbj,mb,trial) of non-NaN value indicators for responses
    rts:        numpy.ndarray(sbj,mb,trial) of reaction times
    scores:     numpy.ndarray(sbj,mb,trial) of point scores
    conditions: numpy.ndarray(noise/steps,sbj,mb) of task conditions
    ids:        list of ids
    sbj_df:     pandas.DataFrame rows:sbj, cols:ID,age,gender,group
    '''

    fnames = listdir(datadir)
    fnames.sort()
    
    runs = len(fnames) - len(ids_to_exclude)  # number of subjects assuming datasets to be excluded are in datadir
    
    mini_blocks = 100  # number of mini blocks in each run
    max_trials = 3  # maximal number of trials within a mini block
    max_depth = 3  # maximal planning depth
    no = 5 # number of outcomes/rewards
    max_points = 2009 # maximum possible total points (out of 100.000 depth3-agent simulations)

    responses = zeros(runs, mini_blocks, max_trials)
    states = zeros(runs, mini_blocks, max_trials+1, dtype=torch.long)
    scores = zeros(runs, mini_blocks, max_depth)
    conditions = zeros(2, runs, mini_blocks, dtype=torch.long)
    confs = zeros(runs, mini_blocks, 6, dtype=torch.long)
    rts = zeros(runs, mini_blocks, 3, dtype=torch.float64)
    ids = []
    age = []
    gender = []
    total_points = []
    performance = []
    
    ne = 0  # exclusion counter
    
    for i,f in enumerate(fnames):
        
        tmp = io.loadmat(path.join(datadir,f))
        
        parts = f.split('_')
        if parts[0] in ids_to_exclude:
            ne += 1
        else:
            ids.append(parts[0])
            age.append(tmp['data']['Age'][0,0][0,0])
            gender.append(tmp['data']['Gender'][0,0][0,0])
    
            responses[i - ne] = torch.from_numpy(tmp['data']['Responses'][0,0]['Keys'][0,0]-1)
            states[i - ne] = torch.from_numpy(tmp['data']['States'][0,0] - 1).long()
            confs[i - ne] = torch.from_numpy(tmp['data']['PlanetConf'][0,0] - 1).long()
            scores[i - ne] = torch.from_numpy(tmp['data']['Points'][0,0])
            strings = tmp['data']['Conditions'][0,0]['noise'][0,0][0]
            conditions[0, i - ne] = torch.tensor(list(map_noise_to_values(strings)), dtype=torch.long)
            conditions[1, i - ne] = torch.from_numpy(tmp['data']['Conditions'][0,0]['notrials'][0,0][:,0]).long()
            rts[i - ne] = torch.from_numpy(tmp['data']['Responses'][0,0]['RT'][0,0])
            
            if conditions[1,i - ne,-1]==3:
                # score correction for normalOrder participants having started with 990 instead of 1000 points
                scores[i - ne] = scores[i - ne] + 10
                total_points.append(scores[i - ne,-1,-1].item())
            else:
                total_points.append(scores[i - ne,-1,-2].item())
            performance.append(total_points[-1] / max_points * 100)
            # TODO: add phase-wise SAT_PER and SAT_Total_points
        
    sbj_df = pd.DataFrame(list(zip(ids,age,gender,total_points,performance)),
                          columns=['ID','age','gender','total_points','performance'])

    states[states < 0] = -1
    confs = torch.eye(no)[confs]

    # define dictionary containing information which participants recieved on each trial
    stimuli = {'conditions': conditions,
               'states': states, 
               'configs': confs}

    mask = ~torch.isnan(responses)
    
    return stimuli, conditions, responses, mask, rts, scores, ids, sbj_df


def load_and_format_IDP_or_SAW_data(datadir,taskname):
    
    t = taskname
    ids = []
    df_list = []
    
    fnames = listdir(datadir)
    fnames.sort()

    for i,f in enumerate(fnames):
        parts = f.split('_')
        ids.append(parts[0])
        
        tmp = pd.read_csv(path.join(datadir,f), sep='\t', dtype={'VPID':object})
        maxtrials = 46 if t=='IDP' else 35 if t=='SAW' else print("WARNING: unknown value for taskname!")
        if len(tmp) < maxtrials: print("WARNING: Trials missing in file of ID "+str(ids[-1]))
        
        # delete rows after deadline was exceeded ('RESP' = 0)
        tmp = tmp.loc[tmp['RESP']>0]
        # exclude RTs < 150ms
        tmp = tmp.loc[tmp['RESPT']>=150]
        
        ntrials = len(tmp) # n of finished trials
        ncorr = len(tmp.loc[tmp['ACC']>0]) # n of correct trials  
        
        # create row for subject-wise dataframe
        entry = pd.DataFrame()
        entry['ID']         = [tmp['VPID'].iat[0] ]
        entry[t+'_CORR']    = [ncorr ]
        entry[t+'_MaxCORR'] = [maxtrials ]
        entry[t+'_PER']     = [ncorr / maxtrials * 100 ]
        entry[t+'_ERR']     = [ntrials - ncorr ]
        entry[t+'_ACC']     = [ncorr / ntrials * 100 ]
        entry[t+'_RT']      = [tmp['RESPT'].mean() / 1000 ]
        entry[t+'_RT_SD']   = [tmp['RESPT'].std()  / 1000 ]
        
        df_list.append(entry)
    
    return pd.concat(df_list, ignore_index=True)


def load_and_format_SWM_data(datadir):
    
    # function only focuses on the first task condition (location memory condition) across both load levels (4 and 7 items)
    
    ids = []
    df_list = []
    
    fnames = listdir(datadir)
    fnames.sort()

    for i,f in enumerate(fnames):
        parts = f.split('_')
        ids.append(parts[0])
        
        tmp = pd.read_csv(path.join(datadir,f), sep='\t', dtype={'VPID':object}, index_col=False) # index_col=False to prevent confusion because files contain delimiters at end of lines
        maxtrials = 96
        if len(tmp) < maxtrials: print("WARNING: Trials missing in file of ID "+str(ids[-1]))
        
        # drop first 4 rows which contain the data of the 4 training trials during instruction
        tmp.drop(tmp.index[:4], inplace=True)
        # delete rows after deadline was exceeded ('Resp1' = -999)
        tmp = tmp.loc[tmp['Resp1']>=0]
        # exclude RTs < 150ms
        tmp = tmp.loc[tmp['RT1']>=150]
        
        ncorr = len(tmp.loc[tmp['Corr1']>0]) # n of correct trials
        ntrials = len(tmp) # n of finished trials
        
        # create row for subject-wise dataframe
        entry = pd.DataFrame()
        entry['ID']          = [tmp['VPID'].iat[0] ]
        entry['SWM_CORR']    = [ncorr ]
        entry['SWM_MaxCORR'] = [maxtrials ]
        entry['SWM_PER']     = [ncorr / maxtrials * 100 ]
        entry['SWM_ERR']     = [ntrials - ncorr ]
        entry['SWM_ACC']     = [ncorr / ntrials * 100 ]
        entry['SWM_RT']      = [tmp['RT1'].mean() / 1000 ]
        entry['SWM_RT_SD']   = [tmp['RT1'].std()  / 1000 ]
        
        df_list.append(entry)
        
    return pd.concat(df_list, ignore_index=True)
