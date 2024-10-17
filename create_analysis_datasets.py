# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:49:48 2023

@author: Johannes Steffen
"""

import pandas as pd
import numpy as np
from os import getcwd, path
from utils import load_and_format_SAT_data, \
                  load_and_format_IDP_or_SAW_data, \
                  load_and_format_SWM_data

# set data directory as relative path from this files directory
reppath = getcwd()
datadir = path.join(reppath,"data")

# get rid of deprecation warning
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
    

#%% load and format behavioural data

# SAT raw data
path_hc  =  path.join(datadir,"HC", "SAT","Experiment")
path_aud =  path.join(datadir,"AUD","SAT","Experiment")

stimuli_hc,  conditions_hc,  responses_hc,  mask_hc,  rts_hc,  scores_hc,  ids_hc,  sbj_df_hc  = load_and_format_SAT_data(path_hc)
stimuli_aud, conditions_aud, responses_aud, mask_aud, rts_aud, scores_aud, ids_aud, sbj_df_aud = load_and_format_SAT_data(path_aud)

# exclude datasets with close to random responses and reload
ids_to_exclude_hc  =  sbj_df_hc.loc[sbj_df_hc.total_points  <= 500, 'ID'].values
ids_to_exclude_aud = sbj_df_aud.loc[sbj_df_aud.total_points <= 500, 'ID'].values
print(str(len(ids_to_exclude_hc))  + " hc ids excluded: "  + str(ids_to_exclude_hc))
print(str(len(ids_to_exclude_aud)) + " aud ids excluded: " + str(ids_to_exclude_aud))
stimuli_hc,  conditions_hc,  responses_hc,  mask_hc,  rts_hc,  scores_hc,  ids_hc,  sbj_df_hc  = load_and_format_SAT_data(path_hc,  ids_to_exclude_hc)
stimuli_aud, conditions_aud, responses_aud, mask_aud, rts_aud, scores_aud, ids_aud, sbj_df_aud = load_and_format_SAT_data(path_aud, ids_to_exclude_aud)

sbj_df_hc['group'] = 0
sbj_df_aud['group'] = 1

# IDP data
idp_df_aud = load_and_format_IDP_or_SAW_data(path.join(datadir,'AUD','IDP'),'IDP')
idp_df_hc  = load_and_format_IDP_or_SAW_data(path.join(datadir,'HC', 'IDP'),'IDP')
   
# SWM data
swm_df_aud = load_and_format_SWM_data(path.join(datadir,'AUD','SWM'))
swm_df_hc  = load_and_format_SWM_data(path.join(datadir,'HC', 'SWM'))

# merge all dfs
sbj_df_hc = sbj_df_hc.merge(idp_df_hc, on='ID', validate="one_to_one")
sbj_df_hc = sbj_df_hc.merge(swm_df_hc, on='ID', validate="one_to_one")
sbj_df_aud = sbj_df_aud.merge(idp_df_aud, on='ID', validate="one_to_one")
sbj_df_aud = sbj_df_aud.merge(swm_df_aud, on='ID', validate="one_to_one")

# drop rows from excluded subjects
sbj_df_hc.drop( sbj_df_hc[  sbj_df_hc.ID.isin( ids_to_exclude_hc)  ].index, inplace=True)
sbj_df_aud.drop(sbj_df_aud[ sbj_df_aud.ID.isin(ids_to_exclude_aud) ].index, inplace=True)


#%% load SAT inference results

# load param posterior samples
pars_df = pd.read_csv(path.join(datadir,"PosteriorSamples",'pars_post_samples.csv'), index_col=0, dtype={'IDs':object})
pars_df.rename(columns={'IDs':'ID'}, inplace=True)
pars_df.rename(columns={'group':'group_label'}, inplace=True)
# get order if IDs of inference results (cannot be controlled during inference)
pars_IDorder_aud = pars_df[pars_df.group_label=='AUD'].groupby(by=['ID','subject']).size().reset_index().subject.to_numpy()
pars_IDorder_hc  = pars_df[pars_df.group_label=='HC' ].groupby(by=['ID','subject']).size().reset_index().subject.to_numpy()

# load depth posterior stats
tmp = np.load(path.join(datadir,"PosteriorSamples","plandepth_stats_aud.npz"), allow_pickle=True)
post_depth_aud = tmp['arr_0'].item()
m_prob_aud     = tmp['arr_1']
exc_count_aud  = tmp['arr_2']
tmp = np.load(path.join(datadir,"PosteriorSamples","plandepth_stats_hc.npz"),  allow_pickle=True)
post_depth_hc  = tmp['arr_0'].item()
m_prob_hc      = tmp['arr_1']
exc_count_hc   = tmp['arr_2']
    

#%% load additional covariates from redcap export data file

redcap_df = pd.read_csv(path.join(datadir,"RedCap_data.csv"), index_col=0, dtype={'ID':object})
# calc raven %-wise performance (max. correct trials = 12)
redcap_df["Raven_PER"] = redcap_df.raven_corr / 12 * 100
# create numeric var for graduation level
redcap_df['graduat_labels'] = redcap_df['graduat']
redcap_df['graduat'].replace(to_replace={'Schüler/in allg.bild. Schule':0,
                                                'Schüler/in berufsorientierte S.':1,
                                                'Hauptschulabschluss':2,
                                                'Realschulabschluss':3,
                                                'Polytechnischen Oberschule':4,
                                                'Fachhochschulreife':5,
                                                'Hochschulreife/Abitur':6},
                                    inplace = True)
# drop rows of excluded IDs
redcap_df.drop( redcap_df[ redcap_df.ID.isin(np.concatenate((ids_to_exclude_aud,ids_to_exclude_hc))) ].index, inplace=True)


#%% create and store dataframes on miniblock-, condition- and subject-level as csv for further analyses
 
# create miniblock-wise dfs
mini_blocks = 100

nsub_aud = sbj_df_aud.shape[0]
m_prob_aud_IDindex = (pars_IDorder_aud.repeat(mini_blocks)-1)*mini_blocks + np.tile(np.arange(0,mini_blocks),nsub_aud)

mb_df_aud = pd.concat([sbj_df_aud] * mini_blocks, ignore_index=True).sort_values(by="ID", ignore_index=True)
mb_df_aud["block_num"] = np.tile(np.arange(1,mini_blocks+1),nsub_aud)
mb_df_aud["noise"] = conditions_aud[0,:,:].reshape(mini_blocks * nsub_aud)
mb_df_aud["steps"] = conditions_aud[1,:,:].reshape(mini_blocks * nsub_aud)
mb_df_aud["SAT_RT"] = rts_aud[:,:,0].reshape(mini_blocks * nsub_aud)
mb_df_aud['MeanPD'] = np.matmul(m_prob_aud[0], np.arange(1,4)).reshape(mini_blocks * nsub_aud, order='F')[m_prob_aud_IDindex]
mb_df_aud['StdPDSamples'] = np.matmul(post_depth_aud[0], np.arange(1,4)).std(0).reshape(mini_blocks * nsub_aud, order='F')[m_prob_aud_IDindex]
mb_df_aud.rename(columns={'total_points':'SAT_Total_points',
                          'performance': 'SAT_PER'}, inplace=True)

nsub_hc = sbj_df_hc.shape[0]
m_prob_hc_IDindex = (pars_IDorder_hc.repeat(mini_blocks)-1)*mini_blocks + np.tile(np.arange(0,mini_blocks),nsub_hc)

mb_df_hc = pd.concat([sbj_df_hc] * mini_blocks, ignore_index=True).sort_values(by="ID", ignore_index=True)
mb_df_hc["block_num"] = np.tile(np.arange(1,mini_blocks+1),nsub_hc)
mb_df_hc["noise"] = conditions_hc[0,:,:].reshape(mini_blocks * nsub_hc)
mb_df_hc["steps"] = conditions_hc[1,:,:].reshape(mini_blocks * nsub_hc)
mb_df_hc["SAT_RT"] = rts_hc[:,:,0].reshape(mini_blocks * nsub_hc)
mb_df_hc['MeanPD'] = np.matmul(m_prob_hc[0], np.arange(1,4)).reshape(mini_blocks * nsub_hc, order='F')[m_prob_hc_IDindex]
mb_df_hc['StdPDSamples'] = np.matmul(post_depth_hc[0], np.arange(1,4)).std(0).reshape(mini_blocks * nsub_hc, order='F')[m_prob_hc_IDindex]
mb_df_hc.rename(columns={'total_points':'SAT_Total_points',
                         'performance': 'SAT_PER'}, inplace=True)

SAT_singleMiniblocks_df = pd.concat([mb_df_aud, mb_df_hc], ignore_index=True)
# add redcap covariates: AUD-SUM, Raven_PER, Employment, Higher Education Entrance Qualification
SAT_singleMiniblocks_df = SAT_singleMiniblocks_df.merge(redcap_df[["ID", "aud_sum", "graduat", "job", "Raven_PER"]], on="ID" )
# recover missing age and gender in SAT logfiles from redcap data
nan_ids = SAT_singleMiniblocks_df[SAT_singleMiniblocks_df.age.isna()].ID.unique()
for n in nan_ids: SAT_singleMiniblocks_df.loc[SAT_singleMiniblocks_df.ID==n, ['age','gender']] = redcap_df[redcap_df.ID==n][['age','gender']].replace({'m':0,'w':1}).values

# create subject-wise df of model params
pars_df_subj = pd.pivot(pars_df.groupby(["ID","group_label","parameter"], as_index=False).mean(),index=["ID","subject","order"],columns="parameter", values="value").reset_index().rename_axis(None, axis=1)
pars_df_subj.rename(columns={"$\\alpha$":"model_alpha","$\\beta$":"model_beta","$\\theta$":"model_theta"},inplace=True)

# add params and order to miniblock-wise dfs
SAT_singleMiniblocks_df = pd.merge(left=SAT_singleMiniblocks_df, right=pars_df_subj, on='ID')
SAT_singleMiniblocks_df['block_id'] = SAT_singleMiniblocks_df['block_num']
SAT_singleMiniblocks_df.loc[SAT_singleMiniblocks_df.order == 2, 'block_id'] = (SAT_singleMiniblocks_df.block_id + 49) % 100 + 1

# create condition-wise df
SAT_conditionLevel_df = SAT_singleMiniblocks_df.groupby(by=['ID','steps','noise'], as_index=False).mean()
SAT_conditionLevel_df.drop(columns=['block_num','block_id'], inplace=True)

# create subject-wise df
SAT_subjectLevel_df = SAT_singleMiniblocks_df.groupby(by=['ID'], as_index=False).mean()
SAT_subjectLevel_df.drop(columns=['noise','steps','block_num','block_id'], inplace=True)

# store all dfs
SAT_singleMiniblocks_df.to_csv('SAT_singleMiniblocks.csv')
SAT_conditionLevel_df.to_csv('SAT_conditionLevel.csv')
SAT_subjectLevel_df.to_csv('SAT_subjectLevel.csv')

