
import numpy as np
from scipy.stats import norm,rankdata
import math
import os


import pandas as pd
from pandas.api.types import is_numeric_dtype

from tqdm import tqdm
from itertools import combinations
import operator
from collections import OrderedDict
import copy

import matplotlib as mpl

import plotly.express as px

from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import gridplot
from bokeh.models.ranges import Range1d
from bokeh.transform import factor_cmap
from bokeh.models import (BasicTickFormatter, FixedTicker, FuncTickFormatter,
                          LinearColorMapper, MultiLine,LinearAxis, ColumnDataSource, 
                          FactorRange, Whisker, NumeralTickFormatter, Legend,Line)
from parallel_reset import ParallelResetTool
from parallel_selection_tool import ParallelSelectionTool

import quantecon as qe





########################################
# create_matrix
########################################
def solution(post_data, x, y, key):
    '''
    Function for counting matrices of the number of intersections 
    '''
    _post_data = copy.deepcopy(post_data)
    _post_data['status'] = _post_data['status'].replace('lead', 'rejected') # replace all the leads with the rejected
    _post_data = _post_data[['MFI id', 'client id', 'click time', 'status']]
    
    mask = _post_data['client id'].value_counts()
    _post_data = _post_data[_post_data['client id'].isin(mask.index[mask.gt(1)])] # select all customers with >=2 applications
    group_on_vis_code = _post_data.groupby(['client id', 'MFI id', 'click time']).agg({'status': set}) # grouped by vs_code + Share: MFI id + click time
    vis_code_unique = group_on_vis_code.index.levels[0] # unique vis_code
    
    list_of_comp = []
    for vis_code in tqdm(vis_code_unique):
        dct_of_comp = {}
        vis_code_frame = group_on_vis_code[group_on_vis_code.index.get_level_values(0) == vis_code] # frame by vis_cod
        if len(vis_code_frame) > 1:  # verify that the client is not served by only one MFI
            for idx_1, row_1 in vis_code_frame.iterrows():
                for idx_2, row_2 in vis_code_frame.iterrows():
                    if idx_1[1] == idx_2[1]: # don't look at the same MFI ids 
                        continue
                    if abs(pd.to_datetime(idx_1[2]).timestamp() - pd.to_datetime(idx_2[2]).timestamp()) >= int(30.5*24*60*60): # do not look at applications with time between them > 1 month (may be removed).
                        continue
                    difference = 0
                    
                    if key == 0:
                        if list(row_1['status'])[0] == x and list(row_2['status'])[0] == y: #  intersection of x and y
                            difference = 1
                    if key == 1:
                        if list(row_1['status'])[0] and list(row_2['status'])[0] == y: #  intersection of something and y
                            difference = 1
                    if key == 2:
                        if list(row_1['status'])[0] and list(row_2['status'])[0]: #  intersection of something and anything
                            difference = 1
                    if f'{idx_1[1]}' not in dct_of_comp:   #if there is no such comparable MFI id. 
                        dct_of_comp[f'{idx_1[1]}'] = {f'{idx_2[1]}': difference} 

                    elif f'{idx_2[1]}' in dct_of_comp[f'{idx_1[1]}']: # if there is an MFI id to be compared and there is already an MFI id to be compared with it
                        dct_of_comp[f'{idx_1[1]}'][f'{idx_2[1]}'] += difference

                    elif f'{idx_2[1]}' not in dct_of_comp[f'{idx_1[1]}']: # if there is an MFI id to be compared, but there is no MFI id to be compared with it
                        dct_of_comp[f'{idx_1[1]}'].update({f'{idx_2[1]}': difference})

        list_of_comp.append(dct_of_comp)
    list_of_comp_ = sum([list(list_of_comp[i].keys()) for i in range(len(list_of_comp))], []) #merge all compared MFIs into one list 
    list_of_comp_.sort()
    list_of_comp_.reverse()
    matrix = pd.DataFrame(index = pd.unique(list_of_comp_), columns = pd.unique(list_of_comp_)).fillna(0) #zero matrix creation 
    for i in tqdm(range(len(list_of_comp))):
        for offer_1 in list(list_of_comp[i].keys()):
            for offer_2 in list(list_of_comp[i].get(offer_1).keys()):
                matrix.loc[offer_1, offer_2] += list_of_comp[i].get(offer_1).get(offer_2)   # matrix filling                      
    return matrix

def Probability_matrix_calc(post_data, pos_pos_sol, neg_neg_sol, pos_neg_sol, smg_neg_sol, all_sol, smg_pos_sol):
    '''
    Function for calculating the probabilities of re-approval and approval | rejection.
    '''
    _post_data = copy.deepcopy(post_data)   
    _post_data['status'] = _post_data['status'].replace('lead', 'rejected')
    _post_data = _post_data[['MFI id', 'client id', 'click time', 'status']]
    
    mask = _post_data['client id'].value_counts()
    _post_data_1 = copy.deepcopy(_post_data)
    _post_data_1 = _post_data_1[_post_data_1['client id'].isin(mask.index[mask.gt(1)])] # select all customers with >=2 applications
    
    dict_pos = {}
    dict_neg = {}
    dict_all= {}
    
    for name in _post_data['MFI id'].unique():   # count all positive statuses for each MFI
        dict_pos[f'{name}'] = 0
        if 'sale' in _post_data[_post_data['MFI id'] == name]['status'].value_counts().index:
            summ_name = _post_data[_post_data['MFI id'] == name]['status'].value_counts()['sale']
            dict_pos[f'{name}'] = summ_name

        dict_neg[f'{name}'] = 0 # count all negative statuses for each MFI
        if 'rejected' in _post_data[_post_data['MFI id'] == name]['status'].value_counts().index:
            summ_name_ = _post_data[_post_data['MFI id'] == name]['status'].value_counts()['rejected'] 
            dict_neg[f'{name}'] = summ_name_

        dict_all[f'{name}'] = sum(_post_data[_post_data['MFI id'] == name]['status'].value_counts()) # count all statuses for each MFI

    P_result_neg_sol = pd.DataFrame(columns = all_sol.columns, index = all_sol.index, data = 0)  # initialise the re-rejection matrix
    P_result_pos_sol = pd.DataFrame(columns = all_sol.columns, index = all_sol.index, data = 0) # initialise the reapproval matrix
    
    # get indexes of cells with no common clients
    zero_all_col = {} 
    for col in all_sol.columns:
        ind_zeros = list(all_sol[all_sol[col] == 0].index)
        ind_zeros.remove(col)
        zero_all_col[col] = ind_zeros
    
    # fill in the cells without common customers
    for name_1 in zero_all_col:
        neg = 0
        pos = 0
        if sum(neg_neg_sol.loc[name_1, :]) != 0:
            neg = sum(neg_neg_sol.loc[name_1, :]) / len(neg_neg_sol)
        if sum(pos_pos_sol.loc[name_1, :]) != 0:
            pos = sum(pos_pos_sol.loc[name_1, :]) / len(pos_pos_sol)
            
        neg_delim = sum(smg_neg_sol.loc[name_1, :]) / len(smg_neg_sol)
        pos_delim = sum(smg_pos_sol.loc[name_1, :]) / len(smg_pos_sol)
        for name_2 in zero_all_col.get(name_1):
            
            if  neg_delim != 0:    
                P_result_neg_sol.loc[name_1, name_2] = neg / neg_delim
            else:
                P_result_neg_sol.loc[name_1, name_2] = 1 # when dividing by zero
                
            if pos_delim != 0:   
                P_result_pos_sol.loc[name_1, name_2] = pos / pos_delim
            else: 
                P_result_pos_sol.loc[name_1, name_2] = 1 # when dividing by zero
    
    # get cells with at least one common client present
    non_zero_smg_neg_col = {}
    for col in smg_neg_sol.columns:
        ind_zeros = list(smg_neg_sol[smg_neg_sol[col] != 0].index)
        non_zero_smg_neg_col[col] = ind_zeros
    mat_neg = neg_neg_sol / smg_neg_sol.replace(0,1)
    
    # fill in the cells with at least one common customer present
    for name_1 in non_zero_smg_neg_col:
        for name_2 in non_zero_smg_neg_col.get(name_1):
            P_result_neg_sol.loc[name_1, name_2] = mat_neg.loc[name_1, name_2]
            
    # get cells with at least one common client present
    non_zero_smg_pos_col = {}
    for col in smg_pos_sol.columns:
        ind_zeros = list(smg_pos_sol[smg_pos_sol[col] != 0].index)
        non_zero_smg_pos_col[col] = ind_zeros
    mat_pos = pos_pos_sol / smg_pos_sol.replace(0,1)
    
     # fill in the cells with at least one common customer present
    for name_1 in non_zero_smg_pos_col:
        for name_2 in non_zero_smg_pos_col.get(name_1):
            P_result_pos_sol.loc[name_1, name_2] = mat_pos.loc[name_1, name_2]
    
    # get cells that do not have at least one client (anything and y)
    zero_smg_neg_col = {}
    for col in smg_neg_sol.columns:
        ind_zeros = list(smg_neg_sol[smg_neg_sol[col] == 0].index)
        ind_zeros.remove(col)
        zero_smg_neg_col[col] = ind_zeros
    
    # fill in cells that do not have at least one client (anything and y)
    for name_1 in zero_smg_neg_col:
        for name_2 in zero_smg_neg_col.get(name_1):
            if smg_neg_sol.loc[name_2, name_1] >= smg_pos_sol.loc[name_2, name_1]:
                if dict_neg[name_2] != 0:
                    P_result_neg_sol.loc[name_1, name_2] = neg_neg_sol.loc[name_2, name_1] / dict_neg[name_2] * dict_all[name_2]
                else:
                    P_result_neg_sol.loc[name_1, name_2] = 1    
            else:
                if dict_neg[name_2] != 0:
                    P_result_neg_sol.loc[name_1, name_2] = 1 - pos_neg_sol.loc[name_1, name_2] / dict_neg[name_2] * dict_all[name_2] 
                else:
                    P_result_neg_sol.loc[name_1, name_2] = 0
                    
    # get cells that do not have at least one client (anything and y)
    zero_smg_pos_col = {}
    for col in smg_pos_sol.columns:
        ind_zeros = list(smg_pos_sol[smg_pos_sol[col] == 0].index)
        ind_zeros.remove(col)
        zero_smg_pos_col[col] = ind_zeros
    
    # fill in cells that do not have at least one client (anything and y)
    for name_1 in zero_smg_pos_col:
        for name_2 in zero_smg_pos_col.get(name_1):
            if smg_pos_sol.loc[name_2, name_1] >= smg_neg_sol.loc[name_2, name_1]:
                if dict_pos[name_2] != 0:
                    P_result_pos_sol.loc[name_1, name_2] = pos_pos_sol.loc[name_2, name_1] / dict_pos[name_2] * dict_all[name_2]
                else:
                    P_result_pos_sol.loc[name_1, name_2] = 1
            else:
                if dict_pos[name_2] != 0:
                    P_result_pos_sol.loc[name_1, name_2] = 1 - pos_neg_sol.loc[name_2, name_1] / dict_pos[name_2] * dict_all[name_2] 
                else:
                    P_result_pos_sol.loc[name_1, name_2] = 0
    
    # We get cells whose MFI ids do not share common clients with others
    missing_offers = list(set(_post_data['MFI id'].unique()) - set(P_result_pos_sol.columns))

    P_result_neg_sol.loc[:, [i for i in missing_offers]] = 0
    P_result_pos_sol.loc[:, [i for i in missing_offers]] = 0
    
    for i in missing_offers:
        P_result_neg_sol.loc[i, :] = 0
        P_result_pos_sol.loc[i, :] = 0
    
    # Fill in the cells whose MFI ids do not have common clients with others
    for offer in missing_offers:
        P_result_neg_sol.loc[:, offer] = [dict_neg[i] / dict_all[i] / dict_neg[offer] * dict_all[offer] if dict_neg[offer] != 0 else 1 for i in P_result_neg_sol.index]
        P_result_pos_sol.loc[:, offer] = [dict_pos[i] / dict_all[i] / dict_pos[offer] * dict_all[offer] if dict_pos[offer] != 0 else 1 for i in P_result_pos_sol.index]
    for offer in missing_offers:
        P_result_neg_sol.loc[offer, :] = [dict_neg[offer] / dict_all[offer] / dict_neg[i] * dict_all[i] if dict_neg[i] != 0 else 1 for i in P_result_neg_sol.index]
        P_result_pos_sol.loc[offer, :] = [dict_pos[offer] / dict_all[offer] / dict_pos[i] * dict_all[i] if dict_pos[i] != 0 else 1 for i in P_result_pos_sol.index]

    for offer in missing_offers:
        P_result_pos_sol.loc[offer, offer] = 0
        P_result_neg_sol.loc[offer, offer] = 0
    
    # Rounded to the bounds of probabilities
    P_result_neg_sol[P_result_neg_sol.values > 1] = 1
    P_result_neg_sol[P_result_neg_sol.values < 0] = 0
    
    P_result_pos_sol[P_result_pos_sol.values > 1] = 1
    P_result_pos_sol[P_result_pos_sol.values < 0] = 0
    
    # P_result_pos_sol = P_result_pos_sol.fillna()
    
    return [P_result_pos_sol + np.eye((len(P_result_pos_sol))) , 1 - P_result_neg_sol - np.eye((len(P_result_neg_sol)))]

def create_matrix(post_data):
    '''
    A function that creates re-approval and approval matrices in case of rejection.
    '''
    # We calculate the number of client requests to calculate the probabilities.
    pos_pos_sol = solution(post_data = post_data, x = 'sale', y = 'sale', key = 0)
    neg_neg_sol = solution(post_data = post_data, x = 'rejected', y = 'rejected', key = 0)
    pos_neg_sol = solution(post_data = post_data, x = 'sale', y = 'rejected', key = 0)
    smg_pos_sol = solution(post_data = post_data, x = 'smh', y = 'sale', key = 1)
    smg_neg_sol = solution(post_data = post_data, x = 'smh', y = 'rejected', key = 1)
    all_sol = solution(post_data = post_data, x = 'smh', y = 'smh', key = 2)
    
    
    # Calculating the probabilities.
    res = Probability_matrix_calc(post_data = post_data,
                                   pos_pos_sol = pos_pos_sol, neg_neg_sol = neg_neg_sol, 
                                   pos_neg_sol = pos_neg_sol, smg_neg_sol = smg_neg_sol,
                                   smg_pos_sol = smg_pos_sol, all_sol = all_sol)
    return res





#####################################################
# MC_Score
#####################################################

# ADDING POSTBACK STATISTICS
def prod_statistics_(df_prod, df_post, df_clicks):
    df_prod_new = df_prod.copy()
    
    list_q95 = []
    for mfo in df_prod_new['MFI id']:
        list_q95.append(df_post[(df_post['MFI id'] == mfo) & (df_post['status'] == 'sale')]['processing period'].dropna().quantile(q=0.95))
  
    df_prod_new['processing perio q95 (sale)'] = list_q95
  
    # НВК
    dict_nvk = {}

    for mfo in df_post['MFI id'].unique():
        if (df_post[df_post['MFI id'] == mfo]['period to conversion'].quantile(0.9)) / (60*60) > 1:
            dict_nvk[mfo] = 1
        else:
            dict_nvk[mfo] = 0

    df_prod_new['НВК'] = df_prod_new['MFI id'].map(dict_nvk)
  
    # Service period
    df_post_copy = df_post.copy()
    df_post_copy['period to conversion (н)'] = df_post_copy['period to conversion']
    mfo_notNVK = list(df_prod_new[df_prod_new['НВК'] == 0]['MFI id'])
    mfo_NVK = list(df_prod_new[df_prod_new['НВК'] == 1]['MFI id'])
    for mfo in mfo_notNVK:
        pbc_q50 = df_post_copy[df_post_copy['MFI id'] == mfo]['period to conversion'].quantile(0.5)  
        df_post_copy.loc[list(df_post_copy[df_post_copy['MFI id'] == mfo].index), ['period to conversion (н)']] = df_post_copy[df_post_copy['MFI id'] == mfo]['period to conversion (н)'].apply(lambda x: pbc_q50 if x > 2*60*60 else x)


    df_post_copy['processing period (н)'] = df_post_copy['processing period']
    df_post_copy['card_id_status'] = df_post_copy['MFI id'].apply(lambda x: str(x)) + '_' + df_post_copy['status']
    dict_mean_ps = {}

    for mfo in df_post_copy['MFI id'].unique():
        tmp = df_post_copy[df_post_copy['MFI id'] == mfo]
        mean_ps = tmp[tmp['status'] == 'sale']['processing period'].mean()
        if not pd.isna(mean_ps):
            dict_mean_ps[str(mfo) + '_' + 'lead'] = mean_ps
            dict_mean_ps[str(mfo) + '_' + 'rejected'] = mean_ps
        else:
            dict_mean_ps[str(mfo) + '_' + 'lead'] = df_post_copy[df_post_copy['status'] == 'sale']['processing period'].mean()
            dict_mean_ps[str(mfo) + '_' + 'rejected'] = df_post_copy[df_post_copy['status'] == 'sale']['processing period'].mean()

    df_post_copy.loc[list(df_post_copy[df_post_copy['status'].isin(['lead','rejected'])].index), ['processing period (н)']] = df_post_copy[df_post_copy['status'].isin(['lead','rejected'])]['card_id_status'].map(dict_mean_ps)

    df_post_copy.drop(['card_id_status'], axis=1, inplace=True)

    df_post_copy['service period (н)'] = df_post_copy['period to conversion (н)'] + df_post_copy['processing period (н)']


    mp_q90 = {}
    mp_q50 = {}

    for mfo in df_post_copy['MFI id'].unique():
        tmp = df_post_copy[df_post_copy['MFI id'] == mfo]
        mp_q90[mfo] = tmp['service period (н)'].quantile(0.9)
        mp_q50[mfo] = tmp['service period (н)'].quantile(0.5)

    df_prod_new['service period (q50)'] = df_prod_new['MFI id'].map(mp_q50)
    df_prod_new['service period (q90)'] = df_prod_new['MFI id'].map(mp_q90)

# fairness: "in time"
    share_dict = {}
    for mfo in list(df_prod_new['MFI id']):
        if df_prod_new[df_prod_new['MFI id'] == mfo]['time of service, max'].iloc[0] == np.inf:
            decl = df_prod_new[df_prod_new['MFI id'] == mfo]['time of service, min'].iloc[0] + 6*60
        else:
            decl = df_prod_new[df_prod_new['MFI id'] == mfo]['time of service, max'].iloc[0] + 6*60
    
        df_mfo = df_post[df_post['MFI id'] == mfo]
        df_mfo_sale = df_mfo[df_mfo['status'] == 'sale']
        if len(df_mfo_sale) == 0:
            share = len(df_mfo[df_mfo['processing period'] <= decl]) / len(df_mfo)
        else:
            share = len(df_mfo_sale[df_mfo_sale['processing period'] <= decl]) / len(df_mfo_sale)
    
        share_dict[mfo] = share
    
    df_prod_new['share on time (sale)'] = df_prod_new['MFI id'].map(share_dict)
    df_prod_new['in time'] = df_prod_new['share on time (sale)'].apply(lambda x: 1 if x >= 0.5 else 0)
  
    tmp = df_post.groupby(['MFI id']).apply(lambda df: 1 if ( (len(df[df['status'] == 'rejected']) / len(df)) > 0.05 and len(df[df['status'] == 'sale']) > 0 )  else 0).reset_index(name = 'refusals')
    dict_rej = dict(zip(tmp['MFI id'],tmp['refusals']))
    df_prod_new['refusals'] = df_prod_new['MFI id'].map(dict_rej)
  
    df_prod_new['fairness'] = (1-df_prod_new['unreliability']) + (1-df_prod_new['НВК']) + df_prod_new['in time'] + df_prod_new['refusals']

    tmp1 = df_clicks.groupby('MFI id').size().reset_index(name='Num. clicks')
    tmp2 = df_clicks.groupby('MFI id')['income'].sum().reset_index(name='Sum income')
    tmp2['EPC'] = tmp2['Sum income'] / tmp1['Num. clicks']

    dict_epc = dict(zip(tmp2['MFI id'],tmp2['EPC']))

    df_prod_new['EPC'] = df_prod_new['MFI id'].map(dict_epc)    
    


    return df_prod_new

def BN(df_post, df_prod):
    df_post_sr = df_post[df_post['status 2'].isin(['sale','rejected'])][['MFI id','status 2']].copy()
    dict_status = dict(zip(['sale','rejected'],[1,0]))
    df_post_sr['status 2'] =  df_post_sr['status 2'].map(dict_status)
    alpha0 = df_post_sr['status 2'].sum()
    beta0 = len(df_post_sr[df_post_sr['status 2'] == 0])
    df_alpha = (df_post_sr.groupby('MFI id').sum('status 2').rename(columns={"status 2": "alpha"}) + alpha0).reset_index()
    df_beta = (beta0 + df_post_sr.groupby('MFI id').size().reset_index(name="beta").set_index('MFI id')
        - df_post_sr.groupby('MFI id').sum('status 2').rename(columns={"status 2": "beta"})).reset_index()
    df_prod_sr = df_post_sr.groupby('MFI id').mean('status 2').rename(columns={"status 2": "Доля одобрений"}).reset_index()
    df_n_k = df_post_sr.groupby('MFI id').size().reset_index(name="n_k")
    df_success = (df_post_sr.groupby('MFI id').sum('status 2').rename(columns={"status 2": "success"})).reset_index()
    dict_alpha = dict(zip(df_alpha['MFI id'],df_alpha['alpha']))
    dict_beta = dict(zip(df_beta['MFI id'],df_beta['beta']))
    dict_n_k = dict(zip(df_n_k['MFI id'],df_n_k['n_k']))
    dict_success = dict(zip(df_success['MFI id'],df_success['success']))

    df_prod_sr['alpha0'] = alpha0
    df_prod_sr['beta0'] = beta0
    df_prod_sr['n_k'] = df_prod_sr['MFI id'].map(dict_n_k)
    df_prod_sr['success'] = df_prod_sr['MFI id'].map(dict_success)
    df_prod_sr['fail'] = df_prod_sr['n_k'] - df_prod_sr['success']
    df_prod_sr['alpha'] = df_prod_sr['MFI id'].map(dict_alpha)
    df_prod_sr['beta'] = df_prod_sr['MFI id'].map(dict_beta)
    df_prod_sr['Norm. LAR'] = df_prod_sr['alpha'] / ( df_prod_sr['alpha'] + df_prod_sr['beta'] )


    df_prod_rank = df_prod[['MFI id','average user rating','number of reviews']].copy()
    sums_of_estimates = (df_prod_rank['average user rating'] * df_prod_rank['number of reviews']).sum()
    sum_of_reviews = df_prod_rank['number of reviews'].sum()
    df_prod_rank['Sum of grades'] = df_prod_rank['average user rating'] * df_prod_rank['number of reviews']
    df_prod_rank['Sum of all grades'] = sums_of_estimates
    df_prod_rank['Total number of reviews'] = sum_of_reviews
    df_prod_rank['Norm. average user rating'] = (sums_of_estimates + df_prod_rank['average user rating'] * df_prod_rank['number of reviews']) / (sum_of_reviews + df_prod_rank['number of reviews'])


    dict_normsale = dict(zip(df_prod_sr['MFI id'],df_prod_sr['Norm. LAR']))
    dict_normrank = dict(zip(df_prod_rank['MFI id'],df_prod_rank['Norm. average user rating']))
    df_prod['Norm. LAR'] = df_prod['MFI id'].map(dict_normsale)
    df_prod['Norm. average user rating'] = df_prod['MFI id'].map(dict_normrank)

    return df_prod

# function comparing two mfos according to the given characteristics for "number of advantages"
def comparison_logic(mfo1, mfo2, mfo1_value, mfo2_value, characteristic):
  s12 = 0
  s21 = 0
  more_win = ['Norm. LAR', 'Norm. average user rating','fairness', 'EPC']
  less_win = ['service period (q90)']
  if characteristic in more_win:
    if mfo1_value > mfo2_value:
      s12 = 1
    elif mfo1_value < mfo2_value:
      s21 = 1

  if characteristic in less_win:
    if mfo1_value > mfo2_value:
      s21 = 1
    elif mfo1_value < mfo2_value:
      s12 = 1

  d = {(mfo1,mfo2): s12,
       (mfo2,mfo1):s21}
  
  return d



# function of pairwise comparison of all mfos according to the given characteristics for "number of advantages"
def comparison_to_matrix(df_prod, characteristics):
  name_col_idx_for_matrix_char = list(df_prod['MFI id'])
  matrix_char = pd.DataFrame(data=np.nan, index=name_col_idx_for_matrix_char,columns=name_col_idx_for_matrix_char)

  for comb in tqdm(list(combinations(name_col_idx_for_matrix_char, 2))):
    df_pair = df_prod[df_prod['MFI id'].isin([comb[0],comb[1]])].set_index('MFI id').copy()
    s12 = 0
    s21 = 0
    
    for char in characteristics:
      d = comparison_logic(mfo1 = comb[0],
                          mfo2 = comb[1],
                          mfo1_value = df_pair.loc[comb[0], char],
                          mfo2_value = df_pair.loc[comb[1], char],
                          characteristic = char)
      s12 = s12 + d[(comb[0],comb[1])]
      s21 = s21 + d[(comb[1],comb[0])]
    
    matrix_char.loc[comb[0],comb[1]] = s21
    matrix_char.loc[comb[1],comb[0]] = s12
  
  return matrix_char

# The non-stochastic matrix is converted into a stochastic matrix
def matrix_to_stochastic(M):
  M_st = M.copy()
  for idx in M.index:
    M_st.loc[idx] = M_st.loc[idx] / M_st.loc[idx].sum()
  
  return M_st

# We find the stationary distribution of the Markov chain given by the stochastic matrix
def compute_stationary_distribution(P):
    """Returns the stationary distribution of the markov chain.

       Solves x = xP, where x is the stationary distribution.
       
       x - xP = 0 <-> x(I - P) = 0, such that sum(x) = 1
       
       :param P: The transition matrix of the markov chain.
    """
    
    mc = qe.MarkovChain(P)

    if not mc.is_irreducible:
        raise ValueError('The Markov chain is not irreducible.')
    if not mc.is_aperiodic:
        raise ValueError('The Markov chain is not aperiodic.')

    n = P.shape[0]
    a = np.eye(n) - P
    a = np.vstack((a.T, np.ones(n)))
    b = np.matrix([0] * n + [1]).T
    w = np.linalg.lstsq(a, b, rcond=None)[0]
    error = np.linalg.norm(a * w - b)

    return w,error

# Ranking mfos using Markov chain
def markov_rank(Matrix_char, meth = ''):
  matrix_char_st = matrix_to_stochastic(Matrix_char)
  w, error = compute_stationary_distribution(matrix_char_st)
  print(meth, ': ', error)
  class_dict = dict(zip(matrix_char_st.index,w.transpose().getA()[0]))
  sorted_tuples = sorted(class_dict.items(), key=operator.itemgetter(1))

  sorted_dict = OrderedDict()
  for k, v in sorted_tuples:
      sorted_dict[k] = v

  d = {'MFI id': sorted_dict.keys(),
     meth: sorted_dict.values()}

  df_rank = pd.DataFrame(d)

  return df_rank


# Ratings for the ranked MFIs using Markov chain
def ranging(df_meth):
  df_rank = df_meth.copy()
  more_win = ['MC_Score','MC_Score_first','MC_Score_usual','MC_Score_long',
              'Norm. LAR', 'Norm. average user rating','fairness', 'EPC']
  less_win = ['service period (q90)']

  for col in list(df_rank.columns)[1:]:
    if col in less_win:
      df_rank[col] = df_rank[col].rank(method='min', ascending=True).astype(np.int32)
    if col in more_win:
      df_rank[col] = df_rank[col].rank(method='min', ascending=None).astype(np.int32)

  return df_rank

def MC_Score(df_prod, characteristics, char_name = '0'):
    mfo_vs_mfo = comparison_to_matrix(df_prod = df_prod, 
                                   characteristics = characteristics)
    mfo_vs_mfo = mfo_vs_mfo.replace(np.nan, 0)

    df_stat_distr = markov_rank(Matrix_char = mfo_vs_mfo, meth = 'MC_Score' + char_name)
    df_rank = ranging(df_stat_distr)
    
    return df_rank






#####################################################
# comp_alg
#####################################################

class My_ranking:
    def __init__(self, df_train, df_test, df_prod, df_clicks, characteristics):
        self.df_train = df_train
        self.df_test = df_test
        self.df_prod = df_prod
        self.df_clicks = df_clicks
        self.characteristics = characteristics
    
        
    def MC_Score(self):
        
        self.MC_char_mfo = prod_statistics_(df_prod = self.df_prod, df_post = self.df_train, df_clicks = self.df_clicks)
        
        self.MC_char_mfo = BN(df_post = self.df_train, df_prod = self.MC_char_mfo)
        
        MC_df_rank = MC_Score(df_prod=self.MC_char_mfo,
                              characteristics=self.characteristics,
                              char_name = '')
        
        self.MC_Score_rank = dict(zip(MC_df_rank['MFI id'],MC_df_rank['MC_Score']))
        
        return self.MC_Score_rank
        
        
    def mixed_VZO_rank(self):
        self.df_test['mVZOr'] = self.df_test['MFI page rank']
        
        subid5(df=self.df_test,
               col='glob_rank')
        self.df_test['glob_rank'] = self.df_test['glob_rank'].replace(np.nan, self.df_test['mVZOr'].max() + 1)
        self.df_test.loc[self.df_test['mVZOr'].isna(), 'mVZOr'] = self.df_test[self.df_test['mVZOr'].isna()]['glob_rank']
        self.mVZOr = self.df_test['mVZOr']
        self.df_test.drop('mVZOr', axis=1, inplace=True)
        return self.mVZOr
    
    


class Create_Status:
    def __init__(self, matrices, df_test, dict_expir_id):
        self.pos_pos = matrices[0]
        self.pos_neg = matrices[1]
        self.df_test = df_test
        self.dict_expir_id = dict_expir_id
    
    
        
    def unique(self):
        
        pospos = {}

        for id_name in self.pos_pos.index:
            for col_name in self.pos_pos.columns:
                pospos[id_name + col_name] = self.pos_pos[id_name][col_name]
        
        
        posneg = {}

        for id_name in self.pos_neg.index:
            for col_name in self.pos_neg.columns:
                posneg[id_name + col_name] = self.pos_neg[id_name][col_name]
                
        self.df_test['tmp_pair'] = self.df_test['MFI id'] + self.df_test['new_offer']
        self.df_test['VcMfo'] = self.df_test['client id'] + self.df_test['new_offer']
        
        self.df_test['unique_status'] = self.df_test['VcMfo'].map(self.dict_expir_id)
        
        self.df_test.loc[(self.df_test['status 2'] == 1) & self.df_test['unique_status'].isna(), 'unique_status'] = self.df_test[(self.df_test['status 2'] == 1) & self.df_test['unique_status'].isna()]['tmp_pair'].map(pospos)
        self.df_test.loc[(self.df_test['status 2'] == 0) & self.df_test['unique_status'].isna(), 'unique_status'] = self.df_test[(self.df_test['status 2'] == 0) & self.df_test['unique_status'].isna()]['tmp_pair'].map(posneg)
        
        self.df_test.drop(['tmp_pair', 'VcMfo'], axis=1, inplace=True)
        
        

class Comparison_of_models(My_ranking, Create_Status):
    def __init__(self, matrices, df_train, df_test, df_prod, df_clicks, characteristics, first_model, second_model, status, dict_expir_id):
        self.pos_pos = matrices[0] # re-approval matrix
        self.pos_neg = matrices[1] # approval | rejection matrix
        self.df_train = df_train
        self.df_test = df_test
        self.df_prod = df_prod
        self.df_clicks = df_clicks
        self.characteristics = characteristics
        self.first_model = first_model
        self.second_model = second_model
        self.status = status
        self.dict_expir_id = dict_expir_id
    
    
    def fit_first_model(self):
        if hasattr(My_ranking, self.first_model):
            self.first_rank = getattr(My_ranking, self.first_model)(self)
            
        else:
            print('There is no "{}" model'.format(self.first_model))
    
    
    def fit_second_model(self):
        if hasattr(My_ranking, self.second_model):
            self.second_rank = getattr(My_ranking, self.second_model)(self)
        else:
            print('There is no "{}" model'.format(self.second_model))
            
    
    def arrange_the_ranks(self):
        if isinstance(self.first_rank, dict):
            self.df_test['first_rank'] = self.df_test['MFI id'].map(self.first_rank)
        else:
            self.df_test['first_rank'] = self.first_rank
            
        if isinstance(self.second_rank, dict):
            self.df_test['second_rank'] = self.df_test['MFI id'].map(self.second_rank)
        else:
            self.df_test['second_rank'] = self.second_rank
    


    def create_new_offer(self):
        d_swap = {v: k for k, v in self.second_rank.items()}
        
        f_r_m = int(self.df_test['first_rank'].max())
        s_r_m = np.max(list(d_swap.keys()))
        
        if f_r_m > s_r_m:
            for r in range(s_r_m+1, f_r_m+1):
                d_swap[r] = d_swap[s_r_m]
            
        
        self.df_test['new_offer'] = self.df_test['first_rank'].map(d_swap)
    
    
    def create_income(self):
        dict_income = dict(zip(self.df_prod['MFI id'],self.df_prod['income mean']))
        
        self.df_test['income new'] = self.df_test['new_offer'].map(dict_income)
        
    
    def comparison_first_to_second(self):
        
        self.arrange_the_ranks()
        self.create_new_offer()
        
        if hasattr(Create_Status, self.status):
            getattr(Create_Status, self.status)(self)
        else:
            print('There is no "{}" status'.format(self.status))
            
        self.create_income()
            
    
    def creat_statistics(self):
        sms = []
        sms.append(self.df_test[self.status + '_status'].sum() / len(self.df_test))
        sms.append(np.sum(self.df_test[self.status + '_status'] * self.df_test['income new']))
        sde_status = self.df_test[self.status + '_status'] - self.df_test['status 2']
        sms.append(sde_status.sum() / len(sde_status))
        sms.append(np.sum(sde_status * self.df_test['income new']))
        
        fms = []
        fms.append(self.df_test['status 2'].sum() / len(self.df_test))
        fms.append(np.sum(self.df_test['status 2'] * self.df_test['income']))
        sde_status = self.df_test['status 2'] - self.df_test['status 2']
        fms.append(sde_status.sum() / len(sde_status))
        fms.append(np.sum(sde_status * self.df_test['income']))
        
        d = {self.first_model: fms, self.second_model: sms}
        self.df_stat = pd.DataFrame(data=d, index=['share_of_sale', 'income', 'sde_share_of_sale', 'sde_debt'])
        


def test_CoM(df_post,
             matrices,
             df_prod,
             df_clicks,
             characteristics,
             dict_expir_id,
             start_date = pd.to_datetime('2023-01-31', format='%Y-%m-%d'),
             len_train = pd.to_timedelta('30d'),
             len_test = pd.to_timedelta('1d')
            ):

    
    d = {'date': [],
        'inc': [],
         'mixed_VZO_rank_i': [],
    
         'shr': [],
         'mixed_VZO_rank_s': [],
        }
    
    min_max_data = pd.Series([df_post['tmp_date'].max(), df_clicks['tmp_date'].max()]).min()
    while start_date + len_test <= pd.Timestamp(min_max_data):
        df_train = df_post[(df_post['tmp_date'] > (start_date - len_train).date()) & (df_post['tmp_date'] <= start_date.date())].copy()
        df_test = df_post[(df_post['tmp_date'] > start_date.date()) & (df_post['tmp_date'] <= (start_date + len_test).date())].copy()
        df_test = df_test[df_test['status 2'].isin(['sale','rejected'])].copy()
        df_test['status 2'] = df_test['status 2'].map({'sale':1,
                                                       'rejected':0})
        df_train_clicks = df_clicks[(df_clicks['tmp_date'] > (start_date - len_train).date()) & (df_clicks['tmp_date'] <= start_date.date())].copy()
        
        df_prod_ = df_prod[df_prod['MFI id'].isin(df_train['MFI id'].unique())].copy()
        
        d['date'].append(start_date + len_test)
        print(start_date + len_test)
        
        for first_model in ['mixed_VZO_rank']:
            CoM = Comparison_of_models(matrices = matrices, 
                                       df_train = df_train,
                                       df_test = df_test,
                                       df_prod = df_prod_,
                                       df_clicks=df_train_clicks,
                                       characteristics = characteristics,
                                       first_model = first_model,
                                       second_model = 'MC_Score',
                                       status = 'unique',
                                       dict_expir_id = dict_expir_id
                                       )

            CoM.fit_first_model()
            CoM.fit_second_model()
            CoM.comparison_first_to_second()
            CoM.creat_statistics()

            d[first_model + '_s'].append(CoM.df_stat[CoM.df_stat.index == 'share_of_sale']['MC_Score'].values[0])
            d[first_model + '_i'].append(CoM.df_stat[CoM.df_stat.index == 'income']['MC_Score'].values[0])
        
        d['shr'].append(CoM.df_stat[CoM.df_stat.index == 'share_of_sale'][first_model].values[0])
        d['inc'].append(CoM.df_stat[CoM.df_stat.index == 'income'][first_model].values[0])
        
        start_date = start_date + len_test
            
        
    return pd.DataFrame(d)




#####################################################
# Charts
#####################################################

pallets = ['#cadbce', '#8ab6b6', '#4a919e', '#ce6a6b', '#936fa4', '#5874dc', '#ac9170', '#ffae03', '#e07545', '#c13c87']
code_pallets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dict_pallets = dict(zip(code_pallets, pallets))



# results_with_comparison_method

def comp_graph(df,width=500, height=450):
    
    
    
    p5 = figure(x_axis_type="datetime", 
                width=width, height=height
               )
    p5.yaxis.axis_label = 'Daily income (blue)'
    p5.extra_y_ranges = {"Share": Range1d(start=0, end=1)}
    ax2_5 = LinearAxis(y_range_name="Share", axis_label='Daily share (red)')
    p5.add_layout(ax2_5, 'right')
    
    line_glyph1 = Line(x='date', y='val',
            line_color='black', line_width=3, 
           )
    line1 = p5.add_glyph(ColumnDataSource(df[df['meth'] == 'inc']), line_glyph1)
    
    line_glyph2 = Line(x='date', y='val',
            line_color='black', line_width=3,  
            line_dash="4 4")
    line2 = p5.add_glyph(ColumnDataSource(df[df['meth'] == 'mixed_VZO_rank_i']), line_glyph2)
    
    
    p5.line(x=df[df['meth'] == 'inc']['date'], y=df[df['meth'] == 'inc']['val'],
            color=dict_pallets[6], line_width=3, 
           )
    p5.line(x=df[df['meth'] == 'mixed_VZO_rank_i']['date'], y=df[df['meth'] == 'mixed_VZO_rank_i']['val'],
            color=dict_pallets[6], line_width=3, 
            line_dash="4 4")
    
    
    p5.line(x=df[df['meth'] == 'shr']['date'], y=df[df['meth'] == 'shr']['val'], 
            color=dict_pallets[4], line_width=3, 
            y_range_name='Share')
    p5.line(x=df[df['meth'] == 'mixed_VZO_rank_s']['date'], y=df[df['meth'] == 'mixed_VZO_rank_s']['val'], 
            color=dict_pallets[4], line_width=3, 
            y_range_name='Share', line_dash="4 4")
    
    
    
    legend = Legend(
        items=[("historical data", [line1]), ("baseline algorithm", [line2])],
        location='top_left',
    )
    p5.add_layout(legend)
    p5.yaxis.axis_label_text_font_size = "10pt"
    p5.y_range.start = 0
    p5.y_range.range_padding = 0.02
    p5.xaxis.major_label_orientation = np.pi/6
    p5.xgrid.grid_line_color = None
    p5.axis.minor_tick_line_color = None
    p5.outline_line_color = None
    p5.background_fill_color=None 
    p5.border_fill_color=None
    p5.output_backend = "svg" 
    

    return p5



# clients_difference

def colorFader(c1,c2,mix=0, degree = 1): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-(mix)**degree)*c1 + (mix)**degree*c2)


def client_difference_size_app(df_post_new, width=800, height=500, color = dict_pallets[10]):
    less_weight = lambda x: [x['index'].min(), x['index'].max()] if len(x)>1 else [list(x['index'])[0]]
    
    vc_size = df_post_new.groupby('client id').size().reset_index(name='size').sort_values('size',ascending=False).reset_index(drop=True).reset_index()
    tmp = vc_size.groupby('size').apply(less_weight).reset_index(name='x')

    x = []
    y = []

    for index,row in tmp.iterrows():
        x = row['x'] + x
        y = [row['size']] * len(row['x']) + y
    
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"

    p1 = figure(toolbar_location=None, x_axis_label='Client id',
                y_axis_label='Number of applications',
                width=width, height=height, tools=TOOLS)

    p1.circle(x, y,size=1, line_color=color, fill_color=color,)
    p1.line(x, y, color=color)
    
    
    p1.xaxis.formatter=NumeralTickFormatter(format="0a")
    p1.xaxis.minor_tick_line_color = None
    p1.xaxis.major_label_text_font_size = '7pt'
    p1.yaxis.major_label_text_font_size = '7pt'
    p1.xaxis.axis_label_text_font_size = "7pt"
    p1.yaxis.axis_label_text_font_size = "7pt"
    p1.background_fill_color=None 
    p1.border_fill_color=None
    p1.output_backend = "svg"


    print('the share of clients with no more than 1 applications is equal to {}'.format(len(vc_size[vc_size['size'] <= 1]) / len(vc_size)))
    print('the share of clients with no more than 5 applications is equal to {}'.format(len(vc_size[vc_size['size'] <= 5]) / len(vc_size)))
    print('the share of the former among the latter is equal {}'.format(len(vc_size[vc_size['size'] <= 1]) / len(vc_size[vc_size['size'] <= 5])))
    
    return p1

def client_difference_app(df_post_new, width=500, height=250):
    vc_size = df_post_new.groupby('client id').size().reset_index(name='size').sort_values('size',ascending=False).reset_index(drop=True).reset_index()
    
    df = df_post_new.copy()
    df = pd.merge(df, vc_size, on='client id')
    df = df[df['size'] > 1]
    
    
    df_cli_mfi = df.groupby(['index', 'MFI id']).size().reset_index(name='size')
    
    df_MFIs = df.groupby('MFI id').size().reset_index(name='size').sort_values('size')
    MFIs = df_MFIs['MFI id'].tolist()

    dict_size_cl = dict(zip(df['index'],df['size']))
    df_cli_mfi['cl_color'] = df_cli_mfi['index'].map(dict_size_cl)

    
    dict_color = {2: dict_pallets[7], 3: dict_pallets[8], 4: dict_pallets[9]}
    k=1
    n = len(df_cli_mfi['cl_color'].unique()) - 3
    for s in df_cli_mfi['cl_color'].sort_values().unique():
        if s <=4:
            continue
        
        dict_color[s] = colorFader(c1 = dict_pallets[5],c2 = dict_pallets[10],mix=k/n, degree = .25)
        k = k + 1
        
    df_cli_mfi['cl_color'] = df_cli_mfi['cl_color'].map(dict_color)

    p1 = figure(toolbar_location=None, x_axis_label='Client id',y_axis_label='MFI id',y_range=MFIs, width=width, height=height)

    p1.scatter(x = 'index', y = 'MFI id', 
               source=df_cli_mfi, color='cl_color',
              fill_alpha=0.3,
              line_color=None,
              size=3
             )
    
    p1.xaxis.formatter=NumeralTickFormatter(format="0a")
    p1.x_range.range_padding = 0.1
    p1.yaxis.minor_tick_line_color = None
    p1.yaxis.major_label_text_font_size = '0.1pt'  # turn off y-axis tick labels
    p1.xaxis.major_label_text_font_size = '7pt'
    p1.xaxis.axis_label_text_font_size = "7pt"
    p1.yaxis.axis_label_text_font_size = "7pt"
    p1.xgrid.grid_line_color = None
    p1.axis.minor_tick_line_color = None
    p1.outline_line_color = None
    p1.background_fill_color=None 
    p1.border_fill_color=None
    p1.output_backend = "svg"
    
    return p1



# Table_subid5
def shares_of_MFIs_for_various_ranks(df_post_new, SHOW=False, SAVE=True):
    post_tmp = df_post_new[['MFI id', 'loan type','page id', 'MFI page rank', 'click time',]].copy()
    post_tmp = post_tmp[post_tmp['MFI page rank'].isin([1,2,3,4,5,6])].sort_values('click time').copy()
    mfo_size = post_tmp.groupby('MFI id').size().reset_index(name='size').sort_values('size')
    
    MFIs = (mfo_size['MFI id'].unique())
    post_tmp.loc[~post_tmp['MFI id'].isin(MFIs[-7:]), 'MFI id'] = 'Others'
    MFIs = list(['Others']) + list(MFIs[-7:])
    
    MFIs = MFIs[::-1]
    
    subid5_top_table = pd.DataFrame(columns = [1,2,3,4,5,6], index = MFIs, data = 0)
    
    for rank in [1,2,3,4,5,6]:
        for mfo in MFIs:
            post_tmp2 = post_tmp[post_tmp['MFI page rank'].isin([rank])]
            subid5_top_table.loc[mfo, rank] = len(post_tmp2[post_tmp2['MFI id'] == mfo])
    
    
    subid5_top_table1 = subid5_top_table.copy()
    for rank in [1,2,3,4,5,6]:
        subid5_top_table1[rank] = ((subid5_top_table1[rank] / subid5_top_table1[rank].sum()) * 100).apply(lambda x: round(x, 1))

    fig = px.imshow(np.array(subid5_top_table1).T, 
                    aspect="auto",
                    color_continuous_scale=['white',dict_pallets[3]],
                    range_color=[0,subid5_top_table1.max().max()],
                    text_auto=True,
                    labels=dict(color="%app.")
                   )

    fig.update_layout(
        yaxis = dict(
            tickmode = 'array',
            tickvals = [0,1,2,3,4,5],
            ticktext = ['1 ','2 ','3 ','4 ','5 ','6 ']
        ),
        xaxis = dict(
            tickmode = 'array',
            tickvals = [i for i in range(len(MFIs))],
            ticktext = MFIs,
            tickangle = 45
        ),
        yaxis_title="MFI page rank",
    )

    if SHOW == True:
        fig.show()
        
    if SAVE == True:
        fig.write_html("Charts/Table_subid5.html")


# status_6_mfi
def status_6_mfis(df_post_new):

    mfi_size = df_post_new.groupby('MFI id').size().reset_index(name='size').sort_values('size')
    MFIs = mfi_size['MFI id'].unique()
    MFIs = list(MFIs[-20:])
    MFIs = MFIs[-2:] + ['MFI 56','MFI 18','MFI 64','MFI 29'] 
    mfi_count_st = df_post_new.groupby('MFI id', as_index=False)['status'].value_counts()

    req = ['p', "r", "s"]
    rejected_req = []
    lead_req = []
    sale_req = []
    all_req = []
    for mfi in MFIs:
        rejected_req.append(mfi_count_st[(mfi_count_st['MFI id'] == mfi) & (mfi_count_st['status'].isin(['rejected']))]['count'].sum())
        lead_req.append(mfi_count_st[(mfi_count_st['MFI id'] == mfi) & (mfi_count_st['status'].isin(['lead']))]['count'].sum())
        sale_req.append(mfi_count_st[(mfi_count_st['MFI id'] == mfi) & (mfi_count_st['status'] == 'sale')]['count'].tolist()[0])
        mfi_all_req = mfi_count_st[mfi_count_st['MFI id'] == mfi]['count'].sum()
        all_req.append(mfi_all_req)
    
    
    data1 = {'MFIs' : MFIs,
             'p'   : lead_req,
             'r'   : rejected_req,
             's'   : sale_req}
    
    data2 = {'MFIs' : MFIs,
             'p'   : lead_req / np.array(all_req),
             'r'   : rejected_req / np.array(all_req),
             's'   : sale_req / np.array(all_req)}
    colors2 = [dict_pallets[6], dict_pallets[4], dict_pallets[3]]


    x = [ (mfi, status) for mfi in MFIs for status in req ]
    counts = sum(zip(data1['p'], data1['r'], data1['s']), ()) # like an hstack

    p1 = figure(x_range=FactorRange(*x), width=500, height=430)
    p1.yaxis.axis_label = "Number of applications"
    
    p1.extra_y_ranges = {"Share app.": Range1d(start=0, end=1)}
    ax2 = LinearAxis(y_range_name="Share app.",axis_label='Shares')
    p1.add_layout(ax2, 'right')
    

    p1.vbar_stack(req, x='MFIs', width=3.5, color=colors2, source=data2,  y_range_name='Share app.',fill_alpha=.8)
    

    p1.circle(x=x, y=counts, size=14,
              line_color='black',
              fill_alpha=1,
              fill_color=factor_cmap('x', palette=colors2, factors=req, start=1, end=2),
              line_width=1)
    source_ = ColumnDataSource(data=dict(base=x, lower=[0]*len(x), upper=counts))
    sp = Whisker(base="base", upper="upper", lower="lower", source=source_,
                level="annotation", line_width=2, upper_head=None, lower_head=None,
                line_color='black'
                )
    p1.add_layout(sp)

    
    p1.xaxis.major_label_text_font_size = '14pt'
    p1.yaxis.major_label_text_font_size = '14pt'
    p1.xaxis.axis_label_text_font_size = "14pt"
    p1.yaxis.axis_label_text_font_size = "14pt"
    p1.y_range.start = 0
    p1.x_range.range_padding = 0.1
    p1.grid.grid_line_color = None
    p1.axis.minor_tick_line_color = None
    p1.outline_line_color = None
    p1.xaxis.group_label_orientation = np.pi/2
    p1.background_fill_color=None 
    p1.border_fill_color=None
    p1.output_backend = "svg"
     
    return p1


# income_4_mfis
def income_4_mfis(df_clicks):
    # MFIs = ['MFI 56','MFI 18','MFI 64','MFI 29']
    MFIs = ['MFI 87','MFI 20','MFI 56','MFI 18','MFI 64','MFI 29']
    colors_mfi = dict(zip(MFIs, [dict_pallets[8],dict_pallets[7],dict_pallets[9],dict_pallets[10],dict_pallets[6],dict_pallets[5]]))
    
    df_tmp = df_clicks[df_clicks['MFI id'].isin(MFIs)][['MFI id', 'income']].copy()
    df_tmp2 = df_tmp.groupby(['MFI id','income']).size().reset_index(name='size')
    df_tmp2['for_map'] = df_tmp2['MFI id'] + df_tmp2['income'].apply(lambda x: str(x))

    tmp = df_tmp.groupby('MFI id').size().reset_index(name='Num. of clicks')
    dict_Num_of_clicks = dict(zip(tmp['MFI id'], tmp['Num. of clicks']))
    df_tmp2['Num. of clicks'] = df_tmp2['MFI id'].map(dict_Num_of_clicks)

    tmp = df_tmp.groupby(['MFI id','income'])['income'].sum().reset_index(name='Sum the income')
    tmp['for_map'] = tmp['MFI id'] + tmp['income'].apply(lambda x: str(x))
    dict_Sum_the_income = dict(zip(tmp['for_map'], tmp['Sum the income']))
    df_tmp2['Sum the income'] = df_tmp2['for_map'].map(dict_Sum_the_income)

    df_tmp2['share'] = 30*(df_tmp2['Sum the income'] / df_tmp2['Num. of clicks'])

    df_tmp2['color_mfi'] = df_tmp2['MFI id'].map(colors_mfi)
    
    df_tmp2 = df_tmp2[df_tmp2['income'] > 0].copy()
    

    source = ColumnDataSource(df_tmp2)

    p3 = figure(width=500, height=500)

    p3.scatter(x="size", y="income", size="share", legend_group='MFI id', source=source, 
              line_color="#202020", fill_color='color_mfi'
              )


    # p3.legend.title = 'MFI id'
    p3.legend.location = (300,240)
    p3.legend.border_line_alpha = 0
    p3.legend.background_fill_alpha = 0
    p3.legend.glyph_height = 30
    p3.legend.glyph_width = 30
    p3.legend.label_text_font_size = '10pt'
    p3.legend.title_text_font_size = '10pt'
    p3.xaxis.major_label_text_font_size = '12pt'
    p3.yaxis.major_label_text_font_size = '12pt'
    p3.xaxis.axis_label_text_font_size = "12pt"
    p3.yaxis.axis_label_text_font_size = "12pt"
    
    p3.x_range.end = 620
    p3.y_range.range_padding = .2
    p3.axis.major_tick_line_color = None
    p3.axis.major_tick_out = 0
    p3.axis.axis_line_color = None
    p3.grid.grid_line_color = None
    p3.outline_line_color = None
    p3.yaxis.axis_label = 'Income'
    p3.xaxis.axis_label = 'Number of sales'
    p3.background_fill_color=None 
    p3.border_fill_color=None
    p3.output_backend = "svg"  

    return p3


# rating_6_mfi
def rating_6_mfi(df_post_new, df_prod_new):

    mfi_size = df_post_new.groupby('MFI id').size().reset_index(name='size').sort_values('size')
    MFIs = mfi_size['MFI id'].unique()
    MFIs = list(MFIs[-20:])
    MFIs = MFIs[-2:] + ['MFI 56','MFI 18','MFI 64','MFI 29'] 

    average_rating_according_to_reviews = []
    number_of_reviews = []
    for mfi in MFIs:
        prod_mfi = df_prod_new[df_prod_new['MFI id'] == mfi]
        average_rating_according_to_reviews.append(prod_mfi['average user rating'].sum() / len(prod_mfi))
        number_of_reviews.append(prod_mfi['number of reviews'].sum() / len(prod_mfi))
    
    
    
    p5 = figure(x_range=MFIs, 
                width=500, height=450
               )
    p5.yaxis.axis_label = 'Number of reviews (blue)'
    
    p5.extra_y_ranges = {"average user rating by reviews": Range1d(start=0, end=5)}
    ax2_5 = LinearAxis(y_range_name="average user rating by reviews", axis_label='average user rating by reviews (red)')
    p5.add_layout(ax2_5, 'right')

    p5.line(x=MFIs, y=number_of_reviews, color=dict_pallets[6], line_width=3)
    p5.circle(x=MFIs, y=number_of_reviews, size=10,
         line_color=dict_pallets[6], fill_color="white", line_width=3)
    
    p5.line(x=MFIs, y=average_rating_according_to_reviews, color=dict_pallets[4], line_width=3, y_range_name='average user rating by reviews')
    p5.circle(x=MFIs, y=average_rating_according_to_reviews, size=10,
         line_color=dict_pallets[4], fill_color="white", line_width=3, y_range_name='average user rating by reviews')
    
    p5.xaxis.major_label_text_font_size = '14pt'
    p5.yaxis.major_label_text_font_size = '14pt'
    p5.xaxis.axis_label_text_font_size = "14pt"
    p5.yaxis.axis_label_text_font_size = "14pt"
    p5.y_range.start = 0
    p5.x_range.range_padding = 0.04
    p5.y_range.range_padding = 0.09
    p5.xaxis.major_label_orientation = np.pi/6
    p5.xgrid.grid_line_color = None
    p5.axis.minor_tick_line_color = None
    p5.outline_line_color = None
    p5.background_fill_color=None 
    p5.border_fill_color=None
    p5.output_backend = "svg" 
    

    return p5


# status_and_loan_type
def status_and_loan_type(df_post_new):
    
    status_dict = dict(zip(['sale', 'lead', 'rejected'], ['1','2','3']))
    df_status = df_post_new.groupby('status').size().reset_index(name='size')
    df_status['share_status'] = df_status['size'] / df_status['size'].sum()
    df_status['code'] = df_status['status'].map(status_dict)
    
    loan_dict = dict(zip(['loan-first', 'loan-usual', 'loan-long'], ['1','2','3']))
    df_loan = df_post_new.groupby('loan type').size().reset_index(name='size')
    df_loan['share_loan'] = df_loan['size'] / df_status['size'].sum()
    df_loan['code'] = df_loan['loan type'].map(loan_dict)

    result = pd.merge(df_status, df_loan, on="code")[['code', 'share_status', 'share_loan']].sort_values('code')
    result.rename(columns={'share_status': 'status',
                           'share_loan': 'loan-type'}, inplace=True)
    

    atr = ['status', 'loan-type']
    share1 = np.array(result[atr].values[0])
    share2 = np.array(result[atr].values[1])
    share3 = np.array(result[atr].values[2])
    
    data1 = {'atr.' : atr,
             '1' : share1,
             '2' : share2,
             '3' : share3,
             'c1': ['s1', 'l1'],
             'c2': ['s2', 'l2'],
             'c3': ['s3', 'l3'],
                }

   
    p4 = figure(x_range=['status', 'loan-type'], width=250, height=250,
               toolbar_location=None)
    
    p4.vbar_stack(['1','2','3'], x='atr.',  width=0.9,
                  fill_color=[
                            factor_cmap(state, 
                                        palette=[dict_pallets[3], dict_pallets[1], dict_pallets[4]] + 
                                        [colorFader(dict_pallets[3],dict_pallets[10],mix=.15, degree = 1),
                                         colorFader(dict_pallets[1],dict_pallets[10],mix=.15, degree = 1),
                                         colorFader(dict_pallets[4],dict_pallets[10],mix=.15, degree = 1)],
                                        factors=["s1","s2","s3","l1","l2","l3"]) 
                            for state in ["c1", "c2", "c3"]],
                  source=data1, 
                  fill_alpha=.9,
                  line_alpha=0
             )
    
    p4.text(x=atr, y=[.04,0], text=['sale ({}%)'.format(round(share1[0]*100,2)), 
                                  'interest-free ({}%)'.format(round(share1[1]*100,2))], 
            text_align = 'center',
           text_font_size="8pt", text_color='black', text_alpha=.5)
    
    p4.text(x=atr, y=[.5,.5], text=['pending ({}%)'.format(round(share2[0]*100,2)),
                                    'standard ({}%)'.format(round(share2[1]*100,2))],
            text_align = 'center',
           text_font_size="8pt", text_color='black', text_alpha=.5)
    y_ = np.max(share1 + share2)
    p4.text(x=atr, y=[y_ + .01, y_ + .005], text=['rejected ({}%)'.format(round(share3[0]*100,2)),
                                    'long-term ({}%)'.format(round(share3[1]*100,2))],
            text_align = 'center',
           text_font_size="8pt", text_color='black', text_alpha=.5)
    
    p4.xaxis.major_label_text_font_size = '17pt'
    p4.xgrid.grid_line_color = None
    p4.ygrid.grid_line_color = None
    p4.yaxis.visible = False
    p4.axis.axis_line_color = None
    p4.axis.major_tick_line_color = None
    p4.outline_line_color = None
    p4.background_fill_color=None 
    p4.border_fill_color=None
    p4.output_backend = "svg" 

    
    return p4



# time_6_mfi
def box_plot(df, score, q):

    # find the quartiles and IQR for each category
    groups = df.groupby('MFI id', sort=False)
    q1 = groups.quantile(q=q)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=1 - q)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    MFIs = groups.apply(lambda x: x.name).tolist()

    # find the outliers for each category
    def outliers(group):
        mfi = group.name
        return group[(group[score] > upper.loc[mfi][score]) | (group[score] < lower.loc[mfi][score])][score]
    out = groups.apply(outliers).dropna()

    # prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = list(out.index.get_level_values(0))
        outy = list(out.values)

    p = figure(x_range=MFIs,width=500, height=500)

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper[score] = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,score]),upper[score])]
    lower[score] = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,score]),lower[score])]

    # stems
    p.segment(MFIs, upper[score], MFIs, q3[score], line_color="black")
    p.segment(MFIs, lower[score], MFIs, q1[score], line_color="black")

    # boxes
    p.vbar(MFIs, 0.7, q2[score], q3[score], fill_color=dict_pallets[8], line_color="black")
    p.vbar(MFIs, 0.7, q1[score], q2[score], fill_color=dict_pallets[7], line_color="black")

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(MFIs, lower[score], 0.2, 0.01, line_color="black")
    p.rect(MFIs, upper[score], 0.2, 0.01, line_color="black")

    # outliers
    if not out.empty:
        p.circle(outx, outy, size=6, color=dict_pallets[9], fill_alpha=0.6)
    
    p.xaxis.major_label_text_font_size = '14pt'
    p.yaxis.major_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.y_range.start = 0
    p.y_range.end = 1410
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = np.pi/6
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.yaxis.axis_label = score + " (h)"
    p.background_fill_color=None 
    p.border_fill_color=None
    p.output_backend = "svg" 

    return p



# page_rank_and_numer of requests
def number_of_req_for_rank_over_time(df_post_new, width=800, height=500):

    # TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"
    
    
    p1 = figure(x_axis_label='Time',
                y_axis_label='Number of applications',
                x_axis_type="datetime",
                width=width, height=height, toolbar_location=None, 
                )

    for rank in range(1, 9):
        df_post_new_rank = df_post_new[df_post_new['MFI page rank'] == rank].sort_values('click time').copy()
        df_post_new_rank['Count'] = [i+1 for i in range(len(df_post_new_rank))]
        x = df_post_new_rank['click time']
        y = df_post_new_rank['Count']

        p1.line(x, y, legend_label=str(rank), color=dict_pallets[11 - rank], line_width=2)

    p1.legend.title = 'MFI page rank'
    p1.legend.location = "top_left"
    p1.xaxis.major_label_text_font_size = '14pt'
    p1.yaxis.major_label_text_font_size = '14pt'
    p1.xaxis.axis_label_text_font_size = "14pt"
    p1.yaxis.axis_label_text_font_size = "14pt"
    p1.x_range.range_padding = 0.001
    
    p1.background_fill_color=None 
    p1.border_fill_color=None
    p1.output_backend = "svg"
    
    return p1

def number_of_requests(df_post_new, width=500, height=250):
 
    
    df_ = df_post_new[['MFI page rank', 'status', 'loan type']].copy()
    df_ = df_[~df_['MFI page rank'].isna()].sort_values('MFI page rank').copy()
    df_ = df_[df_['MFI page rank']<=60]
    
    x = list(df_['MFI page rank'].unique())
    counts = list(df_.groupby('MFI page rank').size())
    
    colors = x.copy()
    
    for i in range(len(x)):
        if x[i] <= 10:
            colors[i] = colorFader('white', dict_pallets[3],mix=.4, degree = 1)
        elif 10 < x[i] <= 20:
            colors[i] = colorFader('white', dict_pallets[3],mix=.65, degree = 1)
        else:
            colors[i] = dict_pallets[3]
    
    p5 = figure(width=width, height=height)

    p5.vbar(x=x, top=counts, width=0.9, fill_color=colors, line_color=None, fill_alpha=.8)
    
    p5.xaxis.major_label_text_font_size = '14pt'
    p5.yaxis.major_label_text_font_size = '14pt'
    p5.xaxis.axis_label_text_font_size = "14pt"
    p5.yaxis.axis_label_text_font_size = "14pt"
    p5.xgrid.grid_line_color = None
    p5.y_range.start = 0
    p5.yaxis.axis_label = 'Total number of applications'
    p5.xaxis.axis_label = 'MFI page rank'
    p5.background_fill_color=None 
    p5.border_fill_color=None
    p5.output_backend = "svg"
    

    return p5



# mormlar and normrate
def parallel_plot(df, color=None, palette=None, MFIs = [], width=340, height=120,):
    """From a dataframe create a parallel coordinate plot
    """
    npts = df.shape[0]
    ndims = len(df.columns)

    if color is None:
        color = np.ones(npts)
    if palette is None:
        palette = ['#ff0000']
        
    if len(MFIs) > 0:
        line_width_list = [.5 if mfi in MFIs else 0.35 for mfi in df.index]
        line_alpha_list = [1 if mfi in MFIs else 0.21 for mfi in df.index]

    cmap = LinearColorMapper(high=color.min(),
                             low=color.max(),
                             palette=palette)
    
    data_source = ColumnDataSource(dict(
        xs=np.arange(ndims)[None, :].repeat(npts, axis=0).tolist(),
        ys=np.array((df-df.min())/(df.max()-df.min())).tolist(),
        color=color,
        line_width=line_width_list,
        line_alpha=line_alpha_list))
    
        
    p = figure(
        x_range=(-0.1, ndims - 0.9),
               y_range=(0, 1),
               width=width, height=height
              )
    

    # Create x axis ticks from columns contained in dataframe
    fixed_x_ticks = FixedTicker(
        ticks=np.arange(ndims), minor_ticks=[])
    formatter_x_ticks = FuncTickFormatter(
        code="return columns[index]", args={"columns": df.columns})
    p.xaxis.ticker = fixed_x_ticks
    p.xaxis.formatter = formatter_x_ticks
    p.xaxis.major_label_text_font_size = "7pt"
    p.xaxis.major_label_text_font_size = "7pt"
    p.xaxis.major_tick_line_width = 0

    p.yaxis.visible = False
    p.y_range.start = 0
    p.y_range.end = 1
    p.y_range.bounds = (-0.1, 1.1) # add a little padding around y axis
    p.xgrid.visible = False
    p.ygrid.visible = False

    # Create extra y axis for each dataframe column
    # tickformatter = BasicTickFormatter(precision=3)
    for index, col in enumerate(df.columns):
        if is_numeric_dtype(df[col]):
            if np.max(df[col].apply(lambda x: x%1)) == 0:
                tickformatter = BasicTickFormatter(precision=0)
            else:
                tickformatter = BasicTickFormatter(precision=3)

            start = df[col].min()
            end = df[col].max()
            bound_min = start + abs(end-start) * (p.y_range.bounds[0] - p.y_range.start)
            bound_max = end + abs(end-start) * (p.y_range.bounds[1] - p.y_range.end)

            range1d = Range1d(start=bound_min, end=bound_max, bounds=(bound_min, bound_max))
            p.extra_y_ranges.update({col: range1d})
            if col != 'MFI id':
                fixedticks = FixedTicker(
                    ticks=np.linspace(start, end, 15), minor_ticks=[])
                major_label_overrides = {}
                p.add_layout(LinearAxis(fixed_location=index, y_range_name=col,
                                    ticker=fixedticks, formatter=tickformatter, 
                                        major_label_overrides=major_label_overrides,
                                        major_label_text_font_size = "5.8pt",
                                        major_tick_line_width=.01,
                                        major_tick_out=1,
                                        axis_line_width=.1), 'right')
            else:
                fixedticks = FixedTicker(
                    ticks=np.arange(end+1), minor_ticks=[])
                major_label_overrides = {i: str(name) for i, name in enumerate(df.index)}
                p.add_layout(LinearAxis(fixed_location=index, y_range_name=col,
                                        ticker=fixedticks, formatter=tickformatter, 
                                        major_label_overrides=major_label_overrides,
                                        major_label_text_font_size = "5.8pt",
                                        major_tick_line_width=.01,
                                        major_tick_out=1,
                                        axis_line_width=.1), 'left')


            



    # create the data renderer ( MultiLine )
    # specify selected and non selected style
    non_selected_line_style = dict(line_color={'field': 'color', 'transform': cmap}, line_width=0.5, line_alpha=0.3)

    selected_line_style = dict(line_color={'field': 'color', 'transform': cmap}, line_width=1)
    
    my_selected_line_style = dict(line_color={'field': 'color', 'transform': cmap}, line_width='line_width', line_alpha='line_alpha')
    
    if len(MFIs) > 0:
        parallel_renderer = p.multi_line(
            xs="xs", ys="ys", source=data_source, **my_selected_line_style)
    else:
        parallel_renderer = p.multi_line(
            xs="xs", ys="ys", source=data_source, **non_selected_line_style)
    

    # Specify selection style
    selected_lines = MultiLine(**selected_line_style)

    # Specify non selection style
    nonselected_lines = MultiLine(**non_selected_line_style)

    parallel_renderer.selection_glyph = selected_lines
    parallel_renderer.nonselection_glyph = nonselected_lines
    p.y_range.start = p.y_range.bounds[0]
    p.y_range.end = p.y_range.bounds[1]

    rect_source = ColumnDataSource({
        'x': [], 'y': [], 'width': [], 'height': []
    })

    # add rectangle selections
    selection_renderer = p.rect(x='x', y='y', width='width', height='height',
                                source=rect_source,
                                fill_alpha=0.7, fill_color='#009933')
    selection_tool = ParallelSelectionTool(
        renderer_select=selection_renderer, renderer_data=parallel_renderer,
        box_width=0.001)
    # custom resets (reset only axes not selections)
    reset_axes = ParallelResetTool()

    # add tools and activate selection ones
    p.add_tools(selection_tool, reset_axes)
    p.toolbar.active_drag = selection_tool
    
    p.yaxis.major_label_text_font_size = "4pt"
    
    p.xaxis.axis_line_alpha = 0
    p.outline_line_color = None
    p.background_fill_color=None 
    p.border_fill_color=None
    p.output_backend = "svg" 

    return p




def gamma(df):
    n = len(df)
    p00 = len(df[(df['status'] == 0) & (df['device browser'] == 0)]) / n
    p11 = len(df[(df['status'] == 1) & (df['device browser'] == 1)]) / n
    p10 = len(df[(df['status'] == 1) & (df['device browser'] == 0)]) / n
    p01 = len(df[(df['status'] == 0) & (df['device browser'] == 1)]) / n
    
    n00 = len(df[(df['status'] == 0) & (df['device browser'] == 0)])
    n11 = len(df[(df['status'] == 1) & (df['device browser'] == 1)])
    n10 = len(df[(df['status'] == 1) & (df['device browser'] == 0)])
    n01 = len(df[(df['status'] == 0) & (df['device browser'] == 1)])
    
    if (p00*p11 == 0) & (p10*p01 != 0):
        gamma = -1
        left, right = np.nan,np.nan
    
    if (p00*p11 != 0) & (p10*p01 == 0):
        gamma = 1
        left, right = np.nan,np.nan
    
    if (p00*p11 == 0) & (p10*p01 == 0):
        gamma = np.nan
        left, right = np.nan,np.nan
    
    if (p00*p11 != 0) & (p10*p01 != 0):
        omega = (p00*p11) / (p10*p01)
        gamma = (np.sqrt(omega) - 1) / (np.sqrt(omega) + 1)
        
        left, right = CI(epsilon = .005, 
                         gamma = gamma,
                         n1=n00, n2=n11, n3=n10, n4=n01)
        
    return gamma, left, right

def tanh_inv(gamma):
    return 0.5 * np.log( (1 + gamma) / (1 - gamma) )

def tanh(z):
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

def sigma_without_n(n1,n2,n3,n4):
    return 1/4 * np.sqrt(1/n1 + 1/n2 + 1/n3 + 1/n4)

def CI(epsilon, gamma, n1,n2,n3,n4):
    z = norm.ppf(1 - (epsilon / 2) )
    left = tanh( tanh_inv(gamma) - z * sigma_without_n(n1,n2,n3,n4))
    right = tanh( tanh_inv(gamma) + z * sigma_without_n(n1,n2,n3,n4))
    
    return left, right


def corr(df, width=500, height=500, ylable = ''):
    df['color'] = df['top'].apply(lambda x: dict_pallets[3] if x >=0 else dict_pallets[4])

    p1 = figure(x_range=df['MFI id'], width=width, height=height)
    p1.yaxis.axis_label = ylable
    
    p1.vbar(x='MFI id', top='top', width=.45, color='color', source=df, fill_alpha=1,
           )
    
    p1.line(x = df['MFI id'], y = [1]*len(df['MFI id']), color='black', line_dash = '4 4', line_width=.4)
    p1.line(x = df['MFI id'], y = [-1]*len(df['MFI id']), color='black', line_dash = '4 4', line_width=.4)
    p1.line(x = df['MFI id'], y = [0]*len(df['MFI id']), color='black', line_width=.4)

    df['color_CI_left'] = df['left'].apply(lambda x: dict_pallets[6] if x == df['left'].max() else "")
    df['color_CI_right'] = df['right'].apply(lambda x: dict_pallets[10] if x == df['right'].min() else "")
    df['color_CI'] = df['color_CI_left'] + df['color_CI_right']
    df['color_CI'] = df['color_CI'].apply(lambda x: x if x != "" else "black")
    
    p1.segment(df['MFI id'], df['left'], df['MFI id'], df['right'], line_color=df['color_CI'], line_width=.1)
    p1.rect(df['MFI id'], df['left'], width=.3, height=.00001, line_width=.01, line_color=df['color_CI'], line_alpha=1, fill_alpha=1)
    p1.rect(df['MFI id'], df['right'], width=.3, height=.00001, line_width=.01, line_color=df['color_CI'], line_alpha=1, fill_alpha=1)
    

    p1.xaxis.major_label_text_font_size = '6pt'
    p1.yaxis.major_label_text_font_size = '4pt'
    p1.xaxis.axis_label_text_font_size = "2pt"
    p1.yaxis.axis_label_text_font_size = "5pt"
    
    p1.xaxis.major_tick_line_width=.4
    p1.xaxis.axis_line_width = .4
    p1.yaxis.major_tick_line_width=.4
    p1.yaxis.axis_line_width = .4
    
    p1.y_range.start = -1.1
    p1.y_range.end = 1.1
    p1.x_range.range_padding = 0.1
    p1.grid.grid_line_color = None
    p1.axis.minor_tick_line_color = None
    p1.outline_line_color = None
    p1.xaxis.major_label_orientation = np.pi/4
    p1.background_fill_color=None 
    p1.border_fill_color=None
    p1.output_backend = "svg"
    
    print('left_max = ', df['left'].max())
    print('right_min = ', df['right'].min())
     
    return p1



# AB test
# process of receiving requests in time for different ranks (AB_comp_page_clicks, AB_comp_page_clicks)
# mode = 'cl' or 'conv'
def AB_comp_page(df, width=800, height=500, mode = 'cl'):

    # TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"
    if mode == 'cl':
        y_axis_label = 'Number of applications'
    if mode == 'conv':
        y_axis_label = 'Number of sales'
    
    p1 = figure(x_axis_label='Time',
                y_axis_label=y_axis_label,
                x_axis_type="datetime",
                width=width, height=height, 
                )
    k = 1
    for p in np.sort(df['AB_test_page_key'].unique()):
        df_page_v = df[(df['AB_test_page_key'] == p)  &  (df['AB_test_by_rank']  == False)].sort_values('click time').copy()
        if mode == 'cl':
            df_page_v['Count'] = [i+1 for i in range(len(df_page_v))]
        if mode == 'conv':
            df_page_v['Count'] = df_page_v.Sales.cumsum()
        x = df_page_v['click time']
        y = df_page_v['Count']
        p1.line(x, y, legend_label='AW ' + p, color=dict_pallets[k], line_width=2)
        
        df_page_m = df[(df['AB_test_page_key'] == p)  &  (df['AB_test_by_rank']  == True)].sort_values('click time').copy()
        if mode == 'cl':
            df_page_m['Count'] = [i+1 for i in range(len(df_page_m))]
        if mode == 'conv':
            df_page_m['Count'] = df_page_m.Sales.cumsum()
        x = df_page_m['click time']
        y = df_page_m['Count']
        p1.line(x, y, legend_label='Bs ' + p, color=dict_pallets[k], line_width=2,line_dash="4 4")
        
        k = k + 2

    
    
    p1.legend.title = 'Algorithm & Page'
    p1.legend.location = "top_left"
    # p1.legend.orientation = "horizontal"
    p1.legend.background_fill_alpha = 0 
    p1.legend.label_text_font_size = '10pt'
    p1.legend.spacing = 1
    p1.legend.margin = 0

    p1.xaxis.major_label_text_font_size = '14pt'
    p1.yaxis.major_label_text_font_size = '14pt'
    p1.xaxis.axis_label_text_font_size = "14pt"
    p1.yaxis.axis_label_text_font_size = "14pt"
    p1.x_range.range_padding = 0.001
    
    p1.background_fill_color=None 
    p1.border_fill_color=None
    p1.output_backend = "svg"
    
    return p1



# results_with_comparison_method_AB_5page_1
def comp_graph_AB_by_day(df,width=500, height=450):

    p5 = figure(x_axis_type="datetime", 
                width=width, height=height
               )
    p5.yaxis.axis_label = 'Daily income by click (blue)'

    
    p5.extra_y_ranges = {"Share": Range1d(start=0, end=0.2)}
    ax2_5 = LinearAxis(y_range_name="Share", axis_label='Daily share by click (red)')
    p5.add_layout(ax2_5, 'right')

    
    line_glyph1 = Line(x='date_click',
            line_color='black', line_width=3, 
           )
    line1 = p5.add_glyph(ColumnDataSource(df), line_glyph1)
    
    line_glyph2 = Line(x='date_click',
            line_color='black', line_width=3,
            line_dash="4 4")
    line2 = p5.add_glyph(ColumnDataSource(df), line_glyph2)
    
    
    p5.line(x=df[df['AB_test_by_rank'] == False]['date_click'], y=df[df['AB_test_by_rank'] == False]['Income by click'],
            color=dict_pallets[6], line_width=3,
           )
    p5.line(x=df[df['AB_test_by_rank'] == True]['date_click'], y=df[df['AB_test_by_rank'] == True]['Income by click'],
            color=dict_pallets[6], line_width=3,
            line_dash="4 4")
    
    
    p5.line(x=df[df['AB_test_by_rank'] == False]['date_click'], y=df[df['AB_test_by_rank'] == False]['Share by click'], 
            color=dict_pallets[4], line_width=3,
            y_range_name='Share',
           line_alpha=0.7)
    p5.line(x=df[df['AB_test_by_rank'] == True]['date_click'], y=df[df['AB_test_by_rank'] == True]['Share by click'], 
            color=dict_pallets[4], line_width=3,
            y_range_name='Share', line_dash="4 4",
           line_alpha=0.7)
    
    
    
    legend = Legend(
        items=[("historical data", [line1]), ("baseline algorithm", [line2])],
        location='top_left'
    )
    p5.add_layout(legend)
    
    p5.yaxis.axis_label_text_font_size = "10pt"
    p5.y_range.start = 0
    p5.y_range.range_padding = 0.02
    p5.xaxis.major_label_orientation = np.pi/6
    p5.xgrid.grid_line_color = None
    p5.axis.minor_tick_line_color = None
    p5.outline_line_color = None
    p5.background_fill_color=None 
    p5.border_fill_color=None
    p5.output_backend = "svg" 

    # p5.yaxis.ticker.max_interval = 1
    

    return p5



# results_with_comparison_method_AB_5page_1
def comp_graph_AB(df,width=500, height=450):

    p5 = figure(
        x_axis_type="datetime", 
                width=width, height=height
               )
    p5.yaxis.axis_label = 'Weekly income by click (blue)'

    
    p5.extra_y_ranges = {"Share": Range1d(start=0, end=0.07)}
    ax2_5 = LinearAxis(y_range_name="Share", axis_label='Weekly share by click (red)')
    p5.add_layout(ax2_5, 'right')

    
    line_glyph1 = Line(x='date_week',
            line_color='black', line_width=3, 
           )
    line1 = p5.add_glyph(ColumnDataSource(df), line_glyph1)
    
    line_glyph2 = Line(x='date_week',
            line_color='black', line_width=3,
            line_dash="4 4")
    line2 = p5.add_glyph(ColumnDataSource(df), line_glyph2)
    
    
    p5.line(x=df[df['AB_test_by_rank'] == False]['date_week'], y=df[df['AB_test_by_rank'] == False]['Income by click'],
            color=dict_pallets[6], line_width=3,
           )
    p5.line(x=df[df['AB_test_by_rank'] == True]['date_week'], y=df[df['AB_test_by_rank'] == True]['Income by click'],
            color=dict_pallets[6], line_width=3,
            line_dash="4 4")
    
    
    p5.line(x=df[df['AB_test_by_rank'] == False]['date_week'], y=df[df['AB_test_by_rank'] == False]['Share by click'], 
            color=dict_pallets[4], line_width=3,
            y_range_name='Share',
           line_alpha=0.7)
    p5.line(x=df[df['AB_test_by_rank'] == True]['date_week'], y=df[df['AB_test_by_rank'] == True]['Share by click'], 
            color=dict_pallets[4], line_width=3,
            y_range_name='Share', line_dash="4 4",
           line_alpha=0.7)
    
    
    
    legend = Legend(
        items=[("historical data", [line1]), ("baseline algorithm", [line2])],
        # location='top_left'
        location=(140,170)
    )
    p5.add_layout(legend)
    
    p5.yaxis.axis_label_text_font_size = "10pt"
    p5.y_range.start = 0
    p5.y_range.range_padding = 0.02
    p5.xaxis.major_label_orientation = np.pi/6
    p5.xgrid.grid_line_color = None
    p5.axis.minor_tick_line_color = None
    p5.outline_line_color = None
    p5.background_fill_color=None
    p5.legend.background_fill_color=None  
    p5.border_fill_color=None
    p5.legend.border_line_color=None
    p5.output_backend = "svg" 

    # p5.yaxis.ticker.max_interval = 1
    

    return p5


def Fig18(df_prod,width=500, height=430):

    
    MFIs = np.array(df_prod['MFI id'])
    on_time = np.array(df_prod['share on time (sale)'])
    not_on_time = 1 - on_time
    # on_time = np.array([1]*len(df_prod['MFI id']))
    # not_on_time = 1 - on_time

    req = ['on time', "not on time"]
    
    
    data2 = {'MFIs' : MFIs,
             'on time'   : on_time,
             'not on time'   : not_on_time
            }
    colors2 = [dict_pallets[3], dict_pallets[4]]



    p1 = figure(x_range=MFIs, width=width, height=height)
    p1.yaxis.axis_label = "Shares"
    

    p1.vbar_stack(req, x='MFIs', width=0.8, color=colors2, source=data2 ,fill_alpha=.8)

    
    p1.xaxis.major_label_text_font_size = '8pt'
    p1.yaxis.major_label_text_font_size = '10pt'
    p1.xaxis.axis_label_text_font_size = "10pt"
    p1.yaxis.axis_label_text_font_size = "10pt"
    # p1.y_range.start = 0
    p1.x_range.range_padding = 0.01
    p1.grid.grid_line_color = None
    p1.axis.minor_tick_line_color = None
    p1.outline_line_color = None
    p1.xaxis.major_label_orientation = np.pi/2
    p1.background_fill_color=None 
    p1.border_fill_color=None
    p1.output_backend = "svg"
     
    return p1







def expir_id(df):
    post_tmp = df.copy()
    post_tmp['status'] = post_tmp['status'].map({'sale':1,
                                              'rejected':0,
                                              'lead':0})

    post_tmp['VcMfo'] = post_tmp['client id'] + post_tmp['MFI id']

    expir_id = post_tmp.groupby('VcMfo')['status'].sum() / post_tmp.groupby('VcMfo').size()
    expir_id = expir_id.reset_index(name = 'expir_id')

    dict_expir_id = dict(zip(expir_id['VcMfo'], expir_id['expir_id']))
    
    return dict_expir_id
    
def subid5(df, col):
    clr1 = np.sort(df[col].dropna().unique())
    clr2 = [i + 1 for i in range(len(clr1))]
    dict_clr = dict(zip(clr1, clr2))
    df[col] = df[col].map(dict_clr)
    return 0

def add_col(df_, col, card):
    df = df_.copy()
    df.rename(columns={col: 'val'}, inplace=True)
    df['meth'] = col
    df['card'] = card
    return df

def tmp_drow_com(list_df):
    tmp_df = list_df[0][list_df[0]['meth'].isin(['inc', 'shr'])].copy()
    tmp_df['meth_char'] = tmp_df['meth']
    list_tmp = [tmp_df]
    for df, char in zip(list_df, ['11111','11101','11011','11001','11110']):
        df_tmp = df[~df['meth'].isin(['inc', 'shr'])].copy()
        df_tmp['char'] = char
        df_tmp['meth_char'] = df_tmp['meth'] + df_tmp['char']
        list_tmp.append(df_tmp)
    
        
    return pd.concat(list_tmp)









