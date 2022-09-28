import pandas as pd
import numpy as np
from multiprocessing import Pool
import os 
from constants import device_name,actual_id
from split import split_traces_mseq

def check_polarity(ip_src:str)->int:
    """Check the direction of a packet

    Args:
        ip_src (str): Source IP Address

    Returns:
        int: -1 for incoming packets, 1 for outgoing packets
    """
    if ip_src.startswith('192'):
        return 1
    else:
        return -1

def omega_num_helper(curr_times,omega):
    # Helper Function
    curr_times = np.array(curr_times)
    return (curr_times-curr_times[0])//omega

def process_row(row):
    # Helper Function
    if row[1] == -1:
        row[1],row[3],row[4] = 1,0,0
    elif row[1] == 1:
        row[1],row[2] = 0,0
    return row

def group_by_first_col(arr):
    # Helper Function
    return np.split(arr[:,1:], np.unique(arr[:, 0], return_index=True)[1][1:])

def process_feature(feat_vect):
    # Helper Function to process one entry
    unique,ind = np.unique(feat_vect[:, 0], return_index=True)
    missing = [i for i in range(40) if i not in unique]
    if len(missing)>0:
        missing_entries = np.array([np.array([i,0,0]) for i in missing])
        feat_vect = np.vstack([feat_vect,missing_entries])
        feat_vect = feat_vect[feat_vect[:, 0].argsort()]
    
    feat_vect = np.hstack([feat_vect,feat_vect[:,1:]])
    feat_vect = np.apply_along_axis(process_row, 1, feat_vect)
    feat_list = group_by_first_col(feat_vect)
    
    return np.array(list(map(lambda x:np.sum(x,0),feat_list)))

def process_csv(experiment_tuple,total_time=4,omega=0.1):
    day,device = experiment_tuple

    file_name = 'data/split-trace/day_'+str(day)+'_'+str(device)+'.csv'
    full = pd.read_csv(file_name)
    full['dir'] = full['ip.src'].apply(lambda x: check_polarity(x))
    full['frame.time_relative'] -= full['frame.time_relative'].iloc[0]
    full['tcp.len'].replace(-1,0,inplace = True)
    full['udp.length'].replace(-1,0,inplace = True)
    full['packetlen'] = full['tcp.len'] + full['udp.length']
    full = full[['dir','packetlen','frame.time_relative']]
    full = full.to_numpy()

    feature_num = list()
    omega_num = list()
    silences = [0]

    start_time,prev_time,curr_feature_num,curr_times = 0,0,0,list()
    for curr_time in full[:,-1]:
        if curr_time-start_time>=total_time:
            silence = curr_time-prev_time
            silences.append(silence)
            
            curr_feature_num+=1
            start_time = curr_time
            
            omega_num.append(omega_num_helper(curr_times,omega))
            curr_times = list()
            
        prev_time = curr_time
        feature_num.append(curr_feature_num)
        curr_times.append(curr_time)

    if len(curr_times)!=0:
        omega_num.append(omega_num_helper(curr_times,omega))
        curr_times = list()
        
    feature_num = np.array(feature_num)
    silences = np.array(silences)
    omega_num = np.hstack(omega_num)
    
    full = np.delete(full,2,1)
    full = np.hstack([feature_num.reshape(-1,1),omega_num.reshape(-1,1),full])
    split_features = group_by_first_col(full)
    
    features_main = np.array(list(map(lambda x:process_feature(x),split_features)))
    labels = np.ones_like(silences)*actual_id[device]
        
    np.save('data/Traces_Mseq/features_day_'+str(day)+'_'+str(device)+'.npy',features_main)
    np.save('data/Traces_Mseq/silences_day_'+str(day)+'_'+str(device)+'.npy',silences)
    np.save('data/Traces_Mseq/labels_day_'+str(day)+'_'+str(device)+'.npy',labels)
    
    return None

def collate_traces():    
    main_file = 'data/Traces_Mseq'

    experiments_all = []
    for day in range(21):
        for device in device_name:
            file_name = main_file + '/features_day_'+str(day)+'_'+str(device)+'.npy'
            if os.path.isfile(file_name):
                experiments_all.append((day,device))

    (day,device) = experiments_all[0]

    combined_main_features = np.load(main_file+'/features_day_'+str(day)+'_'+str(device)+'.npy')
    combined_silences = np.load(main_file+'/silences_day_'+str(day)+'_'+str(device)+'.npy')
    combined_labels = np.load(main_file+'/labels_day_'+str(day)+'_'+str(device)+'.npy')

    for (day,device) in experiments_all[1:]:
        features_main = np.load(main_file+'/features_day_'+str(day)+'_'+str(device)+'.npy')
        silences = np.load(main_file+'/silences_day_'+str(day)+'_'+str(device)+'.npy')
        labels = np.load(main_file+'/labels_day_'+str(day)+'_'+str(device)+'.npy')

        if features_main.shape[0]>0:
            combined_main_features = np.concatenate((combined_main_features,features_main))
            combined_silences = np.concatenate((combined_silences,silences))
            combined_labels = np.concatenate((combined_labels,labels))

    idx_new = np.random.permutation(combined_main_features.shape[0])
    combined_main_features,combined_silences,combined_labels = combined_main_features[idx_new], combined_silences[idx_new], combined_labels[idx_new]
    
    np.save(main_file + '/new_features_main.npy',combined_main_features)
    np.save(main_file + '/new_silences.npy',combined_silences)
    np.save(main_file + '/new_labels.npy',combined_labels)
    
    for day in range(21):
        for device in device_name:
            if os.path.isfile(main_file+'/features_day_'+str(day)+'_'+str(device)+'.npy'):
                os.remove(main_file+'/features_day_'+str(day)+'_'+str(device)+'.npy')
                os.remove(main_file+'/silences_day_'+str(day)+'_'+str(device)+'.npy')
                os.remove(main_file+'/labels_day_'+str(day)+'_'+str(device)+'.npy')
    
    return None


if __name__ == '__main__':    
    if len(os.listdir('data/split-trace')) <= 25:
        print("Splitting original traffic files......")
        split_traces_mseq()

    experiments_all = []
    for day in range(21):
        for device in device_name:
            file_name = 'data/split-trace/day_'+str(day)+'_'+str(device)+'.csv'
            if os.path.isfile(file_name):
                experiments_all.append((day,device))


    print("Generating feature vectors......")

    pool = Pool(6)
    pool.map(process_csv, experiments_all)
    pool.close()
    pool.join()

    print("Collating feature vectors......")

    collate_traces()