import numpy as np
import os,sys
from collections import defaultdict
from tqdm import tqdm
# from datetime import datetime
from datetime import date as date_lib


filename = '../dataset_mood_smartphone.csv'
with open (filename,'r') as f: 
    data = f.readlines()

data = [data[i].replace("'",'').replace('"','').replace('\n','').split(',') for i in range(len(data))]
    
data_header = data[0]
data = data[1:]
print(data_header)
print(data[:10])

data_table = [[],[],[],[],[]]

for i in range(len(data)):
    data_table[0].append(data[i][0]) 
    data_table[1].append(data[i][1]) 
    data_table[2].append(data[i][2]) 
    data_table[3].append(data[i][3]) 
    data_table[4].append(data[i][4])    

# print(data_table[1][:50])
data_table = np.array(data_table)
# meh = data_table[data_table=='AS14.01']
# meh = [data_table[1][i] if data_table[1][i] == 'AS14.01' else continue  for i in range(len(data_table[1])) ] 
# print(len(meh))

# print(list(set(list(data_table[1]))))

# print(len(list(set(list(data_table[1])))))
# for i in range(10):
#     meh = data_table[data_table=='AS14.0'+str(i)]
#     # meh = [data_table[1][i] if data_table[1][i] == 'AS14.01' else continue  for i in range(len(data_table[1])) ] 
#     print(len(meh))

def decode_date(d):
    """
    Decode date into day, season, part of the day
    :param d: str
    :return: (int days, int season, int part_of_the_day)
      days: number of days since 1-1-1
      season: 0 winter, 1 spring, 2 summer, 3 autumn
      part_of_the_day: 0 midnight, 1 morning, 2 afternoon, 3 evening
    """
    date,time = d.split(' ')

    #print(date,time)
    year,month,day = date.split('-')
    year,month,day = int(year),int(month),int(day)
    hours,minutes,seconds = time.split('.')[0].split(':')
    
    hours,minutes,seconds  = int(hours),int(minutes),int(seconds)
    # sub = datetime.fromisoformat('2014-03-05 07:22:26.976637+00:00').timestamp() -datetime.fromisoformat('2014-03-06 07:22:26.976637+00:00').timestamp()
    # print(int(''.join(c for c in '2014-03-04 07:22:26.976637+00:00' if c.isdigit())) - int(''.join(c for c in '2014-03-06 07:22:26.976637+00:00' if c.isdigit())))
    # print(sub)
    
    # 100000*year + 10000*month + 1000 *day + 100 
    # print((date_lib(year,month,day) - date_lib(1,1,1)).days)
    # print((a-b).days)
    date = date_lib(year,month,day)

    part_of_the_day = None
    if hours <6: 
        part_of_the_day = (1,0,0,0)
    elif hours <12:
        part_of_the_day = (0,1,0,0)
    elif hours <18:
        part_of_the_day = (0,0,1,0)
    else:
        part_of_the_day = (0,0,0,1)

    season = None
    if month <= 2  or  month == 12: 
        season = (1,0,0,0)
    elif month < 6:
        season = (0,1,0,0)
    elif month < 9:
        season = (0,0,1,0)
    else:
        season = (0,0,0,1)
        
    return (date_lib(year,month,day) - date_lib(1,1,1)).days , part_of_the_day , season

# print(decode_date('2014-03-04 07:22:26.976'))
# exit()
def group_datapoints_by_id (d_table):
    
    ids = list(set(list(d_table[1])))
    data_dict = defaultdict(list)
    for id in ids:

        # print(id)
        # print(np.where(d_table==id))
        # print(d_table[:,np.where(d_table==id)[1]])
        all_records_of_id = d_table[:,np.where(d_table==id)[1]]
        data_dict[id] = all_records_of_id
    return data_dict
def var_handler(var,val):


    var_ids =[ 'mood'  , 'circumplex.arousal','circumplex.valence' ,'activity','screen' ,'call','sms' ,'appCat.builtin' ,'appCat.communication', 'appCat.entertainment','appCat.finance' ,'appCat.game','appCat.office','appCat.other','appCat.social' ,'appCat.travel' ,'appCat.unknown' ,'appCat.utilities', 'appCat.weather' ]
    idx = var_ids.index(var)
    data = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
    if val == 'NA':
        return data
    val = float(val)
    if idx ==0:
        data[idx] = (val-1)/9.
        return data
    elif idx ==1 or idx == 2 :
        data[idx] = (val+2)/4
        return data
    else:
        data[idx] = val
        return data
    # return 
def group_datapoints_by_day_and_user (d_table):
    
    # dates = list(set(list(d_table[2])))
    # ids = list(set(list(d_table[1])))
    data_dict = defaultdict(None)
    for i , (i,id,d,var,val) in tqdm(enumerate(zip(d_table[0],d_table[1],d_table[2],d_table[3],d_table[4]))):
        # print(i,id,d,var,val)
        # exit()
        var_list = var_handler(var,val)
        
        date_idx,part_of_day,season = decode_date(d)
        mylist =  []
        if id not in data_dict.keys():
            data_dict[id] = defaultdict(list)
        # elif var not in data_dict[id].keys()
        
        for part in part_of_day:
            
            var_list.append(part)
        for s in season:
            
            var_list.append(s)
        
        
        # all_records_of_id = d_table[:,np.where(d_table==id)[1]]
        data_dict[id][date_idx].append(var_list)
    return data_dict
print(data_table.shape)
d_dict_by_id = group_datapoints_by_day_and_user (data_table)
# print(d_dict_by_id[735325])
with open('../text.txt','w') as f:
    for id in enumerate(d_dict_by_id):
        for i,date in d_dict_by_id[id]:
            # print(str(id)+str(d_dict_by_id[date][id])+'\n')
            f.write(str(i)+'_'+str(id)+str(d_dict_by_id[date][id])+'\n')
# --------------------------------------------
# 1. 1 day per datapoint

#             user   season 
# day 
# --------------------------------------------------

# 2. splitting the day per datapoint

#             user   season 
# morning
# noon
# afternoon

# 3. splitting the day for each user per datapoint

#            season 
# morning
# noon
# afternoon




