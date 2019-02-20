import sys
from datetime import datetime
import numpy as np
import h5py
from tqdm import tqdm

'''
This code's input: yellow taxi data.
columns:
VendorID,tpep_pickup_datetime,tpep_dropoff_datetime,passenger_count,trip_distance,
RatecodeID,store_and_fwd_flag,PULocationID,DOLocationID,payment_type,
fare_amount,extra,mta_tax,tip_amount,tolls_amount,
improvement_surcharge,total_amount

we only care about 4:
starttime(col 1)
endtime(col 2)
startzone(col 7)
endzone(col 8)

This code's input: green taxi data.
columns:
VendorID,lpep_pickup_datetime,lpep_dropoff_datetime,
store_and_fwd_flag,RatecodeID,PULocationID,DOLocationID,
passenger_count,trip_distance,fare_amount,extra,mta_tax,
tip_amount,tolls_amount,ehail_fee,improvement_surcharge,
total_amount,payment_type,trip_type

we only care about 4:
starttime(col 1)
endtime(col 2)
startzone(col 5)
endzone(col 6)
'''
def fetch_line_info(sps, kind):
	if kind == 0:
		# yellow
		return sps[1], sps[2], sps[7], sps[8]
	elif kind == 1:
		# green
		return sps[1], sps[2], sps[5], sps[6]

days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# acc_day = days[31]
# for i in range(1, len(days)):
#   acc_day.append(days[i] + acc_day[-1])

def main():
    if len(sys.argv) < 6:
        print('usage: data_path out_path year month adj_path [type,default is yellow]')
        sys.exit(1)
    data_path = sys.argv[1]
    out_path = sys.argv[2]
    year = sys.argv[3]
    month = sys.argv[4]
    day_in_month = days[int(month) - 1]
    adj_path = sys.argv[5]
    if len(sys.argv) > 6:
    	kind = 1
    else:
    	kind = 0

    cols = adj_path[:-4].split('-')
    min_id = int(cols[-2])
    max_id = int(cols[-1])
    grid_len = max_id - min_id + 1

    # m is multi-map: day-timezone-flow-grid
    m = {}
    zoneset = set()
    with open(data_path) as f:
        f.readline()
        f.readline()
        for line in tqdm(f):
            sps = line.strip().split(',')
            sttime_str, edtime_str, stzone_str, edzone_str = fetch_line_info(sps, kind)
            starttime = datetime.strptime(sttime_str,'%Y-%m-%d %H:%M:%S')
            endtime = datetime.strptime(edtime_str,'%Y-%m-%d %H:%M:%S')
            duration = endtime - starttime
            # delete record whose duration is less than 3 minutes
            if duration.days == 0 and duration.seconds < 180:
                continue

            startday = '{}{}{:0>2d}'.format(year, month, starttime.day)
            startzone = starttime.hour * 2 + starttime.minute / 30 + 1
            endday = '{}{}{:0>2d}'.format(year, month, endtime.day)
            endzone = endtime.hour * 2 + endtime.minute / 30 + 1
            startgrid = int(stzone_str) - min_id
            endgrid = int(edzone_str) - min_id

            if startgrid < 0 or startgrid >= grid_len or endgrid < 0 or endgrid >= grid_len:
                continue

            if endday not in m:
                m[endday] = {}
            if endzone not in m[endday]:
                m[endday][endzone] = [[0] * grid_len, [0] * grid_len]
            # print endgrid,min_id, grid_len, max_id
            m[endday][endzone][0][endgrid] += 1

            if startday not in m:
                m[startday] = {}
            if startzone not in m[startday]:
                m[startday][startzone] = [[0] * grid_len, [0] * grid_len]
            m[startday][startzone][1][startgrid] += 1

            # if startgrid not in zoneset:
            #     zoneset.add(startgrid)
            # if endgrid not in zoneset:
            #     zoneset.add(endgrid)

    # zonelist = list(zoneset)
    # zonelist.sort()
    # zone2idx = {val:i for i, val in enumerate(zonelist)}
    # print zone2idx
    daylen = len(m)
    data = []
    timestamp = []
    # for each day
    for i in range(1,day_in_month + 1):
        day = '{}{}{:0>2d}'.format(year, month, i)
        if day in m:
            dayflow = []
            # for each tiemzone
            for k,v in m[day].items():
                # timeflow = np.zeros([2, len(zonelist)])
                # print timeflow.shape
                # for j in m[day][k][0]: # in flow
                #     timeflow[0][zone2idx[j]] += 1
                # for j in m[day][k][1]: # out flow
                #     timeflow[1][zone2idx[j]] += 1
                dayflow.append((k, np.array(v)))
            dayflow.sort(key=lambda x: x[0])
            for item in dayflow:
                timestamp.append('{}{:0>2d}'.format(day, item[0]))
                data.append(item[1])

    data = np.array(data)
    timestamp = np.array(timestamp)
    f = h5py.File(out_path,"w")
    f['data'] = data
    f['date'] = timestamp

    f.close()
    print('data shape:{}'.format(data.shape))
    print('timestamp shape:{}'.format(timestamp.shape))


if __name__ == '__main__':
    main()
