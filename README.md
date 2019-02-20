# queueing-car-hailing
Source code of queueing-based vehicle dispatching framework


### NewYork Data

http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml


##### 1. build adj mat

`shell
cd preprocess
python newyork_adjacency.py ny.adj ny.adj.npy

`

Download the data set from NewYork Taxi data set and put them in 'dataset/raw_data/*'
