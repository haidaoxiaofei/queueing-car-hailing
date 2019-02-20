import sys
import os
import numpy as np
import cPickle as pickle

def build_adj(path, outpath):
	m = {}
	with open(path) as f:
		for line in f:
			sps = line.strip().split()
			source = int(sps[0])
			if source not in m:
				m[source] = [int(i) for i in sps[1:]]
			else:
				m[source] += [int(i) for i in sps[1:]]
	
	min_id = 10000000
	max_id = 0
	for k,v in m.items():
		max_id = max(max_id, k)
		min_id = min(min_id, k)
		for item in v:
			max_id = max(max_id, item)
			min_id = min(min_id, item)

	print 'max_id={}'.format(max_id)
	print 'min_id={}'.format(min_id)
	adj_mat = np.zeros([max_id - min_id + 1, max_id - min_id + 1])
	for k,v in m.items():
		for item in v:
			#print k,v
			adj_mat[k - min_id, item - min_id] = 1

	# check symetry
	for y in range(adj_mat.shape[0]):
		for x in range(y + 1, adj_mat.shape[0]):
			if adj_mat[y][x] != adj_mat[x][y]:
				print 'not symmetry: {} {}, {}->{}[{}], {}->{}][{}]'.format(x, y, x, y, adj_mat[x][y], y, x, adj_mat[y][x])
				adj_mat[y][x] = 1
				adj_mat[x][y] = 1

	np.save('{}-{}-{}'.format(outpath, min_id, max_id), adj_mat)
	print 'finished'

def main():
	build_adj(sys.argv[1], sys.argv[2])
	# data_folder = sys.argv[1]
	# l = os.listdir(data_folder)
	# max_id = 0
	# m = {}
	# for ff in l:
	# 	if ff.endswith('adj'):
	# 		print ff
	# 		print os.path.join(data_folder,ff)
	# 		with open(os.path.join(data_folder,ff)) as f:
	# 			for line in f:
	# 				sps = line.strip().split()
	# 				source = int(sps[0])
	# 				if source not in m:
	# 					m[source] = [int(i) for i in sps[1:]]
	# 				else:
	# 					m[source] += [int(i) for i in sps[1:]]
	# for k,v in m.items():
	# 	max_id = max(max_id, k)
	# 	for item in v:
	# 		max_id = max(max_id, item)

	# print 'max_id={}'.format(max_id)
	# adj_mat = np.zeros([max_id + 1, max_id + 1])
	# for k,v in m.items():
	# 	for item in v:
	# 		print k,v
	# 		adj_mat[k, item] = 1

	# # check symetry
	# for y in range(adj_mat.shape[0]):
	# 	for x in range(y + 1, adj_mat.shape[0]):
	# 		if adj_mat[y][x] != adj_mat[x][y]:
	# 			print 'not symmetry: {} {}, {}->{}[{}], {}->{}][{}]'.format(x, y, x, y, adj_mat[x][y], y, x, adj_mat[y][x])
	# 			adj_mat[y][x] = 1
	# 			adj_mat[x][y] = 1

	# np.save('adj.npy', adj_mat)
	# print 'finished'

if __name__ == '__main__':
	main()
