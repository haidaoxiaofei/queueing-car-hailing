import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error

class HA(object):
    #def __init__(self, regions):
        # self.hist = np.zeros([7 * 48, regions])
        # self.count = np.zeros([7 * 48, 1])

    # def _timestamp2idx(self, t):
    #     year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])-1
    #     day = datetime(year, month, day)
    #     weekday = day.isoweekday() - 1
    #     idx = weekday * 48 + slot
    #     return idx

    # def record(self, data, timestamp):
    #     for d, t in zip(data, timestamp):
    #         idx = self._timestamp2idx(t)
    #         assert idx >= 0 and idx < 7 * 48
    #         self.hist[idx] += d
    #         self.count[idx] += 1

    #     self.hist /= self.count
    #     '''
    #     print('==' * 10)
    #     print(self.hist[0])
    #     print('==' * 10)
    #     '''

    # def infer(self, timestamp):
    #     slot = self._timestamp2idx(timestamp)
    #     return self.hist[slot]

    def eval(self, X, y, mmn):
        Y_pred = np.mean(X, axis=1)
        print(X.shape, Y_pred.shape, y.shape)
        rmse = mean_squared_error(y, Y_pred) ** 0.5
        real_rmse = rmse * (mmn._max - mmn._min) / 2
        print('rmse={}, real rmse={}'.format(rmse, real_rmse))
