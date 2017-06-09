import caffe
import numpy as np
from numpy import linalg as LA
import argparse
import pprint
import scipy.io as scio
import os


def Programpause():
    inputstr = raw_input('Press e to exit, Press other key to continue!!!\n')
    if (inputstr == 'e'):
        os._exit(0)


class FeaturesLayer(caffe.Layer):

    @classmethod
    def parse_args(cls, argsStr):
        parser = argparse.ArgumentParser(description='FeaturesLayer')
        parser.add_argument('--prefix', default='', type=str)
        parser.add_argument('--total_sum_num', default=0, type=int)
        args = parser.parse_args(argsStr.split())
        print('Using Config:')
        pprint.pprint(args)
        return args

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute accuracy!!!")
        elif bottom[0].data.shape[0] != bottom[1].data.shape[0]:
            raise Exception(
                'The first dimension of two imputs should be equal!!!')
        self.params_ = FeaturesLayer.parse_args(self.param_str)
        self.total_sum_num = self.params_.total_sum_num
        self.prefix = self.params_.prefix
        self.icount = 0
        self.testcount = 0
        self.totalnum = 0
        self.avgaccuracy = 0.0
        self.num = bottom[0].data.shape[0]
        self.channel = bottom[0].data.shape[1]
        self.features = np.zeros((self.total_sum_num, self.channel))
        self.labels = np.zeros((self.total_sum_num,), dtype=int)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        startnum = self.num * self.icount
        for sn in xrange(0, self.num):
            self.features[startnum + sn, :] = bottom[0].data[sn, :]
            self.labels[startnum + sn] = bottom[1].data[sn]
        self.totalnum = self.totalnum + self.num
        if (self.totalnum >= self.total_sum_num):
            fealab = np.zeros((self.total_sum_num, bottom[0].data.shape[1] + 1))
            for i in range(self.totalnum):
                fealab[i, 0:-1] = self.features[i]
                fealab[i, -1] = self.labels[i]
            matfile = self.prefix + str(self.testcount) + '.mat'
            scio.savemat(matfile, {'features': fealab})
            self.testcount = self.testcount + 1
            self.icount = 0
            self.totalnum = 0
        else:
            self.icount = self.icount + 1
        # print 'postiterations: ', self.icount, '\n'

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
