import caffe
import numpy as np
import yaml
# from utils.timer import Timer
import config as cfg
from collections import defaultdict
import os
import scipy.io as scio
import argparse
import pprint


def Programpause():
    inputstr = raw_input('Press e to exit, Press other key to continue!!!\n')
    if (inputstr == 'e'):
        os._exit(0)


class TripletLayer(caffe.Layer):

    @classmethod
    def parse_args(cls, argsStr):
        parser = argparse.ArgumentParser(description='BufferPoolingLayer')
        parser.add_argument('--margin', default=0, type=float)
        parser.add_argument('--loss_file', default='', type=str)
        args = parser.parse_args(argsStr.split())
        print('Using Config:')
        pprint.pprint(args)
        return args

    def setup(self, bottom, top):
        """Setup the TripletLayer."""
        assert bottom[0].num == bottom[1].num, '{} != {}'.format(
            bottom[0].num, bottom[1].num)
        assert bottom[0].num == bottom[2].num, '{} != {}'.format(
            bottom[0].num, bottom[2].num)
        self.params_ = TripletLayer.parse_args(self.param_str)
        self.margin = self.params_.margin
        self.loss_file = self.params_.loss_file
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        loss = 0
        if bottom[0].data.shape[0] > 0:
            anchor = np.array(bottom[0].data)
            positive = np.array(bottom[1].data)
            negative = np.array(bottom[2].data)
            aps = np.sum((anchor - positive) ** 2, axis=1)
            ans = np.sum((anchor - negative) ** 2, axis=1)
            dist = self.margin + aps - ans
            dist_hinge = np.maximum(dist, 0.0)
            loss = np.sum(dist_hinge) / bottom[0].num
        f = open(self.loss_file, 'a')
        f.write(str(loss) + '\n')
        f.close()
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""
        if bottom[0].data.shape[0] > 0:
            anchor = np.array(bottom[0].data)
            positive = np.array(bottom[1].data)
            negative = np.array(bottom[2].data)

            coeff = 2.0 * top[0].diff / bottom[0].num
            bottom_a = coeff * (negative - positive)
            bottom_p = coeff * (positive - anchor)
            bottom_n = coeff * (anchor - negative)

            bottom[0].diff[...] = bottom_a
            bottom[1].diff[...] = bottom_p
            bottom[2].diff[...] = bottom_n

    def reshape(self, bottom, top):
        pass
