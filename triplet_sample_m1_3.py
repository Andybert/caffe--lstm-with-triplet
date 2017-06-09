import caffe
import numpy as np
import config as cfg
import os
import yaml
from collections import defaultdict
import argparse
import pprint


def Programpause():
    inputstr = raw_input('Press e to exit, Press other key to continue!!!\n')
    if (inputstr == 'e'):
        os._exit(0)


class TripletSampleLayer(caffe.Layer):

    @classmethod
    def parse_args(cls, argsStr):
        parser = argparse.ArgumentParser(description='BufferPoolingLayer')
        parser.add_argument('--margin', default=0, type=float)
        parser.add_argument('--max_triplet_num', default=1000, type=int)
        args = parser.parse_args(argsStr.split())
        print('Using Config:')
        pprint.pprint(args)
        return args

    def setup(self, bottom, top):
        self.params_ = TripletSampleLayer.parse_args(self.param_str)
        top[0].reshape(*bottom[0].data.shape)
        top[1].reshape(*bottom[0].data.shape)
        top[2].reshape(*bottom[0].data.shape)
        self.margin = self.params_.margin
        self.max_triplet_num = self.params_.max_triplet_num

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""

        bottom_data = bottom[0].data
        bottom_label = bottom[1].data
        self.index_map = []

        top_anchor = []
        top_positive = []
        top_negative = []

        count_ = 0
        for i in range(bottom[0].num):
            if count_ < self.max_triplet_num:
                for j in range(bottom[0].num):
                    if (j != i and bottom_label[j] == bottom_label[i] and count_ < self.max_triplet_num):
                        for k in range(bottom[0].num):
                            if (k != i and k != j and bottom_label[k] != bottom_label[i] and count_ < self.max_triplet_num):
                                aps = np.sum((bottom_data[i] - bottom_data[j]) ** 2)
                                ans = np.sum((bottom_data[i] - bottom_data[k]) ** 2)
                                dist = self.margin + aps - ans
                                if dist > 0:
                                    top_anchor.append(bottom_data[i])
                                    top_positive.append(bottom_data[j])
                                    top_negative.append(bottom_data[k])
                                    self.index_map.append([i, j, k])
                                    count_ += 1
        top[0].reshape(*np.array(top_anchor).shape)
        top[1].reshape(*np.array(top_anchor).shape)
        top[2].reshape(*np.array(top_anchor).shape)
        top[0].data[...] = np.array(top_anchor)
        top[1].data[...] = np.array(top_positive)
        top[2].data[...] = np.array(top_negative)

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""

        # if propagate_down[0]:
        bottom_diff = np.zeros(bottom[0].data.shape)

        for i in xrange(top[0].num):
            bottom_diff[self.index_map[i][0]] += top[0].diff[i]
            bottom_diff[self.index_map[i][1]] += top[1].diff[i]
            bottom_diff[self.index_map[i][2]] += top[2].diff[i]

        bottom[0].diff[...] = bottom_diff

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
