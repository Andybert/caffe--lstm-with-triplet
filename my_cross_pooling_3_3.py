import caffe
import numpy as np
import yaml
import argparse
import pprint
import scipy.io as scio
import os


def Programpause():
    inputstr = raw_input('Press e to exit, Press other key to continue!!!\n')
    if (inputstr == 'e'):
        os._exit(0)


class BufferPoolingLayer(caffe.Layer):

    @classmethod
    def parse_args(cls, argsStr):
        parser = argparse.ArgumentParser(description='BufferPoolingLayer')
        '''
        buffersize   : the number of frames in each buffer
        pool  : the method of pooling, mean or max
        '''
        parser.add_argument('--buffersize', default=0, type=int)
        parser.add_argument('--pool', default='mean', type=str)
        args = parser.parse_args(argsStr.split())
        print('Using Config:')
        pprint.pprint(args)
        return args

    def setup(self, bottom, top):
        self.params_ = BufferPoolingLayer.parse_args(self.param_str)
        self.buffersize = self.params_.buffersize
        self.pool = self.params_.pool
        if (self.buffersize > 0):
            if (bottom[0].data.shape[0] != self.buffersize):
                print 'bottom[0].data shape: ', bottom[0].data.shape
                raise Exception(
                    'bottom[0].data.shape[0] is not equal to buffersize, please check it!!!')
        else:
            raise Exception(
                'buffersize is smaller than 1, please check it!!!')
        if (len(bottom) != 2):
            raise Exception(
                'Cross pooling layer needs two imputs, data and label!!!')
        else:
            if (bottom[0].data.shape[0] != bottom[1].data.shape[0] or bottom[0].data.shape[1] != bottom[1].data.shape[1]):
                raise Exception(
                    'The two inputs for cross pooling layer do not meet the condition!!!')

    def reshape(self, bottom, top):
        self.fea_len = bottom[0].data.shape[2]
        self.batch_num = bottom[0].data.shape[0] * bottom[0].data.shape[1]
        self.person_num = bottom[0].data.shape[1]
        top[0].reshape(self.person_num, self.fea_len)
        top[1].reshape(self.person_num)
        if (self.pool == 'max'):
            self.max_idx_ = np.zeros([self.person_num, self.fea_len])

    def forward(self, bottom, top):
        if (self.pool == 'mean'):
            for pn in xrange(0, self.person_num):
                top[1].data[pn] = bottom[1].data[0][pn]
                for fl in xrange(0, self.fea_len):
                    top[0].data[pn][fl] = np.mean(bottom[0].data[:, pn, fl])
        elif (self.pool == 'max'):
            for pn in xrange(0, self.person_num):
                top[1].data[pn] = bottom[1].data[0][pn]
                for fl in xrange(0, self.fea_len):
                    top[0].data[pn][fl] = bottom[0].data[0][pn][fl]
                    self.max_idx_[pn, fl] = 0
                    for image_id in xrange(1, self.buffersize):
                        if (top[0].data[pn][fl] < bottom[0].data[image_id][pn][fl]):
                            top[0].data[pn][fl] = bottom[
                                0].data[image_id][pn][fl]
                            self.max_idx_[pn, fl] = image_id
        else:
            raise Exception(
                'Please set the pool attribute with right value!')

    def backward(self, top, propagate_down, bottom):
        if (self.pool == 'mean'):
            for pn in xrange(0, self.person_num):
                for fl in xrange(0, self.fea_len):
                    mean_value = top[0].diff[pn][fl] / self.buffersize
                    for image_id in xrange(0, self.buffersize):
                        bottom[0].diff[image_id][pn][fl] = mean_value
        elif (self.pool == 'max'):
            for pn in xrange(0, self.person_num):
                for fl in xrange(0, self.fea_len):
                    for image_id in xrange(0, self.buffersize):
                        bottom[0].diff[image_id][pn][fl] = 0
                    bottom[0].diff[self.max_idx_[pn, fl]][
                        pn][fl] = top[0].diff[pn][fl]
        else:
            raise Exception(
                'Please set the pool attribute with right value!')


class ScalePoolingLayer(caffe.Layer):

    @classmethod
    def parse_args(cls, argsStr):
        parser = argparse.ArgumentParser(description='ScalePoolingLayer')
        '''
        scalenum   : the number of scales
        pool: the method of pooling
        '''
        parser.add_argument('--scalenum', default=0, type=int)
        parser.add_argument('--pool', default='mean', type=str)
        parser.add_argument('--phase', default='train', type=str)
        args = parser.parse_args(argsStr.split())
        print('Using Config:')
        pprint.pprint(args)
        return args

    def setup(self, bottom, top):
        self.params_ = ScalePoolingLayer.parse_args(self.param_str)
        self.scalenum = self.params_.scalenum
        self.pool = self.params_.pool
        if (self.scalenum > 0):
            self.bottom_len = len(bottom)
            if (self.bottom_len != (2 * self.scalenum)):
                raise Exception(
                    'The length of bottom is not equal to (2 * scalenum), please check it!!!')
        else:
            raise Exception(
                'The scalenum is smaller than 1, please check it!!!')

    def reshape(self, bottom, top):
        self.person_num = bottom[0].data.shape[0]
        self.fea_len = bottom[0].data.shape[1]
        top[0].reshape(self.person_num, self.fea_len)
        top[1].reshape(self.person_num)
        if (self.pool == 'max'):
            self.max_idx_ = np.zeros(
                [self.person_num, self.fea_len], dtype=int)

    def forward(self, bottom, top):
        if (self.pool == 'mean'):
            for pn in xrange(0, self.person_num):
                top[1].data[pn] = bottom[-1].data[pn]
                for fl in xrange(0, self.fea_len):
                    top[0].data[pn][fl] = 0
                    for s in xrange(0, self.scalenum):
                        top[0].data[pn][fl] = top[0].data[pn][fl] + \
                            bottom[s].data[pn][fl] / self.scalenum
        elif (self.pool == 'max'):
            for pn in xrange(0, self.person_num):
                top[1].data[pn] = bottom[-1].data[pn]
                for fl in xrange(0, self.fea_len):
                    top[0].data[pn][fl] = bottom[0].data[pn][fl]
                    self.max_idx_[pn, fl] = 0
                    for s in xrange(1, self.scalenum):
                        if (top[0].data[pn][fl] < bottom[s].data[pn][fl]):
                            top[0].data[pn][fl] = bottom[s].data[pn][fl]
                            self.max_idx_[pn, fl] = s
            # matfile = '/mnt/68FC8564543F417E/caffe/caffe-master/examples/Pedestrian2/28/scale_pooling_data.mat'
            # scio.savemat(matfile, {'scale_input_data1': bottom[0].data,
            #                        'scale_input_data2': bottom[1].data,
            #                        'scale_input_data3': bottom[2].data,
            #                        'scale_input_data4': bottom[3].data,
            #                        'scale_output_data': top[0].data,
            #                        'scale_input_label': bottom[1].data,
            #                        'scale_output_label': top[1].data})
            # scio.savemat
        else:
            raise Exception(
                'Please set the pool attribute with right value!')

    def backward(self, top, propagate_down, bottom):
        if self.pool == 'mean':
            for pn in range(0, self.person_num):
                for fl in range(0, self.fea_len):
                    mean_value = top[0].diff[pn][fl] / self.scalenum
                    for s in range(self.scalenum):
                        bottom[s].diff[pn][fl] = mean_value
        elif self.pool == 'max':
            for pn in range(0, self.person_num):
                for fl in range(0, self.fea_len):
                    for s in range(self.scalenum):
                        bottom[s].diff[pn][fl] = 0
                    bottom[self.max_idx_[pn, fl]].diff[
                        pn][fl] = top[0].diff[pn][fl]
            # matfile = '/mnt/68FC8564543F417E/caffe/caffe-master/examples/Pedestrian2/28/scale_pooling_diff.mat'
            # scio.savemat(matfile, {'scale_input_diff': top[0].diff,
            #                        'scale_output_diff1': bottom[0].diff,
            #                        'scale_output_diff2': bottom[1].diff,
            #                        'scale_output_diff3': bottom[2].diff,
            #                        'scale_output_diff4': bottom[3].diff})
        else:
            raise Exception(
                'Please set the pool attribute with right value!')
