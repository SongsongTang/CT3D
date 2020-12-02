import numpy as np
from scipy.signal import convolve
import mat73
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.io as sio
import time


class FDK(object):
    
    """The FDK Method of 3D Image Reconstruction"""
    
    def __init__(self, fname):
        stime = time.time()
        raw_data = np.fromfile(fname, np.float32)
        # read / write the elements using Fortran-like / MATLAB-like index order
        # self.data = raw_data.reshape(256, 256, 360, order='F')  # reshape the data as (b, a, beta)
        self.data = raw_data.reshape(360, 256, 256) # reshape the data as (beta, a, b)
        # self.data = self.data[:, ::-1, :]
        self.kn = np.linspace(-255, 255, 511)    # convolution kernel
        self.l = 20 # virtual detector length
        self.n = 256    # detector number
        self.T = self.l / self.n   # per detector length
        self.a = np.linspace((-self.l + self.T) / 2, (self.l - self.T) / 2, self.n)[np.newaxis, :, np.newaxis] # a array
        self.b = np.linspace((-self.l + self.T) / 2, (self.l - self.T) / 2, self.n)[np.newaxis, np.newaxis, :] # b array
        # self.b = np.linspace((self.l - self.T) / 2, (-self.l + self.T) / 2, self.n)[:, np.newaxis, np.newaxis] # b array F version
        self.R = 40 # the distance between source and rotational center
        #print(self.R / np.sqrt(self.R ** 2 + self.a ** 2 + self.b ** 2))
        #print(np.shape(self.R / np.sqrt(self.R ** 2 + self.a ** 2 + self.b ** 2)))
        self.x_array = np.linspace((-self.l + self.T) / 2, (self.l - self.T) / 2, self.n)[:, np.newaxis, np.newaxis]
        self.y_array = np.linspace((-self.l + self.T) / 2, (self.l - self.T) / 2, self.n)[np.newaxis, :, np.newaxis]
        self.z_array = np.linspace((-self.l + self.T) / 2, (self.l - self.T) / 2, self.n)[np.newaxis, np.newaxis, :]
        self.beta_array = np.arange(0, 2 * np.pi, np.pi / 180)
        self.rf = self.ramp_filter()
        self.pw_data = self.pre_weighting()
        self.convolved_data = self.convolve3D()
        #print(np.shape(self.convolved_data))
        #print(self.convolved_data[180, 128])
        self.f_fdk = np.zeros((256, 256, 256))
        for beta in self.beta_array:
            U = self.R + self.x_array * np.cos(beta) + self.y_array * np.sin(beta)
            # print("xcosb", self.x_array * np.cos(beta))
            # print("ysinb+", np.shape(self.y_array * np.sin(beta)) , np.shape(self.x_array * np.cos(beta)))
            # print("U", np.shape(U), U)
            a = self.R / U * (-self.x_array * np.sin(beta) + self.y_array * np.cos(beta))
            # print(np.shape(a), a)
            b = self.R / U * self.z_array
            # print(b)
            a_around = (np.around((a - self.T / 2) / self.T) * self.T + self.T / 2)[:, ::-1, :]
            # a_around = (np.around((a - self.T / 2) / self.T) * self.T + self.T / 2)[:, ::-1, :] # F version
            b_around = (np.around((b - self.T / 2) / self.T) * self.T + self.T / 2)[:, :, ::-1]
            # b_around = (np.around((b - self.T / 2) / self.T) * self.T + self.T / 2)[:, ::-1, ::-1]    # F version
            a_around[a_around > ((self.l - self.T) / 2)] = (self.l - self.T) / 2
            a_around[a_around < ((-self.l + self.T) / 2)] = (-self.l + self.T) / 2
            b_around[b_around > ((self.l - self.T) / 2)] = (self.l - self.T) / 2
            b_around[b_around < ((-self.l + self.T) / 2)] = (-self.l + self.T) / 2
            a_index = ((a_around + (self.l - self.T) / 2) / self.T).astype(int)
            b_index = ((b_around + (self.l - self.T) / 2) / self.T).astype(int)
            # np.savetxt("a_index.txt", a_index)
            print(int(np.around(beta*180/np.pi)), end='\t')
            f = self.R**2 / U**2 * self.convolved_data[int(np.around(beta*180/np.pi)), a_index, b_index]
            # f = self.R**2 / U**2 * self.convolved_data[b_index, a_index, int(np.around(beta*180/np.pi))]    # F version
            self.f_fdk += f
        mdic = {"fdk_shepplogan": self.f_fdk}
        sio.savemat("./data/fdk_shepplogan.mat", mdic)
        etime = time.time()
        print(np.around(etime - stime))

    def ramp_filter(self):
        return -2 / ((np.pi * self.T) ** 2 * (4 * self.kn ** 2 - 1))[np.newaxis, :, np.newaxis]

    def pre_weighting(self):
        return self.R * self.data / np.sqrt(self.R ** 2 + self.a ** 2 + self.b ** 2)

    def convolve3D(self):
        return convolve(self.pw_data, self.rf, mode='same', method='auto')


class Analysis(object):
    def __init__(self):
        original = mat73.loadmat('data/Shepplogan.mat')['Shepplogan']
        fdk_data = sio.loadmat('data/fdk_shepplogan.mat')['fdk_shepplogan']
        #print(original)
        plt.figure("ori")
        plt.imshow(original[:, 95, :], vmin=0.98, vmax=1.05, cmap='gray') # , vmin=0.98, vmax=1.05
        plt.colorbar()

        plt.figure('fdk')
        plt.imshow(fdk_data[:, 95, :], vmin=123, vmax=126, cmap='gray')
        plt.colorbar()
        plt.show()
if __name__ == "__main__":
    ct = FDK("./data/Circular CBCT_flat_panel_detector.prj")
    a = Analysis()
