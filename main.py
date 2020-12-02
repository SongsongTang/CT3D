import numpy as np
import time
from scipy.signal import convolve
import mat73
import matplotlib.pyplot as plt
import scipy.io as sio


class FDK(object):
    def __init__(self, fname):

        stime = time.time()

        raw_data = np.fromfile(fname, np.float32)
        self.data = raw_data.reshape(360, 256, 256) # reshape projected data as (beta, a, b)
        
        self.kn = np.linspace(-255, 255, 511)
        self.R = 40
        self.l = 20
        self.n = 256
        self.T = self.l / self.n
        self.b = np.linspace((self.l - self.T) / 2, -(self.l - self.T) / 2, self.n)[np.newaxis, np.newaxis, :]
        self.a = np.linspace((self.l - self.T) / 2, -(self.l - self.T) / 2, self.n)[np.newaxis, :, np.newaxis]
        pre_weighting_factor = self.R / np.sqrt(self.R ** 2 + self.a ** 2 + self.b **2)
        self.pw_data = pre_weighting_factor * self.data
        convolve_kernel = 2 / ((np.pi * self.T) ** 2 * (4 * self.kn ** 2 - 1))[np.newaxis, :, np.newaxis]
        self.f_data = convolve(self.pw_data, convolve_kernel, mode='same')

        self.x = np.linspace(-(self.l - self.T) / 2, (self.l - self.T) / 2, self.n)[:, np.newaxis, np.newaxis]
        self.y = np.linspace(-(self.l - self.T) / 2, (self.l - self.T) / 2, self.n)[np.newaxis, :, np.newaxis]
        self.z = np.linspace(-(self.l - self.T) / 2, (self.l - self.T) / 2, self.n)[np.newaxis, np.newaxis, :]
        self.beta = np.arange(0, 2*np.pi, np.pi/180)
        self.f_fdk = np.zeros((256, 256, 256))
        for beta in self.beta:
            U = self.R + self.x * np.cos(beta) + self.y * np.sin(beta)
            a = self.R / U * (-self.x * np.sin(beta) + self.y * np.cos(beta))
            b = self.R * self.z / U
            a_around = -(np.around((a - self.T / 2) / self.T) * self.T + self.T / 2)
            b_around = -(np.around((b - self.T / 2) / self.T) * self.T + self.T / 2)
            a_around[a_around > ((self.l - self.T) / 2)] = (self.l - self.T) / 2
            a_around[a_around < ((-self.l + self.T) / 2)] = (-self.l + self.T) / 2
            b_around[b_around > ((self.l - self.T) / 2)] = (self.l - self.T) / 2
            b_around[b_around < ((-self.l + self.T) / 2)] = (-self.l + self.T) / 2
            a_index = ((a_around + (self.l - self.T) / 2) / self.T).astype(int)
            b_index = ((b_around + (self.l - self.T) / 2) / self.T).astype(int)
            print(int(np.around(beta*180/np.pi)), end='\t')
            f = self.R**2 / U**2 * self.f_data[int(np.around(beta*180/np.pi)), a_index, b_index]
            self.f_fdk += f
        mdic = {"fdk_shepplogan": self.f_fdk}
        sio.savemat("./data/fdk_shepplogan.mat", mdic)
        etime = time.time()
        print(np.around(etime - stime))


class Analysis(object):
    def __init__(self):
        original = mat73.loadmat('data/Shepplogan.mat')['Shepplogan']
        fdk_data = sio.loadmat('data/fdk_shepplogan.mat')['fdk_shepplogan']
        plt.figure("ori")
        plt.imshow(original[:, 95, :], vmin=0.98, vmax=1.05, cmap='gray')
        plt.colorbar()

        plt.figure('fdk')
        plt.imshow(fdk_data[:, 95, :], vmin=123, vmax=126, cmap='gray')
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    ct = FDK("./data/Circular CBCT_flat_panel_detector.prj")
    a = Analysis()
