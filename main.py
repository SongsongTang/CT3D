import numpy as np
import mat73
import time
from scipy.signal import convolve
import matplotlib.pyplot as plt
import scipy.io as sio


class CCBR(object):
    """
        FDK method of Circular Cone-Beam Reconstruction
    """
    def __init__(self):
        
        stime = time.time()
        
        dir = "./"
        self.fname = dir + "data/Circular CBCT_flat_panel_detector.prj"
        self.R = 40     # the distance between source and origin
        self.l = 20     # the total length of virtual detectors or Shepplogan
        self.n = 256    # the number of detector or pixel per row
        self.T = self.l / self.n    # the length of each detector or each pixel
        # the half length of virtual detectors or Shepplogan
        self.hl = (self.l - self.T) / 2
        self.kn = np.linspace(-255, 255, 511)   # the sample of convolve kernel
        raw_data = np.fromfile(self.fname, np.float32)
        # reshape the data as (beta, b, a)
        self.data = raw_data.reshape(360, 256, 256)
        self.pw_data = self.pre_weight()    # pre-weight the data
        # define the convolve kernel
        self.conv_kn = self.rl_filter()[np.newaxis, np.newaxis, :]
        self.conv_data = convolve(self.pw_data, self.conv_kn, mode='same')
 
        # define x, y, z, beta array
        self.x = np.linspace(-self.hl, self.hl, self.n)[np.newaxis, \
                                                        np.newaxis, :]
        self.y = np.linspace(-self.hl, self.hl, self.n)[:, np.newaxis, \
                                                        np.newaxis]
        self.z = np.linspace(self.hl, -self.hl, self.n)[np.newaxis, :, \
                                                        np.newaxis]
        beta_array = np.arange(0, 2 * np.pi, np.pi / 180)
        # initial f_FDK(y, z, x)
        self.f_fdk = np.zeros((self.n, self.n, self.n))
        # integrate f_fdk by beta
        for self.beta in beta_array:
            U, a, b, beta = self.get_Uabbeta_index()
            f = self.R ** 2 / U ** 2 * self.conv_data[beta, b, a] * np.pi / 180
            self.f_fdk += f
            print("\r Progress: " + str(np.around((self.beta+np.pi/180) / \
                                                  np.pi*50, 1)) + "%", end='')
        # transpose to f(z, y, x) corresponding to Shepplogan
        self.f_fdk = (1 / 2) * self.f_fdk.transpose((1, 0, 2))
        # save as .mat file
        mdic = {"ccbr_fdk_shepplogan": self.f_fdk}
        sio.savemat(dir + "data/ccbr_fdk_shepplogan.mat", mdic)
 
        etime = time.time()
        print("\n Time: " + str(np.around(etime - stime, 1)) + "s!")
 
    def pre_weight(self):
        # define a, b array
        a_array = np.linspace(-self.hl, self.hl, self.n)[np.newaxis, \
                                                         np.newaxis, :]
        b_array = np.linspace(-self.hl, self.hl, self.n)[np.newaxis, :, \
                                                         np.newaxis]
        pre_weighting_factor = self.R / np.sqrt(self.R ** 2 + a_array ** 2 + \
                                                b_array ** 2)
        pre_weighted_data = pre_weighting_factor * self.data
        return pre_weighted_data
    
    def rl_filter(self):
        # define R-L filter
        f = np.zeros(np.shape(self.kn))
        j = 0
        for i in self.kn:
            i = int(i)
            if i == 0:
                f[j] = 1/(2*self.T)**2
            elif i%2:
                f[j] = -1/(i*np.pi*self.T)**2
            else:
                f[j] = 0
            j += 1
        return f
    
    def get_Uabbeta_index(self):
        """Round to nearest"""
        # calculate U
        U = self.R + self.x * np.cos(self.beta) + self.y * np.sin(self.beta)
        # calculate a, b
        a = self.R / U * (-self.x * np.sin(self.beta) + self.y * \
                          np.cos(self.beta))
        b = self.R * self.z / U
        a[a > self.hl] = self.hl
        a[a < -self.hl] = -self.hl
        b[b > self.hl] = self.hl
        b[b < -self.hl] = -self.hl
        a_around = (np.around((a - self.T / 2) / self.T) * self.T + self.T / 2)
        b_around = (np.around((b - self.T / 2) / self.T) * self.T + self.T / 2)
        a_index = ((self.hl + a_around) / self.T).astype(int)
        b_index = ((self.hl + b_around) / self.T).astype(int)
        beta_index = int(np.around(self.beta*180/np.pi))
        return U, a_index, b_index, beta_index


class Analysis(object):
    def __init__(self):
        dir = "./"
        original = mat73.loadmat(dir + 'data/Shepplogan.mat')['Shepplogan']
        fdk_data = sio.loadmat(dir + 'data/ccbr_fdk_shepplogan.mat')\
        ['ccbr_fdk_shepplogan']
        plt.figure("ori, y=-2.5")
        plt.imshow(original[:, 95, :], vmin=0.98, vmax=1.05, cmap='gray')
        plt.colorbar()
        # plt.savefig("oriy-25.png")
 
        plt.figure('ccbr, y=-2.5')
        plt.imshow(fdk_data[:, 95, :], vmin=1.25, vmax=1.35, cmap='gray')
        plt.colorbar()
        # plt.savefig("ccbry-25.png")
        plt.show()


def main():
    # c = CCBR()
    a = Analysis()


if __name__ == '__main__':
    main()