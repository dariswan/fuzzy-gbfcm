#Import
import numpy as np
import copy

class FuzzyKohonen:
    """
    Class FuzzyKohonen.
    Kelas ini digunakan untuk melakukan klasterisasi menggunakan metode GBFCM
    """
    def __init__(self, size, W=None, initializer=np.random.randn, alpha=0.1, alpha_decay=0.9):
        """
        init. Konstruktor kelas FuzzyKohonen
       
        @Parameter
        size : jumlah dan bentuk dari titik pusat klaster. [0] = jumlah klaster, [1] = bentuk (fitur) klaster (tupple)
        initializer : peng-inisialisasi nilai titik pusat klaster awal. (object)
        alpha : parameter alpha
        alpha_decay : artefak dari algoritma kohonen
        """
        self.__W = W if W is not None else initializer(size[0], size[1])
        self.__alpha = alpha
        self.__alpha_decay = alpha_decay

    def __u(self, x):
        """
        __u. digunakan untuk menghitung nilai u (derajat keanggotaan)
        
        @Parameter
        x : data (tuple of tuple)

        @Return
        u : nilai u/derajat keanggotaan (tuple of tuple)
        """
        ds = [np.sum(np.power(v-x, 2)) for v in self.__W]
        u = [1/sum([np.power(di/dj, 2) for dj in ds]) for di in ds]
        return u

    def forward(self, X):
        """
        forward. artefak dari algoritma kohonen. mengembalikan nilai crisp dari u.
 
        @Parameter
        X : data (tuple of tuple)

        @Return
        np.argmax(u) : crisp keanggotaan klaster
        """
        u = [self.__u(x) for x in X]
        return np.argmax(u, axis=1)

    def train(self, X):
        """
        train. melakukan pelatihan 1 kali epoch
        
        @Parameter
        X : data (tuple of tuple)

        @Return
        error : float

        """
        u = [self.__u(x) for x in X]
        old_W = copy.deepcopy(self.__W)
        for i in range(len(X)):
            for j in range(len(old_W)):
                self.__W[j] = self.__W[j] - self.__alpha * np.power(u[i][j],2) * (self.__W[j]-X[i])
        error = sum([np.sum(np.power(vk-vkm1, 2)) for vk, vkm1 in zip(self.__W, old_W)])
        return error

    def __call__(self, X, max_epoch=1000, min_err=0.001, verbose=True):
        """
        __call__. melakukan training secara keseluruhan.

        @Parameter
        X : data (tuple of tuple)
        max_epoch : maksimum epoch (int)
        min_error : minimum nilai error (float)
        verbose : menampilkan proses (Boolean)

        @Return
        u : nilai derajat keanggotaan.
        W : titik pusat klaster
        """
        epoch = 0
        error = float("inf")
        while epoch < max_epoch and error > min_err:
            error = self.train(X)
            self.decay()
            epoch += 1
            if verbose:
                print("epoch %d, error %f" % (epoch, error))
        return self.__u(X), self.__W    
   
    def decay(self):
        """
        decay. Artefak dari algoritma kohonen.
   
        """
        self.__alpha = self.__alpha * self.__alpha_decay

    def get_W(self):
        """
        get_W. mengembalikan nilai W (titik pusat klaster)
 
        @Return
        W : titik pusat klaster
        """
        return self.__W
