from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from scipy.spatial.distance import cdist

from tqdm import tqdm
from typing import Union
np.random.seed(13)
MAX_ITERATION = 100
def gram_matrix_exp(feature, s, c):
    '''
    s: parameter of spatial information\n
    c: parameter of color information
    '''
    space = feature[:,:2]
    color = feature[:,2:]
    return np.exp(-s*cdist(space, space, 'euclidean')**2) * \
            np.exp(-c*cdist(color, color, 'euclidean') **2)

def gram_matrix_dist(space):
    return cdist(space, space, 'euclidean')**2


def get_feature(image):
    row, col, ch = image.shape
    r, c = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')
    space = np.stack((r,c), axis=2).reshape(-1, 2)
    color = image.reshape(-1, 3) / 255 # normalized
    feature = np.concatenate([space, color], 1)
    return feature 

class Cluster:
    def __init__(self, num_cluster, prior):
        self.num_cluster:int = num_cluster
        self.prior:np.ndarray = prior
        
    def fit(self, feature):
        raise NotImplementedError()
    
    def record_process(self, video_fpth:str, im_shape):
        self.RECORD = True
        self.im_shape = im_shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.__record_video = cv2.VideoWriter(video_fpth, fourcc, 30.0, self.im_shape)
        
    def release_record(self):
        self.RECORD =False
        self.__record_video.release()
        
    @property
    def guess(self):
        return self.__guess
    
    @guess.setter
    def guess(self, value):
        self.__guess = value
        if hasattr(self, "RECORD") and self.RECORD:
            row, col = self.im_shape
            im = np.zeros([row, col, 3])
            colors = [(0, 0, 255), (0, 255, 0)]
            for i, color in enumerate(colors):
                gr = np.where(self.guess==i)[0]
                indexes = self.feature[gr,:2].astype(np.int32)
                im[indexes[:,0], indexes[:,1], :] = color
            self.__record_video.write(im.astype(np.uint8))
    
    def init_guess(self, size):
        if self.prior is not None:
            return self.prior
        else:
            return np.random.randint(0, self.num_cluster, size=size)

class Kmeans(Cluster):
    def __init__(self, num_cluster, gram_matrix, prior=None,**kwargs):
        super().__init__(num_cluster, prior)
        self.gram_matrix = gram_matrix
        
    def fit(self, feature:np.ndarray, *args, **kwargs):
        '''args: s, c'''
        N = len(feature)
        self.feature = feature
        self.guess = self.init_guess(N)
        W = self.gram_matrix(feature,*args, **kwargs)
        self_W = np.diag(W)[:, None]
        for iter in tqdm(range(MAX_ITERATION)):
            # * M step: Calculate objective function    
            cluster_W = np.zeros([1,self.num_cluster]) # Save for summation of each cluster / num_data^2
            to_all_cluster_W = np.zeros((N, self.num_cluster)) # (N, num_cluster): Save summation of gram_matrix value between each data to certain cluster's data
            for i in range(self.num_cluster):
                num_data = np.sum(self.guess == i )
                indexes = np.where( self.guess == i)[0]
                to_all_cluster_W[:, i]  = np.sum(W[:, indexes], axis=1) *2 / num_data
                cluster_W[0, i]= np.sum( W[indexes,:][:, indexes]) / (num_data**2)
            
            # * E step: Assign each clusters
            new_guess =np.argmin(self_W - to_all_cluster_W + cluster_W , axis=1)
            if np.sum(new_guess != self.guess)/N < 0.05 and iter > MAX_ITERATION*0.8: break
            self.guess = new_guess    
        
        self.guess = new_guess
    
    def __repr__(self):
        return self.__class__.__name__

class Spectral(Cluster):
    
    def __init__(self, num_cluster, mode:str, gram_matrix, prior=None, **kwargs):
        super().__init__(num_cluster, prior)
        self.mode = mode # * 'normalized '| 'ratio'
        self.gram_matrix = gram_matrix
        if mode == "ratio":
            self.fit = self.ratio_fit
        elif mode == "normalized":
            self.fit = self.normalized_fit
    
    def ratio_fit(self, feature, s, c):
        N = len(feature)
        self.feature = feature
        self.guess = self.init_guess(N)
        W = self.gram_matrix(feature, s, c)
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        print(L.shape)
        evalue, evector = np.linalg.eig(L)
        
        print(evalue.shape)
        print(evector.shape)
        exit()
    
    def normalized_fit(self, feature, s, c):
        ...
    def __repr__(self):
        return self.__class__.__name__ + f"_{self.mode}"

CLUSTER_SET = Union[Kmeans, Spectral]

def launch_process(image_pth:str, s, c, model_class:CLUSTER_SET, num_cluster, gram_matrix, mode=None, prior=None):
    model_args = {
        'num_cluster': num_cluster,
        'gram_matrix':gram_matrix,
        'mode':mode,
        'prior':prior
        
    }
    model = model_class(**model_args)
    im = np.array(Image.open(image_pth))
    feature = get_feature(im)
    video_fpth = f"{Path(image_pth).stem}_{model}_num_cluster:{model.num_cluster}_s:{s}_c:{c}.mp4" 
    model.record_process(video_fpth=video_fpth, im_shape=im.shape[:2])
    model.fit(feature, s, c)
    model.release_record()

if __name__ == "__main__":
    # np.set_printoptions(threshold=np.inf)
    import argparse
    parser = argparse.ArgumentParser(prog='ML Homework 6')
    parser.add_argument("--mlp", action="store_true")
    args = parser.parse_args()
    
    if not args.mlp:
        for image_pth in ("image1.png", "image2.png"):
            s = 0.5 ; c = 0.05
            launch_process(image_pth, s, c, Spectral, num_cluster=2, mode='ratio', gram_matrix= gram_matrix_exp)
    else:
        from multiprocessing import Pool
        with Pool(10) as pool:
            for image_pth in ("image1.png", "imag2.png"):
                process_args = [ (image_pth, s*0.1, c*0.1, Kmeans, num_cluster) for s in range(1, 11) for c in range(1, 11) for num_cluster in range(1, 5)]
                pool.starmap(launch_process, process_args)
            