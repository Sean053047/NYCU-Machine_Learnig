from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.distance import cdist

from tqdm import tqdm
from typing import Union
import matplotlib.pyplot as plt
np.random.seed(50)
MAX_ITERATION = 50
IMG = ''
def gram_matrix_exp(feature1, feature2,  s, c):
    '''
    s: parameter of spatial information\n
    c: parameter of color information
    '''
    space1, color1 = feature1[:,:2], feature1[:,2:]
    space2, color2 = feature2[:,:2], feature2[:,2:] 
    return np.exp(-s*(cdist(space1, space2, 'euclidean')**2)) * \
            np.exp(-c*(cdist(color1, color2, 'euclidean')**2))

def gram_matrix_dist(coor1, coor2):
    return cdist(coor1, coor2, 'euclidean')

def get_feature(image):
    row, col, ch = image.shape
    r, c = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')
    space = np.stack((r,c), axis=2).reshape(-1, 2) 
    space = space / np.max(space)
    color = image.reshape(-1, 3) /255 # normalized
    feature = np.concatenate([space, color], 1)
    return feature 

class Cluster:
    def __init__(self, num_cluster, init_method:str):
        self.num_cluster:int = num_cluster
        self.init_method:str = init_method
        self._guess = None
    def record_process(self, video_fpth:str, image):
        self.RECORD = True
        self.image_template = image
        self.__colors = [ (0, 0, 255), (0,255,0), (255,0,0), (200,200,0), (0,200,200), (205,0,205)]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        im_shape = image.shape[:2]
        self.__record_video = cv2.VideoWriter(video_fpth, fourcc, 30.0, im_shape)

    def release_record(self):
        self.RECORD =False
        self.__record_video.release()
        
    @property
    def guess(self):
        return self._guess
    
    @guess.setter
    def guess(self, value):
        self._guess = value
        if hasattr(self, "RECORD") and self.RECORD:
            # create coordinates.
            im = np.copy(self.image_template)
            im_shape = im.shape[:2]
            for cluster, color in zip(range(self.num_cluster), self.__colors):
                interest = (value == cluster).reshape(im_shape)
                im = np.where( np.repeat(interest[..., np.newaxis], 3, axis=2),
                                0.3* im + 0.7 * np.array(color), 
                                im)
            self.__record_video.write(im.astype(np.uint8))
        
    def terminate_condition(self, tmp_guess, iter):
        N = len(self.guess)
        return np.sum(tmp_guess != self.guess)/N < 0.05 and iter > MAX_ITERATION*0.8 or \
                np.sum(tmp_guess != self.guess) ==0

    def clustering(self, W):
        raise NotImplementedError()
    
    def get_W(self, s, c ):
        raise NotImplementedError()
    
    def init_guess(self, N, **kwargs):
        raise NotImplementedError()
    
    def fit(self, feature, s, c): 
        raise NotImplementedError()
class Kmeans(Cluster):
    def __init__(self, num_cluster,  init_method='random', *args, **kwargs,):
        super().__init__(num_cluster, init_method)
    
    def init_guess(self, N, feature, s, c):
        if self.init_method == 'random':
            self.guess = np.random.randint(0, self.num_cluster, size=N)
        elif self.init_method == 'kmeans++':
            N, col = feature.shape
            # Get centroids
            centroids = [feature[np.random.choice(np.arange(N), size=1)].ravel()]
            for i in range(1, self.num_cluster):
                dist_centroids = gram_matrix_exp( feature, np.array(centroids).reshape(-1, col), s=s, c=c).min(axis=1)
                prob = dist_centroids / np.sum(dist_centroids)
                centroids.append(feature[np.random.choice(np.arange(N), p=prob), :])
            # Update Guess based on centroids
            self.guess = gram_matrix_exp(feature, np.array(centroids).reshape(-1,col), s=s, c=c).argmin(axis=1)
        else:
            raise AssertionError("Wrong init_method.")
        
    def clustering(self, W):
        N = len(W)
        self_W = np.diag(W)[:, None]
        for iter in tqdm(range(MAX_ITERATION), desc="Kernel Kmeans: "):
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
            if self.terminate_condition(new_guess, iter): break
            self.guess = new_guess    
        self.guess = new_guess
        return self.guess
    
    def get_W(self, feature:np.ndarray, s, c):
        '''args: s, c'''
        W = gram_matrix_exp(feature, feature, s, c)
        return W

    def fit(self, feature, s, c, **kwargs):
        W = self.get_W(feature, s, c )
        self.init_guess(N =len(W), feature=feature, s=s, c=c)
        result = self.clustering(W)
        return result, W

    def __repr__(self):
        return self.__class__.__name__

class Spectral(Cluster):
    
    def __init__(self, num_cluster, mode:str, init_method='random', *args, **kwargs,):
        super().__init__(num_cluster, init_method)
        self.mode = mode # * 'normalized '| 'ratio'

    def init_guess(self, N, evector:np.ndarray = None):
        if self.init_method == 'random':
            self.guess = np.random.randint(0, self.num_cluster, size=N)
        elif self.init_method == 'kmeans++' and evector is not None:
            N, col = evector.shape
            # Get centroids
            centroids = [evector[np.random.choice(np.arange(N), size=1)].ravel()]
            for i in range(1, self.num_cluster):
                dist_centroids = gram_matrix_dist( evector ,np.array(centroids).reshape(-1, col)).min(axis=1)
                prob = dist_centroids / np.sum(dist_centroids)
                centroids.append(evector[np.random.choice(np.arange(N), p=prob), :])
            # Update Guess based on centroids
            self.guess = gram_matrix_dist(evector, np.array(centroids).reshape(-1, col)).argmin(axis=1)
        else:
            raise AssertionError("Wrong init_method.")

    def get_W(self, feature, s, c):
        global IMG
        W = gram_matrix_exp(feature, feature, s, c)
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        
        if Path(f'{IMG}_{self.mode}_{s}_{c}_evector.npy').exists() and \
            Path(f'{IMG}_{self.mode}_{s}_{c}_evalue.npy').exists():
            evector = np.load(f'{IMG}_{self.mode}_{s}_{c}_evector.npy')
            evalue = np.load(f'{IMG}_{self.mode}_{s}_{c}_evalue.npy')
        else:
            if self.mode == 'ratio':
                evalue, evector = np.linalg.eig(L)
            elif self.mode == 'normalized':
                D_inv = np.divide(1, D, out=np.zeros_like(D), where= D > 1e-4) 
                evalue, evector = np.linalg.eig( D_inv @ L )
            np.save(f'{IMG}_{self.mode}_{s}_{c}_evector.npy' , evector)
            np.save(f'{IMG}_{self.mode}_{s}_{c}_evalue.npy', evalue)
        close_real = np.isclose(np.imag(evalue), 0)         # Chose eigen value which are closed enough to 0j.
        evalue = evalue[close_real].real
        evector =evector[:, close_real].real
        # Treat those eigen value < 1e-4 ~= 0.0, which is out of constraint. Filter them out.
        evalue_large_enough = ~np.isclose(evalue, 0)
        evalue = evalue[evalue_large_enough]    
        evector = evector[:, evalue_large_enough]
        indx = np.argsort(evalue)
        H = evector[..., indx[:self.num_cluster]]
        return H

    def clustering(self, H):
        for iter in tqdm(range(MAX_ITERATION), desc=f"Spectral Clustering-{self.mode}"):
            centroids = self.get_centroids(H)
            distance = gram_matrix_dist(H, centroids)
            new_guess = np.argmin(distance, axis=1)
            if self.terminate_condition(new_guess, iter): break
            self.guess = new_guess
        self.guess = new_guess
    
    def get_centroids(self, H):
        centroids = []
        for i in range(self.num_cluster):
            ex_indx = np.where(self.guess == i)[0]
            centroids.append(np.mean(H[ex_indx, :], axis=0))
        return np.array(centroids)
    
    def fit(self, feature, s, c, **kwargs):
        W = self.get_W(feature, s, c )
        self.init_guess(N =len(W), evector=W)
        self.clustering(W)
        return self.guess, W
    def __repr__(self):
        return self.__class__.__name__ + f"_{self.mode}"
    
def draw_eigenvector( out_fpth, guess, W):
    colors = [ (0, 0, 255), (0,255,0), (255,0,0)]
    num_cluster = len(np.unique(guess))
    fig = plt.figure()
    ax = fig.add_subplot( projection='3d')
    
    for i, color in zip(range(num_cluster), colors):    
        ex_row = np.where(guess == i)[0]     
        ex_coor = W[ex_row, :] 
        x = ex_coor[:, 0]
        y = ex_coor[:, 1]
        if ex_coor.shape[1] == 2:
            z = np.zeros_like(x)
        elif ex_coor.shape[1] == 3:
            z = ex_coor[:, 2]
        
        ax.scatter(x, y, z, color, label=f'cluster {i}' )
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Eigenvector coordinate")
    plt.savefig(out_fpth)
            
CLUSTER_SET = Union[Kmeans, Spectral]

def launch_process( image_pth:str, 
                    s:float, c:float, 
                    model:CLUSTER_SET, 
                    num_cluster:int,  
                    mode=None, init_method='random',
                    output_folder=''):
    model_args = {
        'num_cluster': num_cluster,
        'mode':mode,
        'init_method':init_method
    }
    model = model(**model_args)
    im = cv2.imread(image_pth)
    feature = get_feature(im)
    
    # method_folder = Path(output_folder) / Path(f"{model}")
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    video_fpth = str(
        output_folder / Path(f"{Path(image_pth).stem}_{model}_{init_method}_num_cluster:{model.num_cluster}_s:{s}_c:{c}.mp4") 
    )
    model.record_process(video_fpth=video_fpth, image=im)
    guess, W = model.fit(feature, s, c,)
    model.release_record()
    if model.__class__ is Spectral and num_cluster <=3:
        out_fpth = str(
            output_folder / Path(f"{Path(image_pth).stem}_{model}_{init_method}_num_cluster:{model.num_cluster}_s:{s}_c:{c}.png") 
        )
        draw_eigenvector(str(out_fpth), guess, W)

def part1(s:float=1.0, c:float=1.0):
    global IMG
    output_folder = './part1'
    for image_pth in ("image1.png", "image2.png"):
        IMG = Path(image_pth).stem        
        launch_process(image_pth=image_pth, s=s, c=c, model=Kmeans, num_cluster=2, output_folder =output_folder)
        launch_process(image_pth=image_pth, s=s, c=c, model=Spectral, num_cluster=2, mode='ratio', output_folder =output_folder)
        launch_process(image_pth=image_pth, s=s, c=c, model=Spectral, num_cluster=2, mode='normalized', output_folder =output_folder)

def part2(s:float=1.0, c:float=1.0):
    global IMG
    output_folder = './part2'
    for image_pth in ("image1.png", "image2.png"):
        IMG = Path(image_pth).stem        
        for nc in range(3, 5):
            launch_process(image_pth=image_pth, s=s, c=c, model=Kmeans, num_cluster=nc, output_folder =output_folder)
            launch_process(image_pth=image_pth, s=s, c=c, model=Spectral, num_cluster=nc, mode='ratio', output_folder=output_folder)
            launch_process(image_pth=image_pth, s=s, c=c, model=Spectral, num_cluster=nc, mode='normalized', output_folder=output_folder)

def part3(s:float=1.0, c:float=1.0):
    global IMG
    output_folder = './part3'
    for image_pth in ('image1.png', "image2.png"):
        IMG = Path(image_pth).stem        
        for init_method in ('kmeans++', 'random'):            
            launch_process(image_pth=image_pth, s=s, c=c, model=Kmeans, num_cluster=2, init_method=init_method, output_folder=output_folder)
            launch_process(image_pth=image_pth, s=s, c=c, model=Spectral, num_cluster=2, mode='ratio', init_method=init_method, output_folder=output_folder)
            launch_process(image_pth=image_pth, s=s, c=c, model=Spectral, num_cluster=2, mode='normalized', init_method=init_method, output_folder=output_folder)
            
# For CLI Usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='ML Homework 6')
    parser.add_argument("option", type=str, default=None)
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--model", type=str, help="kmeans | spectral")
    # For those don't use multiple processing.
    parser.add_argument("--mode", type=str, help="ratio | normalized")
    parser.add_argument("--init_method", type=str, default='random', help='random | kmeans++')
    parser.add_argument("-s", type=float, help="Spatial parameters for RBF", default=.4)
    parser.add_argument("-c", type=float, help="Color parameters for RBF", default=1.0)
    parser.add_argument("--num_cluster", type=int, default=2)
    parser.add_argument("--output_folder", type=str, )
    args = parser.parse_args()
    
    model = Kmeans if args.model == 'kmeans' else Spectral if args.model == 'spectral' else None
    mode = args.mode
    init_method = args.init_method
    s = args.s
    c = args.c
    num_cluster = args.num_cluster
    output_folder = args.output_folder
    
    
    if args.option is not None:
        if args.option == 'part1':
            part1(s, c)
        elif args.option == 'part2':
            part2(s, c)
        elif args.option == 'part3':
            part3(s, c)
        exit()

    
    if not args.mlp:
        for image_pth in ("image1.png", "image2.png"):
            launch_process(image_pth, s, c, model, num_cluster, mode, init_method, output_folder)
    else:
        from multiprocessing import Pool
        with Pool(5) as pool:
            for image_pth in ("image1.png", "imag2.png"):
                process_args = [ (image_pth, s*0.1, c*0.1, model, num_cluster) for s in range(1, 11) for c in range(1, 11) for num_cluster in range(1, 5)]
                pool.starmap(launch_process, process_args)   