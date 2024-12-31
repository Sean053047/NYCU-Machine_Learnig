#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import io
from PIL import Image
from pathlib import Path

import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist
class Recorder:
    def __init__(self, X, labels, output_dir):
        self.output_dir = output_dir
        self.scatters = []
        self.distributions = []
        self.X = X 
        self.labels =labels
    
    def scatter(self, Y, title:str=''):
        classes = np.unique(self.labels)
        colormap = plt.get_cmap("Spectral", len(classes))
        for cls in classes:
            plt.scatter(Y[self.labels==cls, 0], Y[self.labels==cls, 1], c=[colormap(cls)], alpha=1.0, label=cls)
        buf = io.BytesIO()
        plt.title(title)
        plt.legend()
        plt.savefig(buf, format='png')
        fig = Image.open(buf)
        plt.close()        
        return fig
    def distribution(self, Y, title:str=''):
        N = len(Y)
        y_dist = cdist(Y, Y, metric='euclidean')
        x_dist = cdist(self.X, self.X, metric='euclidean')
        row, col = np.triu_indices_from(y_dist)
        valid = row != col
        row = row[valid]; col = col[valid]
        y_dist = y_dist[row, col]        
        x_dist = x_dist[row, col]
        # x_dist = x_dist.flatten()
        # y_dist = y_dist.flatten()
        plt.hist(y_dist, bins=100, alpha=0.5, label='Low dim distance', color='red', density=True)
        plt.hist(x_dist, bins=100, alpha=0.5, label='High dim distance', color='green', density=True)
        plt.title(title)
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        fig = Image.open(buf)
        plt.close()        
        return fig
        
    def __call__(self, Y, title:str=''):
        self.scatters.append( self.scatter(Y, title))
        self.distributions.append(self.distribution(Y, title))
    @staticmethod
    def convert_pil_image(pil_img):
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def save(self,  perplexity:float):
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        names = ('scatter', 'distribution')
        for frames, nn in zip( (self.scatters, self.distributions), names):
            save_fname = Path(self.output_dir)/ Path(nn+str(perplexity))
            frames[0].save(
                save_fname.with_suffix('.gif'),
                save_all=True,
                append_images=frames[1:], 
                duration=100, 
                loop=0
            )
            height, width = frames[0].size
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vr = cv2.VideoWriter(save_fname.with_suffix('.mp4'), fourcc=fourcc, fps=20, frameSize=(height, width))
            for fig in frames:
                vr.write( self.convert_pil_image(fig))
            vr.release()
            cv2.imwrite(save_fname.with_suffix('.png'), self.convert_pil_image(frames[-1]))
        
        self.scatters = []
        self.distributions = []

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    # Calculate square euclidean distance between each pair of points.
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X) 
    
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    print("Preprocessing the data using PCA...")
    X = X - np.mean(X, 0 , keepdims=True)
    _, M = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def main(   mode,
            X=np.array([]), 
            no_dims=2, initial_dims=50, 
            perplexity=30.0, max_iter=1000):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    assert mode =='tsne' or mode =='ssne', "Wrong mode"
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1
    
    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)
    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        # * Different Q for t-SNE & SSNE
        if mode == 'tsne':
            num = 1. / (1. + ((num + sum_Y).T + sum_Y))
        elif mode == 'ssne':
            num = np.exp(-((num + sum_Y).T + sum_Y))
        num[range(n), range(n)] = 0.
        
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        # * Different gradient for t-SNE & SSNE
        for i in range(n):
            if mode == 'tsne':
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), axis=0)
            elif mode == 'ssne':
                dY[i, :] = np.sum(PQ[i, :, np.newaxis] * (Y[i, :] - Y), axis=0)

        # Perform the update
        momentum = initial_momentum if iter < 20 else final_momentum
        
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY

        Y = Y - np.mean(Y, axis=0, keepdims=True)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            KL = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, KL))
        
        if iter % 20 == 0:
            KL = np.sum(P * np.log(P / Q))
            RC(Y, f"{mode} | {iter} iterations | KL: {KL:.4f}")
        
        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    # Return solution
    RC(Y, f"{mode} | {iter} iterations | KL: {KL:.4f}")
    RC.save(perplexity)
    return Y, KL



if __name__ == "__main__":
    import argparse
    parser =argparse.ArgumentParser()
    parser.add_argument('mode', choices=['tsne', 'ssne'])
    parser.add_argument('--iter', action='store_true')
    parser.add_argument('--max_iter', type=int, default=1000)
    args = parser.parse_args()
    max_iter = args.max_iter
    mode =args.mode 
    
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("./mnist/mnist2500_X.txt")
    labels = np.loadtxt("./mnist/mnist2500_labels.txt").astype(np.int32)
    
    RC = Recorder(X, labels, output_dir=f'./{mode}')
    if args.iter:
        Prange = np.r_[20:101:20]
    else:        
        Prange = np.array([20.0])
    for perplexity in Prange:
        Y, KL = main(mode, X, 2, 50, perplexity, max_iter)