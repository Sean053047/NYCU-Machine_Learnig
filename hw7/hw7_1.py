from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Callable

class Kernel:
    def __init__(self, kernel_method, **kernel_kwds):
        self.kernel_method = kernel_method
        for k,v in kernel_kwds.items():
            setattr(self, k, v)
    def __call__(self, x, y ):
        kernel = getattr(self, self.kernel_method)
        return kernel(x, y)
    def linear(self, x, y):
        # x: (n, dim)
        return x @ y.T
    def rbf(self, x, y):
        # Normalized x, y to [0, 1]
        x = x.copy()/255
        y = y.copy()/255
        return np.exp(-self.r * cdist(x,y, 'euclidean')**2 )
    def polynomial(self, x, y):
        return (x@ y.T + self.c) ** self.d  
    def __repr__(self):
        if self.kernel_method == 'linear':
            return self.kernel_method
        elif self.kernel_method == 'polynomial':
            return self.kernel_method + f"_c:{self.c}_d:{self.d}"
        elif self.kernel_method == 'rbf':
            return self.kernel_method + f"_r:{self.r}"
class KNN:
    def __init__(self, nn:int):
        self.nn = nn 
        self.data = None
        self.label = None
        super().__init__()
    def fit(self, x, label):
        self.data = x 
        self.label = label
    def predict(self, test, tf_func:Callable = lambda a:a, *func_args, **func_kwds):
        if 'tgt' in func_kwds:
            tgt= func_kwds['tgt']
            del func_kwds['tgt']
        else:
            tgt = self.data
        src = tf_func(test, *func_args, **func_kwds)
        tgt = tf_func(tgt, *func_args, **func_kwds)
        
        dist = cdist(src, tgt, metric='euclidean')
        # Select nn tgt, which are top nn nearest element to src.
        neighbors = np.argsort(dist, axis=1)[:, :self.nn] 
        # Return the most frequent element
        pred = list()
        for neighbor in neighbors:
            nn = self.label[neighbor]
            ps, counts = np.unique(nn, return_counts=True)
            if len(ps) == 1:
                pred.append(ps[0])
            elif np.all(counts == counts[0]):
                pred.append(-1)
            else:
                pred.append(ps[np.argmax(counts)])
        return np.array(pred)

class DimensionReduction(KNN):
    def __init__(self, nn, kernel_method:str, **kernel_kwds):
        self.w = None
        self.kernel = None if kernel_method is None else Kernel(kernel_method, **kernel_kwds)
        super().__init__(nn)
        
    @property
    def C(self):
        raise NotImplementedError()
    
    def fit(self, x, label):
        super().fit(x, label)
        eig_value, eig_vector = np.linalg.eig(self.C)
        order = np.argsort(-eig_value)
        self.w = eig_vector[:, order].real
        self.w = self.w / np.linalg.norm(self.w, axis=0, keepdims=True)
        return self
    
    def transform(self, x:np.ndarray, num_eig:int):
        # x: (bs, dim)
        w = self.w[:, :num_eig]
        return x @ w
        
    def reconstruct(self, x:np.ndarray, num_eig:int):
        # x: (bs, dim)
        w = self.w[:, :num_eig]
        return x @ w @ w.T # (bs, dim)
    
    def predict(self, x:np.ndarray, num_eig:int):
        if self.kernel is None:
            return super().predict(x, tf_func=self.transform, num_eig=num_eig) 
        else:
            x = self.kernel(x, self.data)        
            tgt = self.kernel(self.data, self.data)
            return super().predict(x, tgt=tgt, tf_func=self.transform, num_eig=num_eig, )

class PCA(DimensionReduction):
    def __init__(self, nn:int, kernel_method:str=None, **kernel_kwds):
        super().__init__(nn, kernel_method, **kernel_kwds)

    @property
    def C(self):
        return self._general_C(self.data) if self.kernel is None else self._kernel_C(self.data)
    
    def _general_C(self, x):
        x_mean = np.mean(x, axis=0, keepdims=True, dtype=np.float64)
        return (x-x_mean).T @ (x-x_mean)
    
    def _kernel_C(self, x):
        N = len(x)
        K = self.kernel(x, x)
        one_N = np.ones_like(K, dtype=np.float64) / N
        return K - one_N @ K - K @ one_N + one_N @ K @ one_N
            
class LDA(DimensionReduction):
    @property
    def C(self):
        return self._general_C(self.data, self.label) if self.kernel is None else self._kernel_C(self.data, self.label)
    
    def _general_C(self, x, label):
        dim = x.shape[1]
        classes=  np.unique(label)
        Cw = np.zeros((dim, dim), dtype=np.float64)
        Cb = np.zeros((dim, dim), dtype=np.float64)
        m = np.mean(x, axis=0, keepdims=True)
        for cls in classes:
            cls_x = x[label==cls, ...]
            num_x = np.sum(label == cls)
            cls_mean = np.mean(cls_x, axis=0, keepdims=True)
            Cw += (cls_x - cls_mean).T @ (cls_x - cls_mean)
            Cb += num_x*((cls_mean - m).T @ (cls_mean - m))
        return np.linalg.pinv(Cw) @ Cb
    
    def _kernel_C(self, x, label):
        N = x.shape[0]
        classes = np.unique(label)
        Cw = np.zeros((N, N), dtype=np.float64)
        Cb = np.zeros((N, N), dtype=np.float64)
        K = self.kernel(x, x)
        M = np.mean( K, axis=0, keepdims=True) # (1, N)
        for cls in classes:
            cls_K = K[label==cls, ...]
            num_x = np.sum(label ==cls)
            cls_M = np.mean( cls_K, axis=0, keepdims=True)
            Cw += cls_K.T @ (np.identity(num_x) - 1/num_x) @ cls_K
            Cb += num_x*((cls_M-M).T @ (cls_M-M))
        
        return np.linalg.pinv(Cw) @ Cb    

def draw(eig_faces, nr, nc,  im_shape, pth):
    # eig_faces: (bs, dim)
    def map2image(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))
    fig, axes = plt.subplots(nr, nc, figsize=(12, 8))
    axes = axes.flatten()
    for i, face in enumerate(eig_faces):
        ef = map2image(face.reshape(im_shape))
        axes[i].imshow(ef, cmap='gray')
        axes[i].set_title(f'{i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(pth)

def calculate_performance( pred, gt, title=""):
    unknown = np.sum(pred == -1)
    unknown_ratio = unknown / len(pred)
    acc = np.sum(pred == gt) / len(pred)
    print(  f"{title} Accuracy: {acc*100:.2f}% | "
            f"Unknown ratio: {unknown_ratio*100:.2f}% ({unknown})")
    return acc, unknown_ratio

def plot_tf_scatter(model, data, label, dim=2):
    assert model.w is not None, "The input model hasn't been trained."
    scatter = model.transform(data, dim)
    cls = np.unique(label)
    colors = {c: np.random.rand(3,) for c in cls}  # Each class gets a unique RGB color
    fig = plt.figure(figsize=(10, 10))
    if dim ==3 :
        ax = fig.add_subplot( projection='3d')
    else:
        ax = fig.add_subplot()
        
    for c in cls:
        # Mask data and labels corresponding to the current class
        tmp_data = scatter[label == c, :]
        if dim == 2:
            ax.scatter(tmp_data[:, 0] , tmp_data[:, 1],c = [colors[c]], label=f"{c}", )
        elif dim == 3:
            ax.scatter(tmp_data[:,0 ], tmp_data[:, 1], tmp_data[:, 2], c=[colors[c]], label=f"{c}")
    ax.set_title("Data")
    ax.legend()
    plt.savefig(f"{model.__class__.__name__}_scatter.png")

def load_data(root:str, split:str, resize_shape):
    from pathlib import Path
    faces = list()
    labels = list()
    attrs = list()
    for pth in (Path(root)/Path(split)).iterdir():
        subject, attr = pth.stem.split('.')
        pil_img = Image.open(str(pth)).resize(resize_shape, Image.LANCZOS)
        face = np.array(pil_img).flatten()
        faces.append(face)
        labels.append(int(subject.replace('subject', '')))
        attrs.append(attr)
    faces = np.stack(faces, axis=0, dtype=np.float64)
    labels = np.array(labels, dtype=np.int32)
    return faces, labels, attrs

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['pca', 'lda', 'iter'])
    parser.add_argument('--kernel', type=str, help="linear | polynomial | rbf") 
    parser.add_argument('--root', type=str, default='./Yale_Face_Database')
    parser.add_argument('-d', type=float, default=1.5, help="Constant for polynomial kernel")
    parser.add_argument('-c', type=float, default=1.0, help="Power of polynomial kernel")
    parser.add_argument('-r', type=float, default=1e-8, help="Gamma of RBF kernel")
    parser.add_argument('--plot_dim', type=int, default=2)
    return parser.parse_args()

def choose_model(model, nn, kernel_method, c, d, r):
    if model == 'pca':
        model = PCA(nn, kernel_method, c=c, d=d, r=r)
    elif model == 'lda':
        model = LDA(nn, kernel_method, c=c, d=d, r=r)
    return model    

def process(model, train_data, train_label, test_data, test_label):
    global resize_shape, num_eig, random_faces
    model_name = model.__class__.__name__
    model = model.fit(train_data, train_label)
    if model.kernel is None:
        plot_tf_scatter(model, train_data, train_labels, dim=plot_dim)
        draw(model.w[:, :25].T, 5, 5, resize_shape, f'{model_name}_eigen_faces.png')    
        reconstruct = model.reconstruct(random_faces, num_eig)
        draw(np.concatenate([reconstruct,random_faces], axis=0), 2, 10, resize_shape, f'{model_name}_reconstruction.png')   
    title = model_name if model.kernel is None else f"{model.kernel}_{model_name}"
    pred = model.predict(test_data, num_eig=num_eig)
    acc, unknown_ratio = calculate_performance(pred, test_label, title=title)
    return acc, unknown_ratio

if __name__ == "__main__":
    args = parse_args()
    root = args.root
    model = args.model
    kernel_method = args.kernel
    plot_dim = args.plot_dim 
    im_shape = (231,195)
    
    nn = 3
    resize_shape = (50, 50)
    num_eig = 14
    
    # Load data
    train_data, train_labels, train_attrs = load_data(root, split='Training', resize_shape=resize_shape)
    test_data, test_labels, test_attrs = load_data(root, split='Testing', resize_shape=resize_shape)
    # Random sample faces
    np.random.seed(10)
    rindx = np.random.randint(0, len(train_data), size=10)
    random_faces = train_data[rindx, ...]
    # Initialize model
    if model =='iter':
        if kernel_method == 'rbf':
            for model_name in ('pca', 'lda'):
                for r in np.r_[1:10]:
                    r *= 1e-6
                    model = choose_model(model_name, nn, kernel_method, c=args.c, d=args.d, r=r)
                    process(model, 
                        train_data,
                        train_labels,
                        test_data, 
                        test_labels,
                        )
                print()
        elif kernel_method == 'linear':
            for model_name in ('pca', 'lda'):
                model = choose_model(model_name, nn, kernel_method, args.c, args.d, args.r)
                process(model, train_data, train_labels, test_data, test_labels)
        elif kernel_method =='polynomial':
            for model_name in ('pca', 'lda'):
                for d in (2, 4, 6):
                    for c in np.r_[0:100:10]:
                            model = choose_model(model_name, nn, kernel_method, c, d, args.r)
                            process(model, train_data, train_labels, test_data, test_labels)
                    print()
                print()
    else:
        model = choose_model(model,nn, kernel_method, c=args.c, d=args.d, r=args.r)
        process(model, 
                train_data,
                train_labels,
                test_data, 
                test_labels,
                )