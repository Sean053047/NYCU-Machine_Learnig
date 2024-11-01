import argparse
import numpy as np 
import matplotlib.pyplot as plt
from typing import Union

def read_idx3_ubyte(fpth:str):
    with open(fpth, 'rb') as file:
        byte_data = file.read()
    rows = int.from_bytes(byte_data[8:12], 'big')
    cols = int.from_bytes(byte_data[12:16], 'big')
    # Set an image into a vector.
    data = np.frombuffer(byte_data[16:], dtype=np.uint8).reshape((-1, rows*cols),order='C')
    return rows, cols, data

def read_idx1_ubyte(fpth:str):
    with open(fpth, 'rb') as file:
        byte_data = file.read()
    label = np.frombuffer(byte_data[8:], dtype=np.uint8)
    label_type = np.unique(label)
    return label, label_type

def show_result(label, pred, posterior):
    print("Posterior (in log scale):")
    for i in range(len(posterior)):
        print(f"{i}: {posterior[i]}")
    print(f"Prediction: {pred}, Ans: {label}")

def plot_figure(result_imgs):
    '''
    Value = 1 : white
    Value = 0 : black
    '''
    fig, axes = plt.subplots(2,5, figsize=(10,5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = result_imgs[i].astype(np.uint8) *255
        ax.imshow(img, cmap='gray')
        ax.set_title(f"digit: {i}")
        ax.axis('off') 
    plt.tight_layout()
    plt.show()
    
class ContinuousModel:
    def __init__(self, label_type,  rows, cols, ):
        self.label_type = label_type
        self.label2indx = { label: indx for indx, label in enumerate(label_type)}
        self.img_shape = (rows, cols)
        
        self.mean = np.zeros((len(label_type), rows*cols), dtype=np.float64) # 
        self.var = np.zeros((len(label_type), rows*cols), dtype=np.float64) # 
        self.prior = np.zeros((len(label_type)), dtype=np.float64)
        
    def fit(self, data, label):
        assert len(data) == len(label), "Wrong pair for training data & labels."
        # Treat each pixel is independent.
        N = len(data)
        # Accumulate mean
        for i in range(N):
            img, digit = data[i,:], label[i]
            lindx = self.label2indx[digit]
            self.prior[lindx] +=1 
            self.mean[lindx, :] += img
        # Calculate the final mean 
        for digit in self.label_type:
            lindx = self.label2indx[digit]
            self.mean[lindx, :] /= self.prior[lindx]    
        
        # Accumulate variance
        self.var = np.zeros_like(self.var)
        for i in range(N):
            img, digit = data[i,:], label[i]
            lindx = self.label2indx[digit]
            self.var[lindx, :] += (img - self.mean[lindx, :]) **2
        
        # Calculate the final variance
        for digit in self.label_type:
            lindx = self.label2indx[digit]
            self.var[lindx,:] /= self.prior[lindx]
            
        assert not np.any(np.bitwise_and(self.var < 0, np.abs(self.var)> 1e-6)), "Wrong var."
        self.var[self.var <= 0] = 0.5 * np.pi
        self.prior /= N    
        
    def predict(self, img:np.ndarray):
        log_L_pixelwise = -0.5*np.log(2*np.pi* (self.var) ) - 0.5* (img - self.mean)**2 / (self.var)
        log_L = np.sum(log_L_pixelwise, axis=1)
        posterior = np.log(self.prior) + log_L
        posterior /= sum(posterior)
        pred = np.argmin(posterior)
        return pred, posterior
    
    def get_likelihood_img(self):
        likeli_pixels = np.zeros_like(self.mean)
        likeli_pixels[self.mean > 127] = 1 
        result_imgs = {
            digit:likeli_pixels[digit,:].reshape(self.img_shape) for digit in self.label_type
        }
        return result_imgs

class DiscreteModel:
    def __init__(self, label_type, num_bins, rows, cols, ):
        self.label_type = label_type
        self.label2indx = { label: indx for indx, label in enumerate(label_type)}
        self.N_bins = num_bins
        self.bin_interval = 256/ num_bins # 8
        self.img_shape = (rows, cols)
        
        self.likelihood = np.zeros((len(label_type), num_bins, rows*cols), dtype=np.float64)
        self.prior = np.zeros((len(label_type)), dtype=np.float64)
        
    def fit(self, data, label):
        assert len(data) == len(label), "Wrong pair for training data & labels."
        N = len(data)
        # Treat each pixel is independent to other pixels.
        # This Part will calculate the number of occurrences of each pixel for all labels.
        img_indx= np.arange(self.img_shape[0] * self.img_shape[1])
        for i in range(N):
            img, digit = data[i, :], label[i]
            lindx = self.label2indx[digit]
            bin_indx = (img // self.bin_interval).astype(np.int32)
            self.prior[lindx] =  self.prior[lindx]+ 1
            self.likelihood[lindx, bin_indx, img_indx] =  self.likelihood[lindx, bin_indx, img_indx] +1
        for digit in self.label_type:
            lindx = self.label2indx[digit]
            self.likelihood[lindx, :, :] = self.likelihood[lindx,:,:] / self.prior[lindx]
        self.likelihood[self.likelihood == 0] = 1e-8
        self.prior /= N
        
    def predict(self, img:np.ndarray):
        '''Predict for single image,
        Assume each pixel is independent. P(p1,p2,...,pk | Y) = P(p1|Y) * P(p2|Y) * ... * P(pk|Y) * P(Y)
        '''
        img = img.flatten()
        posterior = np.log(self.prior)
        bin_indx = (img // self.bin_interval).astype(np.int32)
        img_indx= np.arange(len(img))
        log_L_pixelwise = np.log(self.likelihood[:, bin_indx, img_indx]) # 10 label * 784 pixels(feature)
        log_L = np.sum(log_L_pixelwise, axis=1)
        posterior = posterior + log_L
        posterior /= np.sum(posterior)
        pred = np.argmin(posterior) # The posterior value are negative.
        return pred, posterior

    def get_likelihood_img(self):
        result_imgs = dict()
        for digit in self.label_type:
            # Find the most possible bin for each pixel.
            most_likeli_bin = np.argmax(self.likelihood[digit,:,:], axis=0)
            threshold_bin = 128 // self.bin_interval
            likelihood_img = np.zeros_like(most_likeli_bin)
            likelihood_img[most_likeli_bin >threshold_bin] = 1.0
            result_imgs[digit] = likelihood_img.reshape(self.img_shape)
        return result_imgs
                        

def main(train_data, train_label, test_data, test_label, model:Union[DiscreteModel, ContinuousModel]):
    model.fit(data=train_data,
              label=train_label)
    num_test = test_data.shape[0]
    preds = []
    for i in range(num_test):
        img = test_data[i,:]
        pred, posterior = model.predict(img)
        show_result(test_label[i], pred, posterior)
        preds.append(pred)
    error_rate = np.sum(preds!=test_label) / num_test
    print(f"Num_test: {test_label.shape[0]}, Error rate: {error_rate}")
    plot_figure(result_imgs=model.get_likelihood_img())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Machine learning homework2")
    parser.add_argument('--option', required=True, type=int, help="0: discrete mode; 1: continuous mode")
    args = parser.parse_args()
    
    TEST_IMG_FILE = "data/t10k-images.idx3-ubyte_"
    TEST_LABEL = "data/t10k-labels.idx1-ubyte_"
    TRAIN_IMG_FILE = "data/train-images.idx3-ubyte_"
    TRAIN_LABEL = "data/train-labels.idx1-ubyte_"
    rows, cols, test_data = read_idx3_ubyte(TEST_IMG_FILE)
    test_label, label_type = read_idx1_ubyte(TEST_LABEL)
    _, _, train_data = read_idx3_ubyte(TRAIN_IMG_FILE)
    train_label, _ = read_idx1_ubyte(TRAIN_LABEL)
    
    if args.option == 0:
        model = DiscreteModel(label_type=label_type,
                                num_bins=32,
                                rows=rows,
                                cols=cols)
    elif args.option == 1:
        model = ContinuousModel(label_type=label_type, 
                                  rows=rows, cols=cols)
    else:
        raise "Wrong option input."
    main(train_data=train_data,
         train_label=train_label,
         test_data=test_data,
         test_label=test_label,
         model=model,
         )