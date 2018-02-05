from __future__ import print_function
from skimage.feature import multiblock_lbp
import numpy as np
from numpy.testing import assert_equal
from skimage.transform import integral_image
from skimage import data
from matplotlib import pyplot as plt
from skimage.feature import draw_multiblock_lbp

def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def histogram_feature(real_img, img, p_blocks, n_features, h_size, slider):
    xsize, ysize = img.shape
    histogram = np.zeros(h_size, dtype = float)
    y =0
    x =0
    while ((y+ 3*slider) <= ysize):
        while ((x + 3*slider) <= xsize):
            lbp_code = multiblock_lbp(img, y, x, p_blocks, p_blocks)
            histogram[lbp_code]+=1
            x+=slider
            #region_print(real_img, lbp_code, y+20, x+15, p_blocks, p_blocks)
            
        y+=slider
        x =0
    histogram = np.sort(histogram, axis=None) 
    features = np.zeros(n_features, dtype = float)
    histogram[:h_size-n_features]
    features[0:n_features-1] = histogram[h_size-(n_features-1):]
    features[n_features-1] = np.sum(histogram[:h_size-(n_features-1)])
    max = np.amax(features)
    return features/max

def region_print(target, code, y0, x0, y1, x1):
    img_target = draw_multiblock_lbp(target, y0, x0, y1, x1, lbp_code=code, alpha=0.5)
    plt.ion()
    plt.imshow(img_target)
    
    #plt.draw()
    
    #plt.show()
    plt.pause(0.05)


img = data.coins()

int_img = integral_image(img)

p_blocks = 3
h_size = 256
n_features = 59
slider = 3

int_img_target = int_img[20:80,15:75]

region_print(img, 0, 20, 15, 60/3, 60/3)


features_target = histogram_feature(img, int_img_target, p_blocks, n_features, h_size, slider)

region = 60
threshold = 0
ysize, xsize = img.shape
y = 0
x = 0
print (img.shape)
while ((y+ region+ slider) <= ysize):
    while ((x + region+ slider) <= xsize):
        #print ("X: %d - Y: %d" % (x, y))
        window = int_img[y:y+region,x:x+region]
        features_window = histogram_feature(img, window, p_blocks, n_features, h_size, slider)
        metric = kullback_leibler_divergence(features_window, features_target)
        #print(metric)
        if (metric <= threshold):
            print ("------->   X: %d - Y: %d" % (x, y))
            region_print(img, 255, y, x, region/3, region/3)

        x+=slider
    y+=slider
    x=0
