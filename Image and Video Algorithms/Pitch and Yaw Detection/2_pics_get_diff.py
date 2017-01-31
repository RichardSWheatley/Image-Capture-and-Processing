# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:28:31 2017

@author: rwheatley
"""

from skimage import io
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB
from scipy.spatial import distance
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

from tkinter.filedialog import askopenfilename, askopenfilenames

def get_filenames():
#    tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filez = askopenfilenames()
    filenames = filez.split(" ")
    return filenames

def get_filename():
#    tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    return filename

import tkinter
root = tkinter.Tk()
root.withdraw()

image1 = io.imread(get_filename())
image2 = io.imread(get_filename())

img1 = rgb2gray(image1)
img2 = rgb2gray(image2)
rows,cols = img1.shape

descriptor_extractor = ORB(n_keypoints=100)

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

#descriptor_extractor.detect_and_extract(img3)
#keypoints3 = descriptor_extractor.keypoints
#descriptors3 = descriptor_extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

inlier_keypoints_left = matches12[:, 0]
inlier_keypoints_right = matches12[:, 1]

print("Number of matches:", matches12.shape[0])
#print("Number of inliers:", inliers.sum())

# Compare estimated sparse disparities to the dense ground-truth disparities.

#disp = inlier_keypoints_left[:, 1] - inlier_keypoints_right[:, 1]
#==============================================================================
# print(keypoints1)
# print(keypoints2)
# print(matches12)
# print(keypoints1[inlier_keypoints_left[0]])
# print(keypoints2[inlier_keypoints_right[0]])
#==============================================================================
mine = keypoints2[inlier_keypoints_right] - keypoints1[inlier_keypoints_left]
print(mine)
print(np.average(mine[:, 0]))
print(np.average(mine[:, 1]))

plt.scatter(mine[:, 0], mine[:, 1])
plt.show()


##############################################################################
# Generate sample data

batch_size = 45
centers = mine
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

##############################################################################
# Compute clustering with Means

k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

##############################################################################
# Plot result

#==============================================================================
# colors = ['#4EACC5', '#FF9C34', '#4E9A06']
# plt.figure()
# plt.hold(True)
# for k, col in zip(range(n_clusters), colors):
#     my_members = k_means_labels == k
#     cluster_center = k_means_cluster_centers[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], 'w',
#             markerfacecolor=col, marker='.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=6)
# plt.title('KMeans')    
# plt.grid(True)
# plt.show()
#==============================================================================

from pylab import plot,show
from scipy.cluster.vq import kmeans,vq

# data generation
data = mine

# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(data,2)
# assign each sample to a cluster
idx,dist = vq(data,centroids)

if data[idx==0,:].shape > data[idx==1,:].shape:
   plot(data[idx==0,0],data[idx==0,1],'ob')
else:
   plot(data[idx==1,0],data[idx==1,1],'ob')

#plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()
print(data[idx==0,:].shape)
print(data[idx==1,:].shape)

if data[idx==0,:].shape > data[idx==1,:].shape:
   print(np.average(data[idx==0,0]))
   print(np.average(data[idx==0,1]))
else:
   print(np.average(data[idx==1,0]))
   print(np.average(data[idx==1,1]))

#==============================================================================
# height, width, channels = image1.shape
# print(height, width, channels)
#==============================================================================
#matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)

#==============================================================================
# fig, ax = plt.subplots(nrows=1, ncols=1)
# 
# plt.gray()
# 
# plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12)
# ax[0].axis('off')
# 
# #plot_matches(ax[1], img1, img3, keypoints1, keypoints3, matches13)
# #ax[1].axis('off')
# 
# plt.show()
#==============================================================================
#root.destroy()