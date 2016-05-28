import numpy as np

y=2
print y.^2


#
# import csv
# with open("requirements.txt",'rb') as f:
#     file=csv.reader(f, delimiter=' ', quotechar='|')
#     for row in file:
#         print ','.join(row)
#
#
# from scipy.misc import imread, imsave, imresize
# import sys
# sys.path.insert(1,'/Library/Python/2.7/site-packages')
# from scipy.spatial.distance import pdist, squareform
#
# import matplotlib.pyplot as plt
#
# num_folds=5
# for i in range(1,num_folds):
#     print "hi"
#
# # Create a new array from which we will select elements
# a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
# c= a[:,1:3]
# d=a[:,1]
# e=np.array([[1],[2],[3]])
# print a[e].shape
# # print c
# # print c.shape
# # print d.shape
# # print d.reshape(4,-1).shape
# # print c+d.reshape(4,-1)
# b= np.array([1,1,1])
#
# print a-b
# print np.linalg.norm(a,axis=1)
#
# b= np.split(a,4)
# print "shape:",b[1].shape
#
# print np.sum(a,1)
#
# print a  # prints "array([[ 1,  2,  3],
#          #                [ 4,  5,  6],
#          #                [ 7,  8,  9],
#          #                [10, 11, 12]])"
#
# # Create an array of indices
# b = np.array([0, 2, 0, 1])
#
# print b**2
#
# c= [i for i in range(1,5)]
#
#
# print "check"
# for i in range (1,5):
#     print c[i+1:i+num_folds-1]
#
#
# # # boolean indexing
# # a = np.array([[1,2], [3, 4], [5, 6]])
# #
# # print a[a>2]
# #
# #
# # x= np.array([1,2])
# # print x
# # print x.shape
# #
# #
# # v= np.array([1,2,3])
# # w= np.array([4,5])
# # print np.reshape(v, (3,1))*w
# #
# # x= np.array([[1,2,3], [4,5,6]])
# # print x+v
# #
# # img = imread('/Users/shubhambansal/Pictures/IMG_4579.jpg')
# # print img.dtype, img.shape
# #
# # img_tinted = img * [1, 0.95, 0.9]
# #
# # # Resize the tinted image to be 300 by 300 pixels.
# # img_tinted = imresize(img_tinted, (300, 300))
# #
# # # Write the tinted image back to disk
# # imsave('/Users/shubhambansal/Pictures/IMG_4579(2).jpg', img_tinted)
# #
# # # x= np.arange(0,3*np.pi, 0.1)
# # # y= np.sin(x)
# # # plt.plot(x,y)
# # # plt.show()
#
# # import random
# # import numpy as np
# # from cs231n.data_utils import load_CIFAR10
# # import sys
# # sys.path.insert(1,'/Library/Python/2.7/site-packages')
# # import matplotlib.pyplot as plt
# #
# # # This is a bit of magic to make matplotlib figures appear inline in the notebook
# # # rather than in a new window.
# # %matplotlib inline
# # plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# # plt.rcParams['image.interpolation'] = 'nearest'
# # plt.rcParams['image.cmap'] = 'gray'
# #
# # # Some more magic so that the notebook will reload external python modules;
# # # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# # %load_ext autoreload
# # %autoreload 2
#
#
# y= np.arange(0,4000)
# counts = np.bincount(y)
# print y.shape
# print counts
# print np.argmax(counts)