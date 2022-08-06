from matplotlib.image import imread
import matplotlib.pylab as plt

# print('Finished')
# A = imread('peppers-large.tiff')
# # plt.imshow(A)
# # plt.show()
# # print(A.shape)

print('load')
B = imread('peppers-small.tiff')
print('load finished')
plt.imshow(B)
plt.show()
print(B.shape)

print('Finished')
