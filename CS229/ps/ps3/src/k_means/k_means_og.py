from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.io as spio

def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    # Flatten the image from 3d into 2d i.e. (128,128,3) into (16384, 3)
    H, W, C = image.shape
    image_flattened = image.reshape(-1, C)
    centroid_idx = np.random.randint(H * W, size=num_clusters)
    centroids_init = image_flattened[centroid_idx] 
    # *** END YOUR CODE ***
    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    k = len(centroids)
    H, W, C = image.shape
    image_flattened = image.reshape(-1, C)
    
    distances = np.zeros([k, H * W]) # Pre-allocate
    
    for i in range(max_iter):
        
        # Hard assignments
        for j in range(k):
            distances[j] = np.sum((image_flattened - centroids[j]) ** 2, axis=1)
        assignments = np.argmin(distances, axis=0).reshape(-1, 1)
        
        # Re-evaluate centroids
        new_centroids = np.empty([k, C])
        for j in range(k):
            assignments_idx = np.where(assignments == j)[0]
            new_centroids[j] = np.sum(image_flattened[assignments_idx,:], axis=0) / len(assignments_idx)
        
        # Evaluate loss
        loss = (image_flattened - new_centroids[assignments.squeeze()]) ** 2
        print(np.sum(loss))

        # Early stop 
        if np.array_equal(centroids, new_centroids):
            print(f'Done in {i} iterations')
            return new_centroids

        centroids = new_centroids

    print(f'Did not converge')
    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    num_clusters = len(centroids)
    H, W, C = image.shape
    image_flattened = image.reshape(-1, C)
    dist = np.empty([num_clusters, H * W])

    for j in range(num_clusters):
        dist[j] = np.sum((image_flattened - centroids[j]) ** 2, axis=1)
    assignments = np.argmin(dist, axis=0)
    new_image = centroids[assignments].reshape(H, W, C)
    # *** END YOUR CODE ***

    return new_image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    # image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    # image = np.copy(mpimg.imread(image_path_small))
    mat = spio.loadmat('peppers_small.mat', squeeze_me = True) # Workaround.
    image = mat['peppers_small'] / 255

    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large)) / 255
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
