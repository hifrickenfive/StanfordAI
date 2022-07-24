# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
from matplotlib import pyplot as plt

def plot3d(x,y,z):
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=z, cmap='Greens')
    plt.show()


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    theta_data = []
    lbda = 0.01
    i = 0 

    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)

        # With L2 norm regularizatoin
        theta = theta + learning_rate * (grad - 2*lbda*theta) # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.169.3611&rep=rep1&type=pdf p.7

        # Without L2 regularizaton
        # theta = theta + learning_rate *grad

        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print(theta)

        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            # result = np.reshape(theta_data, (-1,3))
            # plot3d(result[:,0], result[:,1],result[:,2])
            util.plot(X, Y, theta, 'ds1_a_line', correction=1.0)
            break

        # To interogate non-converging ds1_b    
        # if i == 200000:
        #     # result = np.reshape(theta_data, (-1,3))
        #     # plot3d(result[:,0], result[:,1],result[:,2])
        #     # util.plot(X, Y, theta, 'ds1_b_line', correction=1.0)
        #     break

        # theta_data.append(theta)
    return


def main():
    # print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)

if __name__ == '__main__':
    main()
