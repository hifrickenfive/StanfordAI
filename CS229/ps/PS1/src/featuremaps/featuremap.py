import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(X[0].shape) # solving for theta, which is the size of one sample

        # Update thetha via the normal equations
        self.theta = np.linalg.solve(X.T @ X, X.T@y) # (A,b). Don't solve inverse matrices directly because could be numerically unstable
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        if X.shape[0] > 1:
            column_base = X[:, 1]
        else:
            column_base = X

        for i in range(2, k+1):
            col = column_base**i
            col = col.reshape(-1, 1)
            X = np.concatenate((X, col), axis=1)
        return X 
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***

        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta.reshape(-1,1)
        y_predict =  X @ self.theta.T
        return y_predict
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)

    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***

        # Train on train_x and train_y
        reg = LinearModel()
        train_x_poly  = reg.create_poly(k, train_x)
        reg.fit(train_x_poly, train_y)

        # Predict on plot_x
        plot_x_poly = reg.create_poly(k, plot_x)
        plot_y = reg.predict(plot_x_poly)

        # *** END CODE HERE ***
        '''
        Here y_predict are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path)
    x_small, y_small = util.load_dataset(small_path)
    x_test, y_test = util.load_dataset(eval_path)

    run_exp(train_path, sine=False, ks=[3], filename='my_poly3.png')


    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
