import numpy as np
import util
import os
import math

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    util.plot(x_valid, y_valid, clf.theta, "plot_of_decision_boundary")

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, clf.predict(x_valid))

    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # y is a flat array with 800 elements (800, 1)
        # x is (800,3)
        if self.theta is None:
            self.theta = np.zeros(x.shape[1]) # implicitly, its dim is (1,3) but 3 element array

        while True:
            hypothesis = self.predict(x)

            # Find average gradient of entire dataset
            gradient = ((hypothesis - y) * x.T).mean(axis=1) # hypothesis is (800,) x.T is (3, 800), when multiplied get (3,800)

            # Find hessian
            hess = ((hypothesis * (1 - hypothesis)) * x.T) @ x / x.shape[1] # (3,3)

            # Perform Newton's Method
            new_theta = self.theta - gradient @ np.linalg.inv(hess.T) # (3,)

            if np.abs((new_theta - self.theta).sum()) < self.eps:
                return self
            else:
                self.theta = new_theta


        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        z = np.matmul(self.theta, x.T) # (1,3) (3x800) = (1x800)
        hypothesis = 1 / (1 + np.exp(-z)) # hypothesis
        return hypothesis
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
