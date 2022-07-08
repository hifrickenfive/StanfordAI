import numpy as np
import util


def main(train_path, valid_path, save_path, plot_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)

    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train) # updates self.theta with the params
    clf.predict(x_valid) # updates self.theta with the params

    # Plot decision boundary on validation set. Add plot path to iamges.
    util.plot(x_valid, y_valid, clf.theta, plot_path)

    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, clf.predict(x_valid))

    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # phi
        phi = np.mean(y)

        # mean1
        y_1d = np. reshape(y, (-1, 1)) # reshape into 800,1 to do element-wise multiplication with x
        mean1 = sum(y_1d*x) / sum(y) # 1,2

        # mean0
        y_bitflip = [1-val for val in y] 
        y_bitflip = np. reshape(y_bitflip, (-1, 1)) # reshape into 800,1 to do element-wise multiplication with x
        mean0 = sum(y_bitflip*x) / sum(y_bitflip) # 1,2

        # covariance
        mean_vector = []
        for val in y:
            if val == 0:
                mean_vector.append(mean0)
            elif val ==1:
                mean_vector.append(mean1)
        mean_vector = np.reshape(mean_vector, (len(y), 2)) # 800,2
        covariance = (x - mean_vector).T @ (x - mean_vector) / y.shape[0] # 2,2
        inv_covariance = np.linalg.inv(covariance)

        # Eval theta as function of phi, mean1, mean0, covariance (ps1 1.5c)
        theta = (mean1 - mean0).T @ inv_covariance
        theta0 = -np.log((1-phi)/phi) - 0.5*(mean1.T @ inv_covariance @ mean1 - mean0.T @ inv_covariance @ mean0)
        self.theta = [theta0, theta[0], theta[1]] # each elem in thetas needs to be individually indexable because of plot util :<
        # print('end debug')

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_pred = self.theta[1:] @ x.T + self.theta[0] # this returns the score: how confident are we in predicting
        return (y_pred > 0).astype('int') # to get predicted class label return 0s or 1s
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt',
         plot_path='my_gda1.jpeg'
         )

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt',
         plot_path='my_gda2.jpeg'
)
