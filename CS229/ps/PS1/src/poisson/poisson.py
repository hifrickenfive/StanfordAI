import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression()
    clf.fit(x_train, y_train)

    # Load val set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_predict = clf.predict(x_val)

    # My plot function
    plt.scatter(y_val, y_predict)
    plt.xlabel('True Counts')
    plt.ylabel('Predicted Counts')
    plt.title('Predicted Counts vs. True Counts Poisson GLM')
    plt.savefig('my_predict_vs_validation')

    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Initialise
        if self.theta is None:
            self.theta = np.zeros(x.shape[1]) # (1,5) We are solving for the weight for a SINGLE example
        iter = 1
        error = np.inf

        while abs(error) > self.eps:
            print(iter) # Debug

            # Full batch SGA
            y_predict = self.predict(x) # (2500,1)
            gradient  = ((y-y_predict)* x.T).mean(axis=1) # Full batch. Take the mean of each column. # (2500,)^T (2500, 5) -> (2500,5). Mean flattens into (1,5)
            theta_update = self.theta +  self.step_size*gradient # Eqn derived in PS1, Q3c
            error = np.linalg.norm(theta_update - self.theta, ord=1) # Scalar / euclidean distance

            # Max iter stop condition
            if iter > self.max_iter:
                return self.theta
            else:
                self.theta = theta_update
                iter +=1

        return self.theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_predict = np.exp(x @ self.theta.T) # (2500,5) (5,1) results in 2500,1
        return y_predict
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
