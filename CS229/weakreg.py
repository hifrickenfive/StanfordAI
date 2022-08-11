import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import itertools

np.random.seed(229)

# n is number of examples in training set
n = 10

# p is degree of polynomial for feature map psi(X)
p = 4

# tolerance to check for convergence
eps = 1e-4

# fixed standard deviation for noise
stdv = 1

# learning rate for full batch gradient descent
lr = 0.001

def psi(X):
    ##
    ## Polynomial Feature map (of degree p). Map X -> psi(X)
    ##
    ## Arguments
    ## ---------
    ##    X : Vector of Inputs. Shape = (n_examples,)
    ##
    ## Return value
    ## ------------
    ##     psi(X) : Mapped features. Shape = (n_examples, p)
    ##
    psiX = np.hstack([np.power(X, i).reshape((-1, 1)) for i in range(p)]).squeeze()
    return psiX

def data_XLU():
    ##
    ## Generate synthetic training data with weak supervision.
    ##
    X = np.linspace(-3, +3, n)
    y = np.sin(X)
    L = norm.ppf(0.5 - np.random.random(n) * 0.5, loc=y, scale=stdv)
    U = norm.ppf(0.5 + np.random.random(n) * 0.5, loc=y, scale=stdv)
    return (X, L, U)

def calc_grad(X, L, U, theta):
    ##
    ## Calculate the gradient of the log-likelihood w.r.t theta.
    ##
    ## Arguments
    ## ---------
    ##    X : Vector of Inputs. Shape = (n_examples,)
    ##    L : Lower bounds of Y. Shape = (n_examples,)
    ##    U : Upper bounds of Y. Shape = (n_examples,)
    ##    theta : current value of parameters. Shape = (p,)
    ##
    ## Return value
    ## ------------
    ##     grad : Gradient of log-likelihood w.r.t theta. Shape = (p,)
    ##
    ## Intermediate value
    ## ------------------
    ##     psiX : psi(X), feature map of X. Shape = (n_examples, p)
    ##
    psiX = psi(X)
    grad = None

    # *** START CODE HERE ***

    # *** END CODE HERE ***
    return grad

def fit(X, L, U):
    ##
    ## Implements full batch gradient descent to fit the model
    ## with inputs X and weak supervision L, U.
    ##
    theta = np.zeros(psi(X).shape[1])    # Initialize theta to 0.

    for i in itertools.count():
        grad = calc_grad(X, L, U, theta)
        print('[Iter {}] Gradient Norm: {}'.format(i, np.linalg.norm(grad)))
        if np.linalg.norm(grad) < eps:
            print('== Converged ==')
            break
        theta = theta + lr * grad

    return theta

def predict(X, theta):
    ##
    ## Make predictions on X using the fitted theta.
    ##
    ## Arguments
    ## ---------
    ##     X : Inputs, dimension. Shape = (n_examples,)
    ##     theta : current value of parameters. Shape = (p,)
    ##
    ## Return value
    ## ------------
    ##     yhat : Predicted Y values corresponding to X. Shape = (n_examples,)
    ##
    ## Intermediate value
    ## ------------------
    ##     psiX : psi(X), feature map of X. Shape = (n_examples, p)
    ##
    psiX = psi(X)
    yhat = None

    # *** START CODE HERE ***

    # *** END CODE HERE ***
    return yhat

def main():
    X, L, U = data_XLU()

    theta = fit(X, L, U)

    plot_X = np.linspace(-4, +4, 100)
    yhat = predict(plot_X, theta)
    y = np.sin(plot_X)

    plt.errorbar(X, np.sin(X), [np.sin(X) - L, U - np.sin(X)], None,
                 ecolor='black', color='black', ls='', capsize=4, label='Data')
    plt.plot(plot_X, yhat, color='red', label='Model')
    plt.plot(plot_X, y, color='lightgray', label='True Y', ls='-.')
    plt.legend(loc='upper right')

    plt.ylim([-4, 4])
    plt.xlim([-4, 4])
    plt.xlabel('X')
    plt.ylabel('[L, U] intervals of Y')
    fname = 'fit.png'
    print('Saving plot to file: {}'.format(fname))
    plt.savefig(fname)

if __name__ == '__main__':
    main()