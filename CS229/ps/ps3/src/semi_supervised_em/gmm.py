import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path) # 1000 examples

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples 40 
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels 40
    x = x_all[~labeled_idxs, :]        # Unlabeled examples 980

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group

    # Initialise
    if is_semi_supervised:
        data = x_all
    else:
        data = x

    n, dim = data.shape
    cluster = np.random.randint(K, size=n) # uniform split into K groups
    mu = np.empty([K, dim])
    sigma = np.empty([K, dim, dim])
    for j in range(K):
        xj = data[cluster==j]
        nj = len(xj)
        mu[j] = np.mean(xj, axis=0)
        sigma[j] = np.matmul((xj - mu[j]).T , (xj - mu[j])) / nj

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones(K) / K

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones([n, K]) / K

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(0,n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w

        # Initialise
        prev_ll = ll
        n, d = x.shape # Num samples and num dimension of each sample
        _numerator = np.empty([n, K])

        for j in range(K):
            # The aim is to solve P(z=j|x; phi, mu, sigma), which is the posterior probabilty given fixed parameters for our distribution
            # We apply Bayes rule to arrive at P(x|z=j)P(z=j) / sum_{l=1}^{K} (P(x|z=l)P(z=l)). See my handnotes in my ipad CS229 concepts
            # The denominator normalises such that we arrive at the cluster allocation probability
            # So for each sample, and each cluster:
            #    numerator = multi-variate gaussian formula x phi (n samples x 1 dimension)
            #    denominator = sum of all clusters (n samples x 1 dimension)
            # This implies each sample will have K elements for each cluster probability
            # The sum of the soft allocations of each sample point to all clusters must sum to 1 of course

            # To aid calculation we take the log of numerator and denominator then unwind this by applying the exponent at the end
            _numerator[:,j] = - d/2*np.log(2*np.pi) \
                              - 1/2*np.log(np.linalg.det(sigma[j])) \
                              - 1/2*np.sum((x - mu[j]) @ np.linalg.inv(sigma[j]) * (x - mu[j]), axis=1) \
                              + np.log(phi[j])
            # Note: The third element is (980x2): <matmul((980,2),(2,2)), 980x2>
            #   But we sum the rows to reduce it to (980x1). I'm not 100% sure why that works.

            _numerator_exp = np.exp(_numerator) # (980x4)
            _denominator = np.sum(_numerator_exp, axis=1).reshape(-1,1) # (980x1)

        w = _numerator - np.log(_denominator)
        w = np.exp(w)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = np.sum(w, axis=0) / n
        for j in range(K):
            sum_wj = np.sum(w[:,j]) 
            mu[j] = w[:,j] @ x / sum_wj
            sigma[j] = w[:,j] * (x - mu[j]).T @ (x-mu[j]) / sum_wj

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        # Reevaluate p(x,z; mu, sigma, phi) for the updated parameters.  
        _temp = np.zeros((n,K))
        for j in range(K):
            _temp[:,j] = - d/2*np.log(2*np.pi) \
                         - 1/2*np.log(np.linalg.det(sigma[j])) \
                         - 1/2*np.sum((x - mu[j]) @ np.linalg.inv(sigma[j]) * (x - mu[j]), axis=1) \
                         + np.log(phi[j])
            _temp[:,j] = np.exp(_temp[:,j]) 
        __temp = np.sum(_temp, axis=1)
        __temp_log = np.log(__temp)
        ll = np.sum(__temp_log, axis=0)

        it += 1
        # *** END CODE HERE ***

    print(f'Iterations: {it}, log loss: {ll}')
    return w

def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None

    # Initialise first time rather than in the while loop
    n_unobserved = x.shape[0]           
    n_tilde = x_tilde.shape[0]        

    x = np.concatenate([x, x_tilde], axis=0)
    n, d = x.shape
    
    # Weights for observed examples. Use this knowledge to patch E-Step
    w_tilde = alpha * (z_tilde == np.arange(K))

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        prev_ll = ll
        _numerator = np.empty([n, K])
        for j in range(K):
            _numerator[:,j] = - d/2*np.log(2*np.pi) \
                              - 1/2*np.log(np.linalg.det(sigma[j])) \
                              - 1/2*np.sum((x - mu[j]) @ np.linalg.inv(sigma[j]) * (x - mu[j]), axis=1) \
                              + np.log(phi[j])

        _numerator_exp = np.exp(_numerator) # (980x4)
        _denominator = np.sum(_numerator_exp, axis=1).reshape(-1,1) # (980x1)
        w = _numerator - np.log(_denominator)
        w = np.exp(w)
        w[-n_tilde:] = w_tilde # Override weights with the known labels from supervised set

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = np.sum(w, axis=0) / (n_unobserved + alpha * n_tilde)
        for j in range(K):
            sum_wj = np.sum(w[:,j]) 
            mu[j] = w[:,j] @ x / sum_wj
            sigma[j] = w[:,j] * (x - mu[j]).T @ (x-mu[j]) / sum_wj


        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        _temp = np.zeros((n,K))
        for j in range(K):
            _temp[:,j] = - d/2*np.log(2*np.pi) \
                         - 1/2*np.log(np.linalg.det(sigma[j])) \
                         - 1/2*np.sum((x - mu[j]) @ np.linalg.inv(sigma[j]) * (x - mu[j]), axis=1) \
                         + np.log(phi[j])
            _temp[:,j] = np.exp(_temp[:,j]) 
        __temp = np.sum(_temp, axis=1)
        __temp_log = np.log(__temp)
        ll = np.sum(__temp_log, axis=0)

        it += 1
        # *** END CODE HERE ***
    print(f'Iterations: {it}, log loss: {ll}')
    return w


# *** START CODE HERE ***
# Helper functions

# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

        main(is_semi_supervised=True, trial_num=t)

        # *** END CODE HERE ***
