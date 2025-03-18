import numpy as np


def run_coresetmcmc(data,
                    S,
                    K,
                    d,
                    coreset,
                    T,
                    subsample=True,
                    replace=False):
    kls = []

    N = data.shape[1]
    M = coreset.shape[1]

    w = ((N / M) * np.ones(M))
    theta = draw(K=K,
                 w=w,
                 d=d,
                 coreset=coreset)

    for t in range(0, T):
        S_t = subsample_data(data=data,
                             S=S,
                             subsampling=subsample,
                             replace=replace)

        g_t = est_gradient(N=N,
                           w=w,
                           theta=theta,
                           S_t=S_t,
                           S=S,
                           K=K,
                           coreset=coreset)

        gamma_t = (N / (10 * M))*(1/np.sqrt(t+1))
        w -= gamma_t * g_t

        theta = draw(K=K,
                     w=w,
                     d=d,
                     coreset=coreset)

        kls.append(kl(w=w,
                      coreset=coreset,
                      data=data,
                      d=d))
    return kls

def log_likelihood(x,
                   theta,
                   K):
    log_lik = np.zeros(K)
    for k in range (0,K):
        log_lik[k] = -0.5 * np.dot((x - theta[:, k]), (x - theta[:, k]))
    return log_lik


def subsample_data(data,
                   S,
                   subsampling,
                   replace):
    N = data.shape[1]
    if subsampling:
        indices = np.random.choice(N, size = S, replace = replace)
        return data[:, indices]
    else: return data

def est_gradient(N,
                 w,
                 theta,
                 S_t,
                 S,
                 K,
                 coreset):
    Q = theta - (theta.sum(axis=1)/K)[:, np.newaxis]
    q_ = (theta**2).sum(axis=0)
    q = q_ - q_.sum()/K
    factor1 = coreset.T.dot(Q)- 1/2*q
    factor2 = Q.T.dot((coreset*w).sum(axis=1)-(S_t.sum(axis=1)*(N/S)))-(w.sum()-N)/2*q

    return (factor1*factor2).sum(axis=1)/(K-1)

def draw(K,
         w,
         d,
         coreset):
    mu_w = (coreset*w).sum(axis=1)/(1+w.sum())
    sigma_w = np.sqrt(1 / (1 + w.sum()))

    cov_matrix = sigma_w ** 2 * np.eye(d)
    return np.random.multivariate_normal(mu_w, cov_matrix, size = K).T

def kl(w,
       coreset,
       data,
       d):
    N = data.shape[1]
    ret = d * np.log((1 + w.sum()) / (1 + N)) - d + d * (1 + N) / (1 + w.sum())
    dy = (coreset * w).sum(axis=1) / (1 + w.sum()) - data.sum(axis=1) / (1 + N)
    ret += (1 + N) * (dy ** 2).sum()
    return 0.5 * ret

def run_unif(data,
             d,
             coreset,
             T):
    kls = []
    N = data.shape[1]
    M = coreset.shape[1]
    w = ((N / M) * np.ones(M))

    for _ in range(0, T):
        kls.append(kl(w=w,
                      coreset=coreset,
                      data=data,
                      d=d))

    return kls

