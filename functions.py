import numpy as np
from scipy.special import erf, erfinv
from scipy.integrate import quad


def owen_s_t_function(h, a):
    if -np.inf < h and a < np.inf:
        if a == 0:
            return 0
        elif h == 0:
            return (1 / (2 * np.pi)) * np.arctan(a)
        else:
            foo = lambda t: (np.e ** (- 0.5 * (h ** 2) * (1 + t ** 2))) / (1 + t ** 2)
            return (1/(2 * np.pi)) * quad(foo, 0, a)[0]


def normal_cdf(arr, mu, sigma):
    try:
        return 0.5 * (1 + erf((arr - mu) / (sigma * np.sqrt(2))))
    except TypeError:
        raise TypeError("Numpy array as argument is expected.")


def normal_pdf(arr, mu, sigma):
    try:
        return (1 / (np.sqrt(2 * np.pi * sigma ** 2))) * np.e ** (-(((arr - mu) ** 2) / (2 * sigma ** 2)))
    except TypeError:
        raise TypeError("Numpy array as argument is expected.")


def normal_quantile(arr, mu, sigma):
    pass


def skew_normal_cdf(arr, epsilon, omega, alpha):
    try:
        owen_s = np.array([owen_s_t_function(((i - epsilon) / omega), alpha) for i in arr])
        return normal_cdf(((arr - epsilon) / omega), epsilon, omega) - 2 * owen_s
    except TypeError:
        TypeError("Numpy array as argument is expected.")


def skew_normal_pdf(arr, epsilon, omega, alpha):
    try:
        sums = np.array([
            quad(lambda t: (1 / (np.sqrt(2 * np.pi))) * np.e ** (-((t ** 2) / 2)),
                 -np.inf, alpha * ((i - epsilon) / omega))[0] for i in arr
        ])
        return (2 / (omega * np.sqrt(2 * np.pi))) * np.e ** (-(((arr - epsilon) ** 2) / (2 * omega ** 2))) * sums
    except TypeError:
        raise TypeError("Numpy array as argument is expected.")


def skew_normal_quantile(arr, mu, sigma):
    pass


if __name__ == "__main__":

    from scipy.stats import norm, skewnorm
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 2)

    x = np.linspace(norm.ppf(0.01),
                    norm.ppf(0.99), 100)

    y = np.linspace(0.01, 0.99, 100)

    ax[0][0].plot(x, norm.pdf(x, loc=0, scale=1), 'bo', alpha=0.6, label='norm pdf (scipy)')
    ax[0][0].plot(x, normal_pdf(x, mu=0, sigma=1), 'r.', alpha=0.6, label='norm pdf (custom)')
    ax[0][0].legend(loc='best', frameon=False)
    ax[0][0].set_title("Normal PDF Comparison ($\mu = 0$ $\sigma = 1$)")

    ax[1][0].plot(x, norm.cdf(x), 'bo', lw=5, alpha=0.6, label='norm cdf (scipy)')
    ax[1][0].plot(x, normal_cdf(x, mu=0, sigma=1), 'r.', lw=5, alpha=0.6, label='norm cdf (custom)')
    ax[1][0].legend(loc='best', frameon=False)
    ax[1][0].set_title("Normal CDF Comparison ($\mu = 0$ $\sigma = 1$)")

    ax[2][0].plot(y, norm.ppf(y), 'bo', lw=5, alpha=0.6, label='norm cdf (scipy)')
    ax[2][0].legend(loc='best', frameon=False)
    ax[2][0].set_title("PPF Comparison ($\mu = 0$ $\sigma = 1$)")

    ax[0][1].plot(x, skewnorm.pdf(x, loc=0, scale=1, a=4), 'bo', lw=5, alpha=0.6, label='norm cdf (scipy)')
    ax[0][1].plot(x, skew_normal_pdf(x, epsilon=0, omega=1, alpha=4), 'r.', lw=5, alpha=0.6, label='norm cdf (custom)')
    ax[0][1].legend(loc='best', frameon=False)
    ax[0][1].set_title("Skew-normal CDF Comparison ($\mu = 0$ $\sigma = 1$ $alpha = -4$)")

    ax[1][1].plot(x, skewnorm.cdf(x, loc=0, scale=1, a=-4), 'bo', lw=5, alpha=0.6, label='norm cdf (scipy)')
    ax[1][1].plot(x, skew_normal_cdf(x, epsilon=0, omega=1, alpha=-4), 'r.', lw=5, alpha=0.6, label='norm cdf (custom)')
    ax[1][1].legend(loc='best', frameon=False)
    ax[1][1].set_title("Skew-normal CDF Comparison ($\mu = 0$ $\sigma = 1$ $alpha = -4$)")

    ax[2][1].plot(y, skewnorm.ppf(y, a=-4), 'bo', lw=5, alpha=0.6, label='norm cdf (scipy)')
    ax[2][1].legend(loc='best', frameon=False)
    ax[2][1].set_title("PPF Comparison ($\mu = 0$ $\sigma = 1$)")

    plt.show()
