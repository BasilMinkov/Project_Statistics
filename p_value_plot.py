import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu

from scipy.stats import norm, skewnorm

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
n = 10
t = np.random.uniform(0, 1, n)
mu = 0
sigma = 1
alpha = 0
s = norm.ppf(t, mu, sigma)
x, y = np.unique(np.around(s, decimals=1), return_counts=True)

l, = plt.plot(x, y, color='red')
l1, = plt.plot(x, y, color='blue')

# plt.axis([-5, 5, 0, 10])
plt.axis([-5, 5, 0, 100])

axcolor = 'lightgoldenrodyellow'
axalpha = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
axsigma = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axmu = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axnum = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

skew = Slider(axalpha, 'Alpha', -5.0, 5.0, valinit=alpha)
mathematic_expectation = Slider(axmu, 'Mu', -5.0, 5.0, valinit=mu)
standard_deviation = Slider(axsigma, 'Sigma', 0.5, 5.0, valinit=sigma)
# sample_n = Slider(axnum, 'Samples', 10.0, 200.0, valinit=n)
sample_n = Slider(axnum, 'Samples', 10.0, 100.0, valinit=n)

te = plt.text(0, 29, "T-test")


def update(val):

    alpha1 = skew.val
    mu1 = mathematic_expectation.val
    sigma1 = standard_deviation.val
    n1 = sample_n.val

    t1 = np.random.uniform(0.0, 1.0, int(n1))
    s1 = skewnorm.ppf(t1, alpha1, mu1, sigma1)
    # s1 = norm.ppf(t1, mu1, sigma1)
    x1, y1 = np.unique(np.around(s1, decimals=1), return_counts=True)

    t2 = np.random.uniform(0.0, 1.0, int(n1))
    s2 = norm.ppf(t2, mu, sigma)
    x2, y2 = np.unique(np.around(s2, decimals=1), return_counts=True)

    t, p1 = np.around(ttest_ind(s1, s2), decimals=3)
    u, p2 = np.around(mannwhitneyu(s1, s2), decimals=3)

    l.set_xdata(x1)
    l.set_ydata(y1)
    l1.set_xdata(x2)
    l1.set_ydata(y2)
    te.set_text("T-test: t = {}, p = {}; Mann-Whitney: U = {}, p = {}".format(t, p1, u, p2))
    fig.canvas.draw_idle()


skew.on_changed(update)
mathematic_expectation.on_changed(update)
standard_deviation.on_changed(update)
sample_n.on_changed(update)

resetax = plt.axes([0.8, 0.000001, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    mathematic_expectation.reset()
    standard_deviation.reset()
    sample_n.reset()
    skew.reset()


button.on_clicked(reset)

plt.show()
