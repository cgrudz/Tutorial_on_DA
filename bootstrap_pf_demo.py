import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider, RadioButtons
from ipywidgets import interactive
from IPython.display import display
from matplotlib.patches import Ellipse
from matplotlib import colors
from matplotlib.colors import LogNorm
import copy
import ipdb


########################################################################################################################
# Simulation and visualization parameters
########################################################################################################################

## Initial simiulation paramters
ensn_0 = 25              # ensemble (sample) size
bstd_0 = 0.1             # backgorund error standard deviation
rstd_0 = 0.1             # observation error standard deviation
k_0 = 1                  # number of forecasts under the numerical model
analysis_0 = 'Bootstrap' # bootstrap particle filter versus free ensemble forecast

# color map paramters for particle weights
wmap = "plasma"
norm = colors.LogNorm(vmin=10e-10, vmax=0.1)

# set histogram bin edges, evenly spaced in log-log scale
bins = 10.0**np.linspace(-16,0,65)

# set histogram bin widths
bin_widths = []
for i in range(len(bins)-1):
    bin_widths.append(bins[i+1] - bins[i])
bin_widths = np.array(bin_widths)

# histogram bin colors to match the parameter weights
bin_colors = []
for i in range(len(bins)-1):
    bin_colors.append((bins[i+1] + bins[i]) / 2)
bin_colors = plt.cm.plasma(norm(bin_colors))

## figure and window sizes
fig = plt.figure(figsize=(16,8))

# prior, forecast and analysis windows
axp = fig.add_axes([.03, .35, .3, .55])
axf = fig.add_axes([.35, .35, .3, .55])
axa = fig.add_axes([.67, .35, .3, .55])

# rmse, histogram of forecast weights, histogram of analysis weights
axr = fig.add_axes([.04, .120, .28, .18])
axhf = fig.add_axes([.36, .120, .28, .18])
axha = fig.add_axes([.68, .120, .28, .18])

# slider and color bar windows
rax = fig.add_axes([0.13, 0.005, 0.1, 0.06])
ax_ens = fig.add_axes([0.450, 0.035, 0.1, 0.03])
ax_nanl = fig.add_axes([0.450, 0.005, 0.1, 0.03])
ax_bstd = fig.add_axes([0.770, 0.005, 0.1, 0.03])
ax_rstd = fig.add_axes([0.770, 0.035, 0.1, 0.03])
ax_cbar = fig.add_axes([0.67, 0.950, .30, .013])

# tick parameters 
axa.tick_params(
        labelleft=False,
        labelright=True,
        right=True,
        labelsize=14)
axf.tick_params(
        labelleft=False,
        right=True,
        labelsize=14)
axp.tick_params(
        labelsize=14,
        right=True)
axr.tick_params(
        labelsize=14,
        right=True)
axhf.tick_params(
        labelleft=False,
        right=True,
        labelsize=14)
axha.tick_params(
        labelleft=False,
        labelright=True,
        right=True,
        labelsize=14)
ax_cbar.tick_params(
        labelsize=14)
ax_bstd.tick_params(
        labelsize=15)


########################################################################################################################
## Numerical simulation code
########################################################################################################################

def Ikeda_map(X_0, u):
    """The array X_0 will define the initial condition and the parameter u controls the chaos of the map.
    
    This the Ikeda map propagator, it will return the forward evolution X_1 of the intial states X_0. 
    This is vectorized to run on an array of dimension 2 X ensemble size, assume a 1-D ensemble for a single state.
    The parameter u controls the chaos in the map."""
    
    t_1 = 0.4 - 6 / (1 + np.sum(X_0*X_0, axis=0) )
    
    x_1 = 1 + u * (X_0[0, :] * np.cos(t_1) - X_0[1, :] * np.sin(t_1))
    y_1 = u * (X_0[0, :] * np.sin(t_1) + X_0[1, :] * np.cos(t_1))
                 
    X_1 = np.array([x_1, y_1])
    
    return X_1


########################################################################################################################

def pf_RMSE(x_t, ens, ens_w):
    """Compute the RMSE of the weighted ensemble (sample) mean versus the true state x_t."""

    # find the weighted mean
    ens = np.sum(ens * ens_w, axis=1)
    
    # compute the root mean square deviation of the weighted mean from the true state
    RMSE = x_t - ens
    RMSE = np.sqrt(np.mean(RMSE**2))

    return RMSE


########################################################################################################################

def run_pf(B_std, R_std, k, ens_n, analysis):
    """This runs the full numerical simulation including generating observations and performing assimilation.

    This is a function of the intial parameters, which can be controlled by sliders elsewhere in the code."""

    # set to integer for indexing
    k = int(k)
    ens_n = int(ens_n)

    # define the static background and observational error covariances
    P_0 = B_std**2 * np.eye(2)
    R = R_std**2 * np.eye(2)

    # set a random seed for the reproducibility of the observations
    np.random.seed(1)
    
    # we define the mean for the background
    x_b = np.array([0,0])
    
    # and the initial condition of the real state as a random draw from the prior
    x_t = np.random.multivariate_normal(x_b, P_0, size=1).transpose()
    x_ts = np.zeros([2,k])
    y_obs = np.zeros([2,k])
    RMSE = np.zeros(k)
    
    # define the Ikeda-Debian map parameter
    u = 0.9
    
    for i in range(k):
        # we forward propagate the true state
        x_t_0 = copy.copy(x_t)
        x_t = Ikeda_map(x_t, u)
        
        # store the true timeseries for computing DA statistics
        x_ts[:, i] = np.squeeze(x_t)
        
        # and generate a noisy observation
        y_obs[:, i] = np.squeeze(x_t) + np.random.multivariate_normal([0,0], R)
    
    
    # set a random seed for the reproducibility of the ensembles
    np.random.seed(2)

    # we define the ensemble as a random draw of the prior with equal weights
    ens_a = np.random.multivariate_normal(x_b, P_0, size=ens_n).transpose()
    ens_w_f = np.ones(ens_n) / ens_n
    ens_w_a = copy.copy(ens_w_f)
    
    for i in range(k):
        
        # copy the last analysis ensemble and weights for the prior
        ens_p = copy.copy(ens_a)
        ens_w_p = copy.copy(ens_w_a)

        # forward propagate the last analysis
        ens_f = Ikeda_map(ens_a, u)
        
        # store the forecast weights before update for the plotting in the final loop of the forecasts
        ens_w_f = copy.copy(ens_w_a)
        
        # compute the likelihood of the observation given the samples and update the weights
        # unless running the free forecast
        if analysis == 'Bootstrap':
            lik = np.exp( -.5 * np.sum( (ens_f.transpose() - y_obs[:, i])**2, axis=1) / R_std**2  ) \
                    / np.sqrt( (2*np.pi)**2 * np.linalg.det(R))
            ens_w_a = ens_w_f * lik

        # normalize the weights and update the analysis ensemble positions to the forecast
        ens_w_a = ens_w_a / np.sum(ens_w_a)
        ens_a = copy.copy(ens_f)

        # compute the analysis RMSE at the current time
        RMSE[i] = pf_RMSE(x_ts[:, i], ens_a, ens_w_a)

        # compute the effective number of samples
        N_eff = 1 / np.sum(ens_w_a**2)
        
        # resampling
        if N_eff < (0.1 * ens_n):
            C = np.zeros(ens_n)
            
            for j in range(ens_n):
                C[j] = sum(ens_w_a[0:j+1])

            xi_1 = np.random.uniform( low=0, high=(1/ens_n) )
            m = 0

            for j in range(ens_n):
                xi = xi_1 + j * (1 / ens_n)
                
                while xi > C[m]:
                    m = m + 1
                
                ens_a[:, j] = ens_f[:,m] + np.random.multivariate_normal([0,0], np.eye(2) * 0.001)

            ens_w_a = np.ones(ens_n) / ens_n

        # compute the analysis RMSE at the current time
        RMSE[i] = pf_RMSE(x_ts[:, i], ens_a, ens_w_a)

    running_RMSE = copy.copy(RMSE)
    
    for i in range(1, k):
        running_RMSE[i] = np.mean(RMSE[:i+1])

    x_t = np.squeeze(x_t)
    x_t_0 = np.squeeze(x_t_0)

    return ens_p, ens_w_p, ens_f, ens_w_f, ens_a, ens_w_a, y_obs, x_t_0, x_t, running_RMSE


########################################################################################################################
# Animations code
########################################################################################################################

def animate_pf(bstd, rstd, k, ensn, analysis):
    "Generates the animation of the filtering cycle up to the last time instance."

    # compute the particle filter statistics
    ens_p, ens_w_p, ens_f, ens_w_f, ens_a, ens_w_a, y_obs, x_t_0, x_t, running_RMSE = run_pf(bstd, rstd, k, ensn, 
                                                                                             analysis)
    # clear old data from axes
    axp.cla()
    axf.cla()
    axa.cla()
    axr.cla()
    axhf.cla()
    axha.cla()

    # sort weights so that they are plotted in ascending order
    indx = np.argsort(ens_w_p)
    ens_w_p = ens_w_f[indx]
    ens_p = ens_p[:, indx]

    indx = np.argsort(ens_w_f)
    ens_w_f = ens_w_f[indx]
    ens_f = ens_f[:, indx]

    indx = np.argsort(ens_w_a)
    ens_w_a = ens_w_a[indx]
    ens_a = ens_a[:, indx]

    # plot the heat maps
    l0 = axp.scatter(ens_p[0, :], ens_p[1, :], c=ens_w_f, s=20, alpha=.5, marker=',', cmap=wmap, norm=norm)
    l1 = axf.scatter(ens_f[0, :], ens_f[1, :], c=ens_w_f, s=20, alpha=.5, marker=',', cmap=wmap, norm=norm)
    l3 = axa.scatter(ens_a[0, :], ens_a[1, :], c=ens_w_a, s=20, alpha=.5, marker=',', cmap=wmap, norm=norm)
    l2 = axa.scatter(y_obs[0, -1], y_obs[1, -1], c='r', s=46, marker='d', edgecolor='k')
    l4 = axp.scatter(x_t_0[0], x_t_0[1], c='#00ccff', s=90, marker='X', edgecolor='k')
    l4 = axf.scatter(x_t[0], x_t[1], c='#00ccff', s=90, marker='X', edgecolor='k')
   
    # plot the running RMSE
    if k == 1:
        l5 = axr.scatter(k, running_RMSE)
        l6 = axr.hlines(rstd, 0.5, k+0.5, color='r')
    else:
        l5 = axr.plot(range(1,k+1), running_RMSE) 
        l6 = axr.hlines(rstd, 0.5, k+0.5, color='r') 
   
    # plot the observation error standard deviation
    axa.add_patch(Ellipse(y_obs[:,-1], 2*rstd, 2*rstd, ec='r', fc='none'))

    # relative frequency of forecast weights plot
    hist_w, hist_bin = np.histogram(ens_w_f, bins=bins)
    hist_w = hist_w / ensn
    axhf.bar(bins[:-1], hist_w, width=bin_widths, color=bin_colors, align='edge')

    # relative frequency of analysis weights plot
    hist_w, hist_bin = np.histogram(ens_w_a, bins=bins)
    hist_w = hist_w / ensn
    axha.bar(bins[:-1], hist_w, width=bin_widths, color=bin_colors, align='edge')
    
    # state vector plot settings
    axf.set_xlim([-2.3, 3.3])
    axa.set_xlim([-2.3, 3.3])
    axp.set_xlim([-2.3, 3.3])
    axf.set_ylim([-3.3, 2.3])
    axa.set_ylim([-3.3, 2.3])
    axp.set_ylim([-3.3, 2.3])

    # rmse plot settings
    axr.set_xlim([0.5, k+0.5])
    axr.set_ylim([10e-4, 7.0])
    axr.set_yscale('log')
    axr.set_yticks([1,.1,.01,.001])
    
    # histogram plot settings
    axhf.set_xlim([10e-17,1])
    axha.set_xlim([10e-17,1])
    axhf.set_xscale('log')
    axha.set_xscale('log')
    axha.set_ylim([10e-4, 7.0])
    axhf.set_ylim([10e-4, 7.0])
    axhf.set_yscale('log')
    axha.set_yscale('log')
    axhf.set_yticks([1,.1,.01,.001])
    axha.set_yticks([1,.1,.01,.001])
    fig.canvas.draw_idle()

    return [l4, l2, l6, l5]


########################################################################################################################

def update(val):
    """Defines the update when parameter sliders are adjusted"""

    # infer the new values
    bstd = s_bstd.val
    rstd = s_rstd.val
    k = int(s_nanl.val)
    ensn = s_ens.val
    analysis = radio.value_selected

    # animate
    animate_pf(bstd, rstd, k, ensn, analysis)


########################################################################################################################
# Initiate the plot below
########################################################################################################################

# first simulation
lines = animate_pf(bstd_0, rstd_0, k_0, ensn_0, analysis_0)
cmap = cm.ScalarMappable(norm=norm, cmap=wmap)
plt.colorbar(cmap, cax=ax_cbar, orientation='horizontal')

# define the parameter sliders
s_ens = Slider(ax_ens, 'Sample size', 25, 500, valinit=ensn_0, valstep=25)
s_bstd = Slider(ax_bstd, 'Initial error Std', .1, 1.0, valinit=bstd_0, valstep=0.1)
s_rstd = Slider(ax_rstd, 'Obs error Std', .1, 1.0, valinit=rstd_0, valstep=0.1)
s_nanl = Slider(ax_nanl, 'Forecast', 1, 100, valinit=k_0, valstep=1)
radio = RadioButtons(rax, ('Bootstrap', 'Free forecast'), active=0)

# new parameter values in sliders update the figure
s_ens.on_changed(update)
s_bstd.on_changed(update)
s_rstd.on_changed(update)
s_nanl.on_changed(update)
radio.on_clicked(update)

# set plot labels and legend
labels = ['Truth', 'Observation', 'Obs Error Std', 'Posterior RMSE']
fig.legend(lines, labels, bbox_to_anchor=[0.0280,0.990], loc='upper left', ncol=4, fontsize=17)
plt.figtext(.18, .87, r'Last posterior sample', horizontalalignment='center', verticalalignment='center', fontsize=17)
plt.figtext(.50, .87, r'Forecast sample', horizontalalignment='center', verticalalignment='center', fontsize=17)
plt.figtext(.82, .87, r'New posterior sample', horizontalalignment='center', verticalalignment='center', fontsize=17)
plt.figtext(.18, .136, r'Running average versus time', horizontalalignment='center', verticalalignment='center', fontsize=16)
plt.figtext(.50, .282, r'Relative frequency - forecast weights', horizontalalignment='center', verticalalignment='center', fontsize=16)
plt.figtext(.82, .282, r'Relative frequency - posterior weights', horizontalalignment='center', verticalalignment='center', fontsize=16)
plt.figtext(.82, .983, r'Sample weight', horizontalalignment='center', verticalalignment='center', fontsize=17)
plt.show()
