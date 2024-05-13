import numpy as np
np.random.seed(20)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
import sys
import os
sys.path.append('../utilities')
import simulation_utils
import plotting_utils
from matplotlib import cm
from scipy.optimize import fmin

import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel
from emukit.experimental_design.acquisitions import ModelVariance,IntegratedVarianceReduction
from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter, InformationSourceParameter
from emukit.core.acquisition import Acquisition
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch



# Construct a linear multi-fidelity model

def linear_multi_fidelity_model(x_train_l, y_train_l, noise_lf, x_train_h, y_train_h, noise_hf):
    X_train, Y_train = convert_xy_lists_to_arrays([x_train_l,x_train_h], [y_train_l,y_train_h])
    

    kernels = [GPy.kern.RBF(X_train[0].shape[0]-1),GPy.kern.RBF(1, variance=noise_hf)]
    #kernels = [GPy.kern.RBF(4),GPy.kern.RBF(1)]

    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)

    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(noise_lf)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0.0)

    ## Wrap the model using the given 'GPyMultiOutputWrapper'
    lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=100, verbose_optimization=False)

    ## Fit the model
    lin_mf_model.optimize()
    mf_model = lin_mf_model
    return mf_model

# Acqusition Curve

# Define cost of different fidelities as acquisition function
class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        x_cost = np.array([self.costs[i] for i in fidelity_index])
        return x_cost[:, None]
    
    @property
    def has_gradients(self):
        return True
    
    def evaluate_with_gradients(self, x):
        return self.evalute(x), np.zeros(x.shape)


def max_acquisition_func(mf_model, xlow, xhigh, labels):
    ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
    spaces_tmp = []
    for i in range(len(labels)):
        spaces_tmp.append(ContinuousParameter(labels[i], xlow[i], xhigh[i]))
    
    
    
    spaces_tmp.append(InformationSourceParameter(2))
    parameter_space = ParameterSpace(spaces_tmp)
    cost_acquisition = Cost([1., 2000.])
    us_acquisition = MultiInformationSourceEntropySearch(mf_model, parameter_space) / cost_acquisition
    
    ## Compute mean and variance predictions
    #spaces_tmp.append(DiscreteParameter("f",[1]))
    #parameter_space = ParameterSpace(spaces_tmp)
    #us_acquisition = IntegratedVarianceReduction(mf_model, parameter_space)
    
    optimizer = GradientAcquisitionOptimizer(parameter_space)
    x_new, _ = optimizer.optimize(us_acquisition)
    return x_new, us_acquisition

def add_sample(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, labels, sample, version='v1'):
    data=pd.read_csv(f'in/Ge77_rates_new_samples_{version}.csv', skiprows=1)

    row_h=list(set(data.index[data['Sample'] == sample].tolist()).intersection(set(data.index[data['Mode'] == 1.0].tolist())))
    row_l=list(set(data.index[data['Sample'] == sample].tolist()).intersection(set(data.index[data['Mode'] == 0.0].tolist())))
    for i in row_h:
        x_new=[]
        for l in labels:
            x_new.append(data.iloc[i][l])
        x_train_h=np.append(x_train_h,np.array([x_new]),axis=0)
        y_train_h=np.append(y_train_h,np.array([[data.iloc[i]['Ge-77_CNP']]]),axis=0)
        #print(f"Adding HF sample at {x_new} with Ge-77 Rate of {y_train_h[len(y_train_h)-1]}")
        
    for i in row_l:
        x_new=[]
        for l in labels:
            x_new.append(data.iloc[i][l])
        x_train_l=np.append(x_train_l,np.array([x_new]),axis=0)
        y_train_l=np.append(y_train_l,np.array([[data.iloc[i]['Ge-77_CNP']]]),axis=0)
        #print(f"Adding LF sample at {x_new} with Ge-77 Rate of {y_train_l[len(y_train_l)-1]}")
    
    
    X_train, Y_train = convert_xy_lists_to_arrays([x_train_l,x_train_h], [y_train_l,y_train_h])
    mf_model.set_data(X_train, Y_train)
    return x_train_l, y_train_l, x_train_h, y_train_h, mf_model

def add_samples(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, labels, version='v1', sample=-1):
    sample_stop=get_num_new_samples(version)[0]
    if sample < sample_stop and sample >= 0:
        sample_stop=sample

    for i in range(sample_stop+1):
        x_train_l, y_train_l, x_train_h, y_train_h, mf_model = add_sample(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, labels, i, version)
    return x_train_l, y_train_l, x_train_h, y_train_h, mf_model

def get_num_new_samples(version='v1'):
    data=pd.read_csv(f'in/Ge77_rates_new_samples_{version}.csv', skiprows=1)
    nsamples_hf=len(data.index[data['Mode'] == 1.].tolist())
    nsamples_lf=len(data.index[data['Mode'] == 0.].tolist())
    return [nsamples_hf,nsamples_lf]


# Get an evaluation of the HF model prediction at a certain point

def hf_model(x, mf_model):
    x_search=[]
    x_search.append([x[0],x[1],x[2],x[3],x[4]])
    x_search.append([0,0,0,0,0]) # model needs at least two points to evaluate prediction
    x_search=(np.atleast_2d(x_search))
    X_plot = convert_x_list_to_array([x_search , x_search])
    mean,_ = mf_model.predict(X_plot[2:])
    mean=mean[0][0]
    return mean

def hf_model_uncertainty(x, mf_model):
    x_search=[]
    x_search.append([x[0],x[1],x[2],x[3],x[4]])
    x_search.append([0,0,0,0,0]) # model needs at least two points to evaluate prediction
    x_search=(np.atleast_2d(x_search))
    X_plot = convert_x_list_to_array([x_search , x_search])
    mean,var = mf_model.predict(X_plot[2:])
    var=var[0][0]
    var=np.sqrt(var)
    return var

# Get the minimum of the HF model prediction

def get_min(mf_model, x_0=[100, 5, 360, 0, 2]):
    def prediction(x):
        return hf_model(x,mf_model=mf_model)
    xmin=fmin(prediction, x_0)
    model_min=prediction(xmin)
    model_uncer_min=hf_model_uncertainty(np.array(xmin),mf_model)
    return [xmin, model_min, model_uncer_min]
