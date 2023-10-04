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
from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter


# Construct a linear multi-fidelity model

def linear_multi_fidelity_model(x_train_l, y_train_l, x_train_h, y_train_h, noise_lf):
    X_train, Y_train = convert_xy_lists_to_arrays([x_train_l,x_train_h], [y_train_l,y_train_h])
    kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1)]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(noise_lf)
    
    #The Low Fidelity noise level need to be independently estimated. Here I provide a guess of 5e-7
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)

    ## Wrap the model using the given 'GPyMultiOutputWrapper'
    lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=20)

    ## Fit the model
    lin_mf_model.optimize()
    mf_model = lin_mf_model
    return mf_model

# Acqusition Curve
# - The acquisition curve is an important part of the active learning process. The next step we try using HF simulation dependes on where the acquisiton function takes its maximal value.
# - Define a parameter space (here we only have a single parameter radius), we need to add another parameter fidelity into our data, this parameter is always 1, meaning that we always run acquisition function on the high fidelity (1) space.
#- Note: it is important to deine the lower and upper range of our optimization parameter. Looking at the previous plot, anything below 80cm is probably unphysical, therefore we should not waste any attempt there. I selected a region of 90-250 to run the acquisition function. Selecting the wrong range could significantly change the shape of acquisition function.

def MaximizeAcquisitionFunction(mf_model, xlow, xhigh, labels):
    ## Here we run a gradient-based optimizer over the acquisition function to find the next point to attempt. 
    parameter_space = ParameterSpace([ContinuousParameter(labels[0], xlow[0], xhigh[0]), ContinuousParameter(labels[1], xlow[1],xhigh[1]),
                                  ContinuousParameter(labels[2], xlow[2], xhigh[2]), ContinuousParameter(labels[3], xlow[3],xhigh[3]),
                                  ContinuousParameter(labels[4], xlow[4], xhigh[4]), DiscreteParameter("f",[1])])
    ## Compute mean and variance predictions
    us_acquisition = IntegratedVarianceReduction(mf_model, parameter_space)
    
    optimizer = GradientAcquisitionOptimizer(parameter_space)
    x_new, _ = optimizer.optimize(us_acquisition)
    return x_new, us_acquisition

def AddNewSample(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, sample, version='v1'):
    x_new_data=np.array([])
    y_new_data_h=np.array([])
    data=pd.read_csv(f'in/Ge77_rates_new_samples_{version}.csv', skiprows=1)
    row_h=list(set(data.index[data['Sample'] == sample].tolist()).intersection(set(data.index[data['Mode'] == 'HF'].tolist())))
    row_l=list(set(data.index[data['Sample'] == sample].tolist()).intersection(set(data.index[data['Mode'] == 'LF'].tolist())))

    for i in row_h:
        x_new_data_h=np.array([[data.iloc[i]['Radius[cm]'],data.iloc[i]['Thickness[cm]'],data.iloc[i]['NPanels'],data.iloc[i]['Theta[deg]'],data.iloc[i]['Length[cm]'], 1.]])
        y_new_data_h=np.array([[data.iloc[i]['Ge77-Rate[nucleus/(kg yr)]']]])
        x_train_h = np.append(x_train_h,[x_new_data_h[0][:-1]],axis=0)
        y_train_h = np.append(y_train_h,y_new_data_h,axis=0)
        print(f"Adding HF sample at {x_new_data_h} with Ge-77 Rate of {y_new_data_h}")
    for i in row_l:
        x_new_data_l=np.array([[data.iloc[i]['Radius[cm]'],data.iloc[i]['Thickness[cm]'],data.iloc[i]['NPanels'],data.iloc[i]['Theta[deg]'],data.iloc[i]['Length[cm]'], 1.]])
        y_new_data_l=np.array([[data.iloc[i]['Ge77-Rate[nucleus/(kg yr)]']]])
        x_train_l = np.append(x_train_l,[x_new_data_l[0][:-1]],axis=0)
        y_train_l = np.append(y_train_l,y_new_data_l,axis=0)
        print(f"Adding LF sample at {x_new_data_l} with Ge-77 Rate of {y_new_data_l}")
    
    X_train, Y_train = convert_xy_lists_to_arrays([x_train_l,x_train_h], [y_train_l,y_train_h])
    mf_model.set_data(X_train, Y_train)
    return x_train_l, y_train_l, x_train_h, y_train_h, x_new_data, y_new_data_h, mf_model

def AddNewSamples(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, version='v1'):
    for i in range(GetNumberOfNewSamples[0]):
        x_train_l, y_train_l, x_train_h, y_train_h, _, _, mf_model = AddNewSample(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, i, version)
    return x_train_l, y_train_l, x_train_h, y_train_h, mf_model

def GetNumberOfNewSamples(version='v1'):
    data=pd.read_csv(f'in/Ge77_rates_new_samples_{version}.csv', skiprows=1)
    nsamples_hf=len(data.index[data['Mode'] == 'HF'].tolist())
    nsamples_lf=len(data.index[data['Mode'] == 'LF'].tolist())
    return [nsamples_hf,nsamples_lf]

def ActiveLearning(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, xmin, xmax, xlow, xhigh, labels, factor, fig1, fig2, version, x_fixed, sample):
    # add new data point to training data and update model with new training data
    x_new_data=np.array([])
    y_new_data_h=np.array([])
    if sample > 0:
        x_train_l, y_train_l, x_train_h, y_train_h, x_new_data, y_new_data_h, mf_model = AddNewSample(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, sample-1, version)   
    # run the model drawing
    DrawMultiFideliyModel(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, xmin, xmax, labels, factor, x_new_data, y_new_data_h, version)#
    fig1 = DrawUpdatedMultiFideliyModel(fig1, x_train_l, y_train_l, x_train_h, y_train_h, mf_model, xmin, xmax, labels, factor, x_new_data, y_new_data_h, version)
    DrawMultiFideliyModel2D(mf_model,xmin, xmax,index1=0,index2=2,labels=labels,version=version,ml='LF', x_fixed=x_fixed)
    DrawMultiFideliyModel3D(mf_model,xmin, xmax,index1=0,index2=2,index3=3,labels=labels,version=version,ml='LF', x_fixed=x_fixed)
    
    # find the next data point
    x_next_sample, us_acquisition = MaximizeAcquisitionFunction(mf_model, xlow, xhigh, labels)
    print(f'next suggested point to simulated is at: {x_next_sample}')
    simulation_utils.PrintGeant4Macro(x_next_sample[0][0],x_next_sample[0][1],x_next_sample[0][2],x_next_sample[0][3],x_next_sample[0][4],sample,'LF',version)
    simulation_utils.PrintGeant4Macro(x_next_sample[0][0],x_next_sample[0][1],x_next_sample[0][2],x_next_sample[0][3],x_next_sample[0][4],sample,'HF',version)
    
    fig2 = DrawAcquisitionFunction(fig2, us_acquisition, xlow, xhigh, labels, np.array(x_next_sample), version, x_fixed)
    DrawModelAcquisitionFunction(fig1, fig2, 0,'Radius [cm]', version)
    return x_train_l, y_train_l, x_train_h, y_train_h, mf_model, fig1, fig2

def ActiveLearningModelOnly(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, xmin, xmax, xlow, xhigh, labels, factor, fig1, fig2, version, x_fixed, sample):
    # add new data point to training data and update model with new training data
    x_new_data=np.array([])
    y_new_data_h=np.array([])
    if sample > 0:
        x_train_l, y_train_l, x_train_h, y_train_h, x_new_data, y_new_data_h, mf_model = AddNewSample(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, sample-1, version)
        
    # run the model drawing
    DrawMultiFideliyModel(x_train_l, y_train_l, x_train_h, y_train_h, mf_model, xmin, xmax, labels, factor, x_new_data, y_new_data_h, version)#
    fig1 = DrawUpdatedMultiFideliyModel(fig1, x_train_l, y_train_l, x_train_h, y_train_h, mf_model, xmin, xmax, labels, factor, x_new_data, y_new_data_h, version)
    DrawMultiFideliyModel2D(mf_model,xmin, xmax,index1=0,index2=2,labels=labels,version=version,ml='LF', x_fixed=x_fixed)
    DrawMultiFideliyModel3D(mf_model,xmin, xmax,index1=0,index2=2,index3=3,labels=labels,version=version,ml='LF', x_fixed=x_fixed)
    
    return x_train_l, y_train_l, x_train_h, y_train_h, mf_model, fig1, fig2


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

def GetMinimum(mf_model, x_0=[100, 5, 360, 0, 2]):
    def prediction(x):
        return hf_model(x,mf_model=mf_model)
    xmin=fmin(prediction, x_0)
    model_min=prediction(xmin)
    model_uncer_min=hf_model_uncertainty(np.array(xmin),mf_model)
    return [xmin, model_min, model_uncer_min]

# Drawings of the model predictions

def DrawMultiFideliyModel(x_train_l,y_train_l,x_train_h,y_train_h,mf_model,xmin, xmax, labels, factor=1., x_new_data=np.array([]),y_new_data=np.array([]), version='v1', x_fixed=[200., 10., 40., 45., 30.]):
    outlaying_indices=[]
    SPLIT = 100
    with PdfPages(f'out/{version}/neutron-moderator-multi-fidelity-model_{version}.pdf') as pdf:
        for i in range(0,5):   
            ## Compute mean and variance predictions
            x_plot=[x_fixed[:] for l in range(0,SPLIT)]
            x_tmp = np.linspace(xmin[i], xmax[i], SPLIT)
            for k in range(0,SPLIT):
                x_plot[k][i]=x_tmp[k]
            x_plot = (np.atleast_2d(x_plot))
            X_plot = convert_x_list_to_array([x_plot , x_plot])

            lf_mean_mf_model, lf_var_mf_model = mf_model.predict(X_plot[:SPLIT])
            lf_std_mf_model = np.sqrt(lf_var_mf_model)
        
            hf_mean_mf_model, hf_var_mf_model = mf_model.predict(X_plot[SPLIT:])
            hf_std_mf_model = np.sqrt(hf_var_mf_model)

            ## Plot posterior mean and variance of nonlinear multi-fidelity model

            plt.figure(figsize=(12,8))
            #outlaying_indices.extend(IsWithinStdModelPredictionErrorBand(np.atleast_2d((x_train_l[:].T[i])).T,y_train_l,mf_model,factor))
            #outlaying_indices=list(set(outlaying_indices))

            plt.fill_between(x_tmp.flatten(), (lf_mean_mf_model - factor * lf_std_mf_model).flatten(), 
                         (lf_mean_mf_model + factor * lf_std_mf_model).flatten(), color='y', alpha=0.3)
        
            plt.fill_between(x_tmp.flatten(), (hf_mean_mf_model - hf_std_mf_model).flatten(), 
                         (hf_mean_mf_model + hf_std_mf_model).flatten(), color='g', alpha=0.3)

            #print(lf_mean_mf_model)
            plt.plot(x_tmp, lf_mean_mf_model, '--', color='y')
            plt.plot(x_tmp, hf_mean_mf_model, '--', color='g')
            plt.plot(np.atleast_2d((x_train_l[:].T[i])).T, y_train_l, 'b',marker=".",linewidth=0)
            plt.plot(np.atleast_2d((x_train_h[:].T[i])).T, y_train_h, 'r',marker=".",markersize=10,linewidth=0)

            if x_new_data.any() and y_new_data.any():
                plt.scatter(x_new_data[0][i], y_new_data[0][0], color="orange")
            plt.xlabel(labels[i])
            plt.ylabel(r'$^{77(m)}$Ge Production Rate')
            plt.xlim(xmin[i], xmax[i])
            plt.legend([f'{factor} $\sigma$ Prediction Low Fidelity',f'{factor} $\sigma$ Prediction High Fidelity', 'Predicted Low Fidelity', 'Predicted High Fidelity', 'Low Fidelity Training Data', 'High Fidelity Training Data'])
            ##plt.legend()
            plt.title('linear multi-fidelity model fit to low and high fidelity functions');
            pdf.savefig()
            
    #print(f'{np.round((1-len(outlaying_indices)/len(y_train_l))*100,1) }% within {factor} std.')

def DrawMultiFideliyModel3D(mf_model,xmin, xmax, labels,index1=0,index2=3,index3=2,ml='HF',version='v1', x_fixed=[160., 5., 40., 45., 100.]):
    SPLIT=25
    i=index1
    j=index2
    m=index3
    xi_tmp = np.linspace(xmin[i], xmax[i], SPLIT)
    xj_tmp = np.linspace(xmin[j], xmax[j], SPLIT)
    xm_tmp = np.linspace(xmin[m], xmax[m], SPLIT)

    lf_mean=np.zeros((SPLIT*SPLIT,SPLIT))
    X, Y, Z = np.meshgrid(xi_tmp,xj_tmp, xm_tmp)

    for l,pair in enumerate(zip(Z,X,Y)): # -> range(SPLIT)
        vtmp=np.array(pair)

        for t in range(SPLIT): 
            x_plot=[x_fixed[:] for l in range(0,SPLIT)]
            x_plot = (np.atleast_2d(x_plot))
            x_plot[:].T[i]=vtmp[1][t]
            x_plot[:].T[j]=vtmp[2][t]
            x_plot[:].T[m]=vtmp[0][t]
            x_plot = (np.atleast_2d(x_plot))
            X_plot = convert_x_list_to_array([x_plot , x_plot])
            if ml=='LF':
                mean, _ = mf_model.predict(X_plot[SPLIT:])
            else:
                mean, _ = mf_model.predict(X_plot[:SPLIT])
            mean=np.array([k[0] for k in mean])
            lf_mean[l*SPLIT+t]=mean

    fig=plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel(labels[i])
    ax.set_ylabel(labels[j])
    ax.set_zlabel(labels[index3])
    cz = ax.scatter(X, Y, Z, c=lf_mean, linewidth=0, cmap=cm.plasma, antialiased=False)
    cbaxes = fig.add_axes([0.1, 0.2, 0.03, 0.6])
    plt.colorbar(cz, cax=cbaxes)
    plt.savefig(f'out/{version}/neutron-moderator-multi-fidelity-model-3D_{version}.png')

def DrawMultiFideliyModel2D(mf_model,xmin, xmax, labels,index1=0,index2=3,ml='HF',version='v1', x_fixed=[160., 5., 40., 45., 100.]):
    SPLIT=100
    i=index1
    j=index2

    xi_tmp = np.linspace(xmin[i], xmax[i], SPLIT)
    xj_tmp = np.linspace(xmin[j], xmax[j], SPLIT)


    lf_mean=np.zeros((SPLIT,SPLIT))
    X, Y = np.meshgrid(xi_tmp,xj_tmp)

    for l,pair in enumerate(zip(X,Y)): # -> range(SPLIT)
        vtmp=np.array(pair)

        x_plot=[x_fixed[:] for l in range(0,SPLIT)]
        x_plot = (np.atleast_2d(x_plot))
        x_plot[:].T[i]=vtmp[0]
        x_plot[:].T[j]=vtmp[1]

        x_plot = (np.atleast_2d(x_plot))
        X_plot = convert_x_list_to_array([x_plot , x_plot])
        if ml=='LF':
            mean, _ = mf_model.predict(X_plot[SPLIT:])
        else:
            mean, _ = mf_model.predict(X_plot[:SPLIT])
        mean=np.array([k[0] for k in mean])
        lf_mean[l]=mean

    fig=plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel(labels[i])
    ax.set_ylabel(labels[j])
    ax.set_zlabel(r"$^{77(m)}$Ge production rate")
    surf = ax.plot_surface(X,Y,lf_mean,rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False, shade=False)
    cbaxes = fig.add_axes([0.1, 0.2, 0.03, 0.6])
    fig.colorbar(surf, cax=cbaxes)
    plt.savefig(f'out/{version}/neutron-moderator-multi-fidelity-model-2D_{version}.png')

def DrawUpdatedMultiFideliyModel(fig, x_train_l,y_train_l,x_train_h,y_train_h,mf_model,xmin, xmax, labels, factor=1., x_new_data=np.array([]),y_new_data=np.array([]), version='v1', x_fixed=[200., 10., 40., 45., 30.]):
    outlaying_indices=[]
    SPLIT = 100
    with PdfPages(f'out/{version}/updated-neutron-moderator-multi-fidelity-model_{version}.pdf') as pdf:
        for i in range(0,5):   
            ## Compute mean and variance predictions
            x_plot=[x_fixed[:] for l in range(0,SPLIT)]
            x_tmp = np.linspace(xmin[i], xmax[i], SPLIT)
            for k in range(0,SPLIT):
                x_plot[k][i]=x_tmp[k]
            x_plot = (np.atleast_2d(x_plot))
            X_plot = convert_x_list_to_array([x_plot , x_plot])

            lf_mean_mf_model, lf_var_mf_model = mf_model.predict(X_plot[:SPLIT])
            lf_std_mf_model = np.sqrt(lf_var_mf_model)
        
            hf_mean_mf_model, hf_var_mf_model = mf_model.predict(X_plot[SPLIT:])
            hf_std_mf_model = np.sqrt(hf_var_mf_model)

            ## Plot posterior mean and variance of nonlinear multi-fidelity model
            ax2 = fig[i].gca()
            ax2.fill_between(x_tmp.flatten(), (lf_mean_mf_model - factor * lf_std_mf_model).flatten(), 
                         (lf_mean_mf_model + factor * lf_std_mf_model).flatten(), color='y', alpha=0.1)
        
            ax2.fill_between(x_tmp.flatten(), (hf_mean_mf_model - hf_std_mf_model).flatten(), 
                         (hf_mean_mf_model + hf_std_mf_model).flatten(), color='g', alpha=0.2)

            ax2.plot(x_tmp, lf_mean_mf_model, '--', color='y')
            ax2.plot(x_tmp, hf_mean_mf_model, '--', color='g')
            ax2.plot(np.atleast_2d((x_train_l[:].T[i])).T, y_train_l, 'b',marker=".",linewidth=0)
            ax2.plot(np.atleast_2d((x_train_h[:].T[i])).T, y_train_h, 'r',marker=".",markersize=10,linewidth=0)

            if x_new_data.any() and y_new_data.any():
                ax2.scatter(x_new_data[0][i], y_new_data[0][0],color="orange")
            ax2.set_xlabel(labels[i])
            ax2.set_ylabel(r'$^{77(m)}$ Production Rate')
            ax2.set_xlim(xmin[i], xmax[i])
            ax2.legend([f'{factor} $\sigma$ Prediction Low Fidelity',f'{factor} $\sigma$ Prediction High Fidelity', 'Predicted Low Fidelity', 'Predicted High Fidelity', 'Low Fidelity Training Data', 'High Fidelity Training Data'])
            ax2.set_title('linear multi-fidelity model fit to low and high fidelity functions');

            fig[i].savefig(f'out/{version}/updated-neutron-moderator-multi-fidelity-model-{i}_{version}.png')
            pdf.savefig(fig[i])
    return fig

# Drawings of the aquisition function

def DrawAcquisitionFunction(fig, us_acquisition, xlow, xhigh, labels, x_next=np.array([]), version='v1', x_fixed=[160,5,40,45,100]):
    SPLIT = 50
    df= pd.DataFrame()
    with PdfPages(f'out/{version}/activation-function_{version}.pdf') as pdf:
        for i in range(0,len(xlow)):
            ax2 = fig[i].gca()
            ax2.set_title(f'Projected activation function - {labels[i]}');
            x_plot=[x_fixed[:] for l in range(0,SPLIT)]
            x_tmp = np.linspace(xlow[i], xhigh[i], SPLIT)
            for k in range(0,SPLIT):
                x_plot[k][i]=x_tmp[k]
            x_plot = (np.atleast_2d(x_plot))
            X_plot = convert_x_list_to_array([x_plot , x_plot])

            acq=us_acquisition.evaluate(X_plot[SPLIT:])
            
            ax2.plot(x_tmp,acq/acq.max())
            if x_next.any():
                ax2.axvline(x_next[0,i], color="red", label="x_next", linestyle="--")
                ax2.text(x_next[0,i]+0.5,0.95,f'x = {round(x_next[0,i],1)}', color='red', fontsize=8)

            ax2.set_xlabel(f"{labels[i]}")
            ax2.set_ylabel(r"$f(x)$")
            pdf.savefig(fig[i])
            fig[i].savefig(f'out/{version}/activation-function-{labels[i]}_{version}.png')
            
            for j in range(int(len(ax2.lines)/2)):
                df[f'x{j}_{labels[i]}']=np.array(ax2.lines[2*j].get_xdata())
                df[f'y{j}_{labels[i]}']=np.array(ax2.lines[2*j].get_ydata())
    df.to_csv(f'out/{version}/acqueision-function_{version}.csv')
    return fig

def DrawModelAcquisitionFunction(fig1, fig2, index, xlabel='x', version='v1'):
    ax0=fig1[index].gca()
    ax1=fig2[index].gca()
    ncurves_per_fig_update=4
    nsamples=int(len(ax0.lines)/ncurves_per_fig_update)
    fig5,(ax20,ax21)=plt.subplots(nrows=2,ncols=1,sharex=True)
    ax20.set_ylabel('$^{77(m)}$Ge Production Rate')
    ax21.set_ylabel('a(x)')
    ax21.set_xlabel(xlabel)
    ax20.grid(color='lightgray', linestyle='-', linewidth=0.5)
    ax21.grid(color='lightgray', linestyle='-', linewidth=0.5)
    colors=['blue','cyan','deepskyblue','midnightblue','teal','slategray', 'blueviolet','aquamarine','dodgerblue']
    #colors1=['lightcoral','darkred','red','darkorange','sienna','tomato','orangered','indianred','firebrick']
    for i in range(len(ax0.collections)):
        if ax0.collections[i].get_alpha()==0.2:
            poly=ax0.collections[i]
            x1=poly.get_paths()[0].vertices
            x1s=x1[:,0]
            split=int((len(x1s)-1)/2)
            x1s=x1s[:split]
            y1s=x1[:split,1]
            x2=poly.get_paths()[0].vertices
            y2s=x2[split+1:,1]
            y2s=y2s[::-1]
            ax20.fill_between(x1s.flatten(),y1s.flatten(),y2s.flatten(),color='green',alpha=0.2)
    
    for i in range(nsamples):
        curve0=ax0.lines[(ncurves_per_fig_update*i)+1]
        ax20.plot(curve0.get_xdata(),curve0.get_ydata(), '--', color='g')
        if len(ax1.lines) > 0 :
            curve1=ax1.lines[2*i]
            y=curve1.get_ydata()/np.sum(curve1.get_ydata())
            ax21.plot(curve1.get_xdata(),y,color=colors[i])
            ymax=np.max(y)
            xmax=np.argmax(y)
            x=curve1.get_xdata()[xmax]
            if index > 0 and i < nsamples-1:
                curve_tmp=ax0.lines[len(ax0.lines)-1]
                x=curve_tmp.get_xdata()[3+i]
            if index == 0 or ( index > 0 and i < nsamples-1):
                ax21.scatter(x,ymax,color=colors[i])
        #ax21.plot(curve2.get_xdata(),curve2.get_ydata(), '--', color='lightgray')
    curve0=ax0.lines[len(ax0.lines)-1]
    
    ax20.plot(curve0.get_xdata()[0:4],curve0.get_ydata()[0:4],'o', color='black')
    for i in range(3,len(curve0.get_xdata())):
        ax20.plot(curve0.get_xdata()[i],curve0.get_ydata()[i],'o', color=colors[i-3])

    fig5.savefig(f"out/{version}/model-acquivision-evolution-{index}_{version}.png")