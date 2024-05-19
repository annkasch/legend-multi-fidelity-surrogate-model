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
import drawing_utils
from matplotlib import cm
from scipy.optimize import fmin
import matplotlib as mpl


from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays




# Drawings of the model predictions

def draw_model(x_train_l,y_train_l,x_train_h,y_train_h,mf_model,xmin, xmax, labels, factor=1., x_new_data=np.array([]),y_new_data=np.array([]), version='v1', x_fixed=[200., 10., 40., 45., 30.]):
    outlaying_indices=[]
    SPLIT = 100

    pdf=PdfPages(f'out/{version}/neutron-moderator-multi-fidelity-model_{version}.pdf')
    ncol=3
    nrow=int(np.ceil(len(labels)/ncol))
    fig,ax  = plt.subplots(nrow, ncol, figsize=(24, 12))
    ax = fig.axes
    for i in range(len(labels)):  

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

        #outlaying_indices.extend(IsWithinStdModelPredictionErrorBand(np.atleast_2d((x_train_l[:].T[i])).T,y_train_l,mf_model,factor))
        #outlaying_indices=list(set(outlaying_indices))

        ax[i].fill_between(x_tmp.flatten(), (lf_mean_mf_model - factor * lf_std_mf_model).flatten(), 
                        (lf_mean_mf_model + factor * lf_std_mf_model).flatten(), color='cadetblue', alpha=0.4)
    
        ax[i].fill_between(x_tmp.flatten(), (hf_mean_mf_model - hf_std_mf_model).flatten(), 
                        (hf_mean_mf_model + hf_std_mf_model).flatten(), color='coral', alpha=0.9)

        #print(lf_mean_mf_model)
        ax[i].plot(x_tmp, lf_mean_mf_model, '--', color='cadetblue')
        ax[i].plot(x_tmp, hf_mean_mf_model, '--', color='orangered')
        ax[i].plot(np.atleast_2d((x_train_l[:].T[i])).T, y_train_l, 'teal',marker=".",linewidth=0)
        ax[i].plot(np.atleast_2d((x_train_h[:].T[i])).T, y_train_h, 'orangered',marker=".",markersize=10,linewidth=0)

        if x_new_data.any() and y_new_data.any():
            ax[i].plot(x_new_data[i], y_new_data, color="red", marker=".", markersize=10, linewidth=0)
        ax[i].set_xlabel(labels[i])
        ax[i].set_ylabel(r'Neutron Capture Probability')
        ax[i].set_xlim(xmin[i], xmax[i])
        ax[i].legend([f'{factor} $\sigma$ Prediction Low Fidelity',f'{factor} $\sigma$ Prediction High Fidelity', 'Predicted Low Fidelity', 'Predicted High Fidelity', 'Low Fidelity Training Data', 'High Fidelity Training Data'])

    #fig.set_title('linear multi-fidelity model fit to low and high fidelity functions');
    pdf.savefig()
    pdf.close()
    #print(f'{np.round((1-len(outlaying_indices)/len(y_train_l))*100,1) }% within {factor} std.')

def draw_model_3D(mf_model,xmin, xmax, labels,index1=0,index2=3,index3=2,ml=1.0,version='v1', x_fixed=[160., 5., 40., 45., 100.]):
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


def draw_model_2D(mf_model,xmin, xmax, labels,index1=0,index2=3,ml=1.,version='v1', x_fixed=[160., 5., 40., 45., 100.]):
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
    ax.set_zlabel(r"Neutron Capture Probability")
    surf = ax.plot_surface(X,Y,lf_mean,rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False, shade=False)
    cbaxes = fig.add_axes([0.1, 0.2, 0.03, 0.6])
    fig.colorbar(surf, cax=cbaxes)
    plt.savefig(f'out/{version}/neutron-moderator-multi-fidelity-model-2D_{version}.png')

def draw_model_updated(fig, x_train_l,y_train_l,x_train_h,y_train_h,mf_model,xmin, xmax, labels, factor=1., x_new_data=np.array([]),y_new_data=np.array([]), version='v1', x_fixed=[200., 10., 40., 45., 30.]):
    outlaying_indices=[]
    SPLIT = 100
    ax = fig.axes

    pdf=PdfPages(f'out/{version}/updated-neutron-moderator-multi-fidelity-model_{version}.pdf')
    for i in range(0,len(labels)):   

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

        ax[i].fill_between(x_tmp.flatten(), (lf_mean_mf_model - factor * lf_std_mf_model).flatten(), 
                        (lf_mean_mf_model + factor * lf_std_mf_model).flatten(), color='cadetblue', alpha=0.4)
    
        ax[i].fill_between(x_tmp.flatten(), (hf_mean_mf_model - hf_std_mf_model).flatten(), 
                        (hf_mean_mf_model + hf_std_mf_model).flatten(), color='coral', alpha=0.9)

        ax[i].plot(x_tmp, lf_mean_mf_model, '--', color='teal')
        ax[i].plot(x_tmp, hf_mean_mf_model, '--', color='orangered')
        ax[i].plot(np.atleast_2d((x_train_l[:].T[i])).T, y_train_l, 'teal',marker=".",linewidth=0)
        ax[i].plot(np.atleast_2d((x_train_h[:].T[i])).T, y_train_h, 'orangered',marker=".",markersize=10,linewidth=0)

        if x_new_data.any() and y_new_data.any():
            ax[i].plot(x_new_data[i], y_new_data, color="orange", marker=".", markersize=10, linewidth=0)
        ax[i].set_xlabel(labels[i])
        ax[i].set_ylabel(r'Neutron Capture Probability')
        ax[i].set_xlim(xmin[i], xmax[i])
        ax[i].legend([f'{factor} $\sigma$ Prediction Low Fidelity',f'{factor} $\sigma$ Prediction High Fidelity', 'Predicted Low Fidelity', 'Predicted High Fidelity', 'Low Fidelity Training Data', 'High Fidelity Training Data'])
        #ax[i].set_title('linear multi-fidelity model fit to low and high fidelity functions');

    for i in range(len(labels),len(ax)): 
        ax[i].set_axis_off()
    pdf.savefig(fig)
    pdf.close()

    return fig

# Drawings of the aquisition function

def draw_acquisition_func(fig, us_acquisition, xlow, xhigh, labels, x_next=np.array([]), version='v1', x_fixed=[160,5,40,45,100]):
    SPLIT = 50
    df= pd.DataFrame()
    ax2 = fig.axes
    pdf=PdfPages(f'out/{version}/acquisition-function_{version}.pdf')
    
    for i in range(0,len(xlow)):
        ax2[i].set_title(f'Projected acquisition function - {labels[i]}');
        
        x_plot=[x_fixed[:] for l in range(0,SPLIT)]
        x_tmp = np.linspace(xlow[i], xhigh[i], SPLIT)
        for k in range(0,SPLIT):
            x_plot[k][i]=x_tmp[k]
        x_plot = (np.atleast_2d(x_plot))
        X_plot = convert_x_list_to_array([x_plot , x_plot])
        acq=us_acquisition.evaluate(X_plot[SPLIT:])
        color = next(ax2[i]._get_lines.prop_cycler)['color']
        ax2[i].plot(x_tmp,acq/acq.max(),color=color)
        acq=us_acquisition.evaluate(X_plot[:SPLIT])
        ax2[i].plot(x_tmp,acq/acq.max(),color=color,linestyle="--")
        
        #x_plot = np.linspace(xlow[i], xhigh[i], 500)[:, None]
        #x_plot_low = np.concatenate([np.atleast_2d(x_plot), np.zeros((x_plot.shape[0], 1))], axis=1)
        #x_plot_high = np.concatenate([np.atleast_2d(x_plot), np.ones((x_plot.shape[0], 1))], axis=1)
        #print(us_acquisition.evaluate(x_plot_low))
        #ax2[i].plot(x_plot_low[:, 0], us_acquisition.evaluate(x_plot_low), 'b')
        #ax2[i].plot(x_plot_high[:, 0], us_acquisition.evaluate(x_plot_high), 'r')
        
        if x_next.any():
            ax2[i].axvline(x_next[0,i], color="red", label="x_next", linestyle="--")
            ax2[i].text(x_next[0,i]+0.5,0.95,f'x = {round(x_next[0,i],1)}', color='red', fontsize=8)

        ax2[i].set_xlabel(f"{labels[i]}")
        ax2[i].set_ylabel(r"$f(x)$")
        
        #for j in range(int(len(ax2[i].lines)/2)):
        #    df[f'x{j}_{labels[i]}']=np.array(np.round(ax2[i].lines[2*j].get_xdata(),3))
        #    df[f'y{j}_{labels[i]}']=np.array(np.round(ax2[i].lines[2*j].get_ydata(),3))

    pdf.savefig(fig)
    pdf.close()
    df.to_csv(f'out/{version}/acquisition-function_{version}.csv')
    return fig
        
def draw_model_acquisition_func(fig1, fig2, labels, version='v1'):

    pdf=PdfPages(f'out/{version}/model-acquisition-evolution_{version}.pdf')
    ax1 = fig1.axes
    ax2 = fig2.axes

    for index, label in enumerate(labels):
        ncurves_per_fig_update=5
        nsamples=int((len(ax1[index].lines)+1)/ncurves_per_fig_update)
        
        fig5,(ax20,ax21)=plt.subplots(nrows=2,ncols=1,sharex=True)
        ax20.set_ylabel('Neutron Capture Probability')
        ax21.set_ylabel('a(x)')
        #ax21.set_ylim(0,0.07)
        ax21.set_xlabel(label)
        curve0=ax1[index].lines[len(ax1[index].lines)-3]
        ax20.plot(curve0.get_xdata(),curve0.get_ydata(),'.', markersize=3, color='teal')

        #colors=['blue','cyan','deepskyblue','midnightblue','teal','slategray', 'blueviolet','aquamarine','dodgerblue']
        #colors=['green','darkred','red','blue','sienna','tomato','orangered','indianred','firebrick']
        colors=['steelblue','mediumpurple','forestgreen','violet','y','orange','red','darkturquoise','saddlebrown','gray']
        
        for i in range(len(ax1[index].collections)):
            if ax1[index].collections[i].get_alpha()==0.4:
                poly=ax1[index].collections[i]
                x1=poly.get_paths()[0].vertices
                x1s=x1[:,0]
                split=int((len(x1s)-1)/2)
                x1s=x1s[:split]
                y1s=x1[:split,1]
                x2=poly.get_paths()[0].vertices
                y2s=x2[split+1:,1]
                y2s=y2s[::-1]
                ax20.fill_between(x1s.flatten(),y1s.flatten(),y2s.flatten(),color='cadetblue', alpha=0.4)
            
            if ax1[index].collections[i].get_alpha()==0.9:
                poly=ax1[index].collections[i]
                x1=poly.get_paths()[0].vertices
                x1s=x1[:,0]
                split=int((len(x1s)-1)/2)
                x1s=x1s[:split]
                y1s=x1[:split,1]
                x2=poly.get_paths()[0].vertices
                y2s=x2[split+1:,1]
                y2s=y2s[::-1]
                ax20.fill_between(x1s.flatten(),y1s.flatten(),y2s.flatten(),color='coral', alpha=0.9)
    
        for i in range(nsamples):
            idx=1
            if i > 0 :
                idx=(ncurves_per_fig_update-1)+ncurves_per_fig_update*(i-1)+1
            curve0=ax1[index].lines[idx-1]
            ax20.plot(curve0.get_xdata(),curve0.get_ydata(),'--',color='teal')
            curve0=ax1[index].lines[idx]
            ax20.plot(curve0.get_xdata(),curve0.get_ydata(),'--',color='orangered')
            if len(ax2[index].lines) > 0 :
                color = next(ax21._get_lines.prop_cycler)['color']
                curve1=ax2[index].lines[3*i]
                ax21.plot(curve1.get_xdata(),curve1.get_ydata(), color=color, label=f"sample #{i} HF (-- LF)")
                curve2=ax2[index].lines[3*i+1]
                ax21.plot(curve2.get_xdata(),curve2.get_ydata(), color=color, linestyle="--")

                if i < nsamples-1:
                    idx=(ncurves_per_fig_update-1)+ncurves_per_fig_update*(i)+4
                    curve_tmp=ax1[index].lines[idx]
                    x=curve_tmp.get_xdata()
                    diff=np.abs(curve1.get_xdata()-x)
                    closest_index = diff.argmin()
                    ax21.scatter(x,curve1.get_ydata()[closest_index],color=color)
                #else:
                #    curve2=ax2[index].lines[3*i+2]
                #    ax21.plot(curve2.get_xdata(),curve2.get_ydata(), '--', color='lightgray')
            ax21.legend(fontsize=7,loc='upper center', bbox_to_anchor=(0.5, -0.26),fancybox=True, shadow=False, ncol=4)
        curve0=ax1[index].lines[len(ax1[index].lines)-2]
        ax20.plot(curve0.get_xdata(),curve0.get_ydata(),'o', color='red')
        pdf.savefig(fig5)
    
    pdf.close()
    plt.show()


def read_acquisition_function(filename, labels):
    data=pd.read_csv(filename,index_col=0)
    nsamples=int(len(data.columns)/(2*len(labels)))

    fig = [plt.figure(figsize=(12,5)) for i in range(len(labels))]
    for idx,l in enumerate(labels):
        ax = fig[idx].gca()
        for i in range(nsamples):
            x=data[f'x{i}_{l}'].to_numpy()
            y=data[f'y{i}_{l}'].to_numpy()
            ax.plot(x,y)
            ax.set_xlabel(l)
            ax.set_ylim(0,1.1)