# https://community.plotly.com/t/rotating-3d-plots-with-plotly/34776/2
# https://community.plotly.com/t/how-to-export-animation-and-save-it-in-a-video-format-like-mp4-mpeg-or/64621/2
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle

def GetFormated(x_train_l, index_x,index_y,index_z,y_train_l):
    xtmp=np.array([i for i in np.atleast_2d((x_train_l[:].T[index_x]))])
    ytmp=[i for i in np.atleast_2d((x_train_l[:].T[index_y]))]
    ztmp=[i for i in np.atleast_2d((x_train_l[:].T[index_z]))]

    for i in xtmp:
        x=i
    for i in ytmp:
        y=i
    for i in ztmp:
        z=i
    c=y_train_l.reshape((len(y_train_l),))
    return x,y,z,c

def draw_samples_distribution_3D(x,y,z,c):
    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers",marker=dict(
        size=8,
        color=c,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8,
        showscale=True
    )))
    x_eye = -1.25
    y_eye = 2
    z_eye = 1.0


    fig.show()

def draw_samples_distribution_3D_rotating(x,y,z,c):
    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z,mode="markers",
        marker=dict(
            size=8,
            color=c,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8,
            showscale=True
        )
    ))
    x_eye = 0
    y_eye = 2
    z_eye = 1.0

    fig.update_layout(
            title="",
            width=600,
            height=600,
            scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
            updatemenus=[
                dict(
                    type="buttons",
                showactive=False,
                y=1,
                x=0.8,
                xanchor="left",
                yanchor="bottom",
                pad=dict(t=45, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=5, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ],
    )


    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    frames = []
    pil_frames = []
    for t in np.arange(0, 3.14, 0.025):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
    fig.frames = frames

    fig.show()

def DrawParameterDependencies(ax, x_train_l, y_train_l, x_label='Radius', y_label='Thickness',labels=['Radius','Thickness','Phi', 'Theta', 'Length']):
    ltmp=np.array(labels)
    x_idx=np.where(ltmp==x_label)[0][0]
    y_idx=np.where(ltmp==y_label)[0][0]
    colors = cm.hsv(y_train_l/max(y_train_l))
    x,y,z,c = GetFormated(x_train_l,x_idx,y_idx,2,y_train_l)
    c=1
    ax.grid(True,linestyle='-',color='0.75')
    # scatter with colormap mapping to z value
    a = ax.scatter(x,y,s=20,c=colors, marker = 'o', cmap = cm.jet );
    
    #fig.colorbar(a)
    #plt.show()

def DrawParameterCorrelations(x_train_l,y_train_l,labels):
    fig, axs = plt.subplots(len(labels), len(labels),figsize=(9,9), layout="constrained")

    colmap = cm.ScalarMappable(cmap=cm.hsv)
    colmap.set_array(y_train_l)

    for i in range(len(labels)):
        for j in range(len(labels)):
            DrawParameterDependencies(axs[j,i], x_train_l, y_train_l,labels[i],labels[j],labels)
            if i==0:
                axs[j,i].set_ylabel(labels[j])
            if j==len(labels)-1:
                axs[j,i].set_xlabel(labels[i])

    fig.colorbar(colmap)

def DrawModeratorConfiguration(radius, thickness, npanels, theta, length):
    figure, axes = plt.subplots( 1 )
    axes.set_aspect( 1 )
    axes.set_facecolor([135./265.,151./265.,154./265.])
    axes.set_xlim(-300,300)
    axes.set_ylim(-300,300)
    theta2 = np.linspace( 0 , 2 * np.pi , 150 )
    r = 265
    a = r * np.cos( theta2 )
    b = r * np.sin( theta2 )
    axes.plot( a, b, color=[182./265., 182./265., 182./265.] )
    axes.fill_between(a,b,color=[182./265., 182./265., 182./265.])
    
    r2 = 90
    a2 = r2 * np.cos( theta2 )
    b2 = r2 * np.sin( theta2 )
    axes.plot( a2, b2, color=[172./265., 165./265., 162./265.])
    axes.fill_between(a2,b2,color=[128./265., 155./265., 151./265.])

    phi = 360./npanels
    #axes.grid(color='lightgray', linestyle='-', linewidth=1)
    for i in range(int(npanels)):
        
        ang=180+i*phi+theta
        x=-thickness/2+radius*np.cos(i* phi * np.pi/180)
        y=-length/2+ radius*np.sin(i*phi* np.pi/180)
        axes.add_patch(Rectangle((x,y),thickness,length, rotation_point='center',
                    angle= ang,
                    edgecolor='none',
                    facecolor=[0./265., 125./265., 115./265.],
                    lw=4))
    axes.text(-200,-380, f'r= {round(radius,1)}, d={round(thickness,1)}, N={round(npanels,0)}, '+r'$\theta$'+f'={round(theta,1)}, L={round(length,1)}', fontsize=8)

def Rotate(x,y,theta):
    x_new=x * np.cos(theta) - y * np.sin(theta)
    y_new=x * np.sin(theta) + y * np.cos(theta)
    return x_new, y_new

def GetPointsXY(radius,thickness, npanels,theta, length):
    l = length/2
    # Convert theta to radians
    theta = theta * np.pi/180.
    phi = 360./float(npanels)
    phi = phi* np.pi/180.
    x=[]
    y=[]
    for i in range(npanels):
        x.append(0)
        y.append(0)
        x.append(radius*np.cos(i*phi))
        y.append(radius*np.sin(i*phi))
        x.append(radius*np.cos(i*phi) + Rotate(0.,length/2, np.pi + i*phi + theta)[0])
        y.append(radius*np.sin(i*phi) + Rotate(0.,length/2, np.pi + i*phi + theta)[1])
        x.append(radius*np.cos(i*phi) + Rotate(0.,-length/2, np.pi + i*phi + theta)[0])
        y.append(radius*np.sin(i*phi) + Rotate(0.,-length/2, np.pi + i*phi + theta)[1])

        x.append(radius*np.cos(i*phi) + Rotate(thickness/2,length/2, np.pi + i*phi + theta)[0])
        y.append(radius*np.sin(i*phi) + Rotate(thickness/2,length/2, np.pi + i*phi + theta)[1])
        x.append(radius*np.cos(i*phi) + Rotate(thickness/2,-length/2, np.pi + i*phi + theta)[0])
        y.append(radius*np.sin(i*phi) + Rotate(thickness/2,-length/2, np.pi + i*phi + theta)[1])

        x.append(radius*np.cos(i*phi) + Rotate(-thickness/2,length/2, np.pi + i*phi + theta)[0])
        y.append(radius*np.sin(i*phi) + Rotate(-thickness/2,length/2, np.pi + i*phi + theta)[1])
        x.append(radius*np.cos(i*phi) + Rotate(-thickness/2,-length/2, np.pi + i*phi + theta)[0])
        y.append(radius*np.sin(i*phi) + Rotate(-thickness/2,-length/2, np.pi + i*phi + theta)[1])
        
    return x, y
#fig = plt.figure(figsize=(12, 8))
#ax = fig.add_subplot(111, projection='3d')
#colors = cm.hsv(y_train_l/max(y_train_l))
#colmap = cm.ScalarMappable(cmap=cm.hsv)
#colmap.set_array(y_train_l)
#x_l=np.atleast_2d((x_train_l[:].T[0])).T
#y_l=np.atleast_2d((x_train_l[:].T[1])).T
#z_l=np.atleast_2d((x_train_l[:].T[2])).T
#x_h=np.atleast_2d((x_train_h[:].T[0])).T
#y_h=np.atleast_2d((x_train_h[:].T[1])).T
#z_h=np.atleast_2d((x_train_h[:].T[2])).T
#ax.scatter(x_l, y_l, z_l, c=colors, marker='o')
#ax.scatter(x_h, y_h, z_h,color='black',marker='x')
#print(np.atleast_2d((x_train_h[:].T[2])))
#cb = fig.colorbar(colmap)
#ax.set_xlabel('radius')
#ax.set_ylabel('thickness')
#ax.set_zlabel('N panels')
#plt.clabel('f (x)')
#plt.xlim([0, 260])
#plt.legend(['Low fidelity', 'High fidelity'])
#plt.title('High and low fidelity functions');
#plt.show()

def DrawModerator(radius, thickness, npanels, length, theta,draw_radius=0):

    figure, axes = plt.subplots( 1 )
    axes.set_aspect( 1 )
    alpha = np.linspace( 0 , 2 * np.pi , 150 )
    r = 265
    a = r * np.cos( alpha )
    b = r * np.sin( alpha )
    axes.plot( a, b, color='gray' )
    
    r2 = 90
    a2 = r2 * np.cos( alpha )
    b2 = r2 * np.sin( alpha )
    axes.plot( a2, b2, color='gray')

    if draw_radius >0:
        r2 = draw_radius
        a2 = r2 * np.cos( alpha )
        b2 = r2 * np.sin( alpha )
        axes.plot( a2, b2, color='blue')
    
    phi = 2*np.pi/npanels
    for i in range(npanels):
        center_x = np.cos(phi*i)*radius
        center_y = np.sin(phi*i)*radius

        #plt.gca().add_patch(Rectangle((center_x-thickness/2,center_y-length/2),thickness, length, color='gray'))
        plt.gca().add_patch(Rectangle((center_x-thickness/2,center_y-length),thickness, length*2, angle=-theta, rotation_point='center'))

    return [figure, axes]