#nSamples=100
#counter = 0
#for m in range(39,nSamples):
#    filename=f'/global/cfs/cdirs/legend/users/aschuetz/simulation/out/large_reentrance_tube/low_fidelity/scans_5dim/neutron-sim-D4-LF-{m}_'
    #print(filename)
#    files=get_all_files(filename,ending='.csv')
    #print(len(files))
#    tmp=read_design_parameters(filename)
#    tmp2=f"{tmp}"
    #print(tmp2)
#    counter+=len(files)
#    print(m)
#    for file in tqdm(files):
    #    print(file)
    #    print("Reading in data...")
#        df_in = pd.read_csv(file,skiprows=1)
#        test=df_in['nC_A'].to_numpy()
#        df_in = df_in.loc[:, ~df_in.columns.str.contains('^Unnamed')]
#        f = open(file, "w")#
#        f.write(f"# [LF, {tmp2[1:-1]}] # mode design r d Npanels angle L H z V"+"\n")
#        f.close()
#        df_in.to_csv(file,mode='a')

#print("total files ",counter)

# https://community.plotly.com/t/rotating-3d-plots-with-plotly/34776/2
# https://community.plotly.com/t/how-to-export-animation-and-save-it-in-a-video-format-like-mp4-mpeg-or/64621/2
import plotly.graph_objects as go
import numpy as np


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