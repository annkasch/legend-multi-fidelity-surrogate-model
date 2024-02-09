import numpy as np
import pandas as pd
from NeutronSimulation import NeutronSimulation as sim
import argparse
import os

def main(filename_base, path_to_files, start, end): # python file
    print(filename_base, path_to_files, start, end)
    for m in range(start, end):
        filename=f'{path_to_files}/{filename_base}_{m:04d}'
        print(filename)
        sim_tmp=sim(filename)
        if len(sim_tmp.files)==0:
            continue
        [is_captured,time_0,x_0,y_0,z_0,px_0,py_0,pz_0,ekin_0,edep_0,vol_0,nCZ,nCA,time_t,x_t,y_t,z_t,px_t,py_t,pz_t,ekin_t,edep_t,vol_t,_,_]=sim_tmp.get_is_in_LAr(["time[mus]","x[m]","y[m]","z[m]","xmom[m]","ymom[m]","zmom[m]","ekin[eV]","edep[eV]","vol","nC_Z","nC_A"],"info")
        [fidelity, radius, thickness, npanels, _, theta, length, height, z_offset, volume] = sim_tmp.get_design()
        df = pd.DataFrame({'fidelity': fidelity, 'radius': radius, 'thickness': thickness,'npanels': npanels, 'theta': theta, 'length': length, 'height': height, 'z_offset': z_offset, 'volume': volume,'nC_Ge77': is_captured,'time_0[ms]':time_0,'x_0[m]':x_0,'y_0[m]': y_0,'z_0[m]': z_0,'px_0[m]': px_0,'py_0[m]': py_0,'pz_0[m]': pz_0,'ekin_0[eV]': ekin_0,'edep_0[eV]': edep_0,'vol_0': vol_0,'time_t[ms]':time_t,'x_t[m]':x_t,'y_t[m]': y_t,'z_t[m]': z_t,'px_t[m]': px_t,'py_t[m]': py_t,'pz_t[m]': pz_t,'ekin_t[eV]': ekin_t,'edep_t[eV]': edep_t,'vol_t': vol_t,'nC_A': nCA,'nC_Z': nCZ})
    
        if not os.path.exists(f'{path_to_files}/summary'):
            os.makedirs(f'{path_to_files}/summary')
        df.to_csv(f'{path_to_files}/summary/summary_{filename_base}_{m:04d}.csv')


 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="neutron-sim-D4-LF")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--path_to_files', type=str, default=".")
    args = parser.parse_args()

    main(args.filename, args.path_to_files, args.start, args.end)