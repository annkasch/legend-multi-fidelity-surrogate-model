import numpy as np
import pandas as pd
from NeutronSimulation import NeutronSimulation as sim
import argparse
import os

def main(filename_base, path_to_files, tier, start, end, nprimaries): # python file

    for m in range(start, end):
        filename=f'{path_to_files}/{filename_base}-{m:04d}-{tier}_'
        print(filename)
        sim_tmp=sim(filename)
        nfiles_expected = int(nprimaries/25000)
        if len(sim_tmp.files)==0 or len(sim_tmp.files) != nfiles_expected:
            continue

        [is_captured,time_0,x_0,y_0,z_0,r_0,L_0,px_0,py_0,pz_0,ekin_0,edep_0,lnE0vsET,vol_0,nCZ,nCA,nsec,nsec_with_nC,nsteps,time_t,x_t,y_t,z_t,r_t,L_t,px_t,py_t,pz_t,ekin_t,edep_t,_,vol_t,_,_,_,_,_]=sim_tmp.get_is_in_LAr(["time[mus]","x[m]","y[m]","z[m]","r[m]","L[m]","xmom[m]","ymom[m]","zmom[m]","ekin[eV]","edep[eV]","ln(E0vsET)","vol","nC_Z","nC_A","nsecondaries","nsecondaries_with_nC","nsteps"],"summary")
        [fidelity, radius, thickness, npanels, _, theta, length, height, z_offset, volume] = sim_tmp.get_design()

        df = pd.DataFrame({'fidelity': fidelity, 'radius': radius, 'thickness': thickness,'npanels': npanels, 'theta': theta, 'length': length, 'height': height, 'z_offset': z_offset, 'volume': volume,'nC_Ge77': is_captured,'time_0[ms]':time_0,'x_0[m]':x_0,'y_0[m]': y_0,'z_0[m]': z_0,'r_0[m]': r_0,'L_0[m]': L_0,'px_0[m]': px_0,'py_0[m]': py_0,'pz_0[m]': pz_0,'ekin_0[eV]': ekin_0,'edep_0[eV]': edep_0,'vol_0': vol_0,'time_t[ms]':time_t,'x_t[m]':x_t,'y_t[m]': y_t,'z_t[m]': z_t,'r_t[m]': r_t,'L_t[m]': L_t,'px_t[m]': px_t,'py_t[m]': py_t,'pz_t[m]': pz_t,'ekin_t[eV]': ekin_t,'edep_t[eV]': edep_t,'ln(E0vsET)': lnE0vsET,'vol_t': vol_t,'nC_A': nCA,'nC_Z': nCZ, 'nsec': nsec, 'nsec_with_nC': nsec_with_nC, 'nsteps': nsteps})
    
        path_out = path_to_files.replace(f"{tier}", "tier2")
        print("path to",path_to_files,"path out",path_out)
        df.to_csv(f'{path_out}/{filename_base}-{m:04d}-tier2.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="neutron-sim-LF")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=300)
    parser.add_argument('--path_to_files', type=str, default=".")
    parser.add_argument('--tier', type=str, default="tier1")
    parser.add_argument('--nprimaries', type=int, default=50000)
    args = parser.parse_args()

    main(args.filename, args.path_to_files, args.tier, args.start, args.end, args.nprimaries)