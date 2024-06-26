# legend-multi-fidelity-surrogate-model

The in-situ production of long-lived radio-isotopes by cosmic muon interactions may generate a non-negligible background for rare event searches deep underground. The delayed decay of Ge-77(m) has been identified as the dominant in-situ cosmogenic contributor for a neutrinoless double-beta decay search with Ge-76. The future ton-scale LEGEND-1000 experiment requires a total background of < 10^{-5} cts/(keV kg yr). 

Neutron backgrounds have a strong dependence on laboratory depth, shielding material, and cryostat design. The addition of passive neutron moderators  results in a reduced background contribution. Therefore, Monte Carlo studies using a custom simulation module (https://github.com/MoritzNeuberger/warwick-legend/tree/master) based on Geant4 are performed to optimize the moderator screening effect. 

However, using traditional Monte Carlo simulations a full optimization of a many parameter space may still be a time consuming and difficult task to address. Machine learning can help in both speeding up common modeling problems, as well as help to minimize the application of computational expensive standard Monte Carlo methods. The Multi-Fidelity Gaussian Process based study presented here aims to demonstrate a technique on a small-scale application, which then is gradually adaptable to the more ambitious task of exploring innovative solutions to the design of detectors for future Ge-76 experiments.

![alt text](https://github.com/annkasch/legend-multi-fidelity-surrogate-model/blob/main/MF-GP_concept.png)

### Visualization

[Link to visualization tool](https://annkasch.github.io/legend-multi-fidelity-surrogate-model/)

<img src="https://github.com/annkasch/legend-multi-fidelity-surrogate-model/blob/main/utilities/vis.png" width="600">

