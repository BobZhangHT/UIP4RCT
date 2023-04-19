# UIP4RCT

The source codes for "Unit Information Prior for Incorporating the Real-World Evidence into Randomized Controlled Trials". See 

## Notebooks (`.ipynb`):

- `UIP4RCT_ExperimentSettings`: Generate the plot that describe distributions of propensity scores under different scenarios.  (Figure 1)
- `UIP4RCT_Simulation`: Perform the simulations of Scenarios 1-3 for continuous, binary and survivial outcomes.  (Tables 1-3)
- `UIP4RCT_Simulation_Extreme_Linear&Binary`: Perform the simulations of Scenarios 4&5 for linear and binary outcomes. (Tables 1&2)
- `RealDataApplication_HCQ`: Real data analysis for comparing UIP and NIP. (Table 4)

## Scripts (`.py`):

- `balance_methods`: Functions for different balancing methods (`IPW` and `Match`). 
- `datgen`: Functions to generate simulated dataset.
- `samplers`: MCMC sampler for UIP and models.  

## Dataset (`.csv`):
- `HCQ_RCT`: Patient-level data reconstructed from Figure 2 in the HCQ study (https://doi.org/10.1371/journal.pone.0257238).
