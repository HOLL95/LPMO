from Source.single_e_class_unified import single_electron
from Source.harmonics_plotter import harmonics
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pints
import time
experiment_type="FTACV"
current_dir=os.getcwd()
file_path=("/").join([current_dir, "experimental_data", "Extracted_data", experiment_type])
dec_amount=16
"""current_file_name="LPMO_FTACV_after_SV_full_range_cv_current"
voltage_file_name="LPMO_FTACV_after_SV_full_range_cv_voltage"
file_path=("/").join([current_dir, "experimental_data", "Extracted_data", experiment_type])
current_data=np.loadtxt(file_path+"/"+current_file_name)
voltage_data=np.loadtxt(file_path+"/"+voltage_file_name)
current_results1=current_data[0::dec_amount, 1]
time_results1=current_data[0::dec_amount,0]
voltage_results1=voltage_data[0::dec_amount, 1]
deced_file=np.zeros((3, len(current_results1)))
deced_file[0, :]=time_results1
deced_file[1, :]=voltage_results1
deced_file[2, :]=current_results1"""
results=np.loadtxt(file_path+"DECED_16_BEFORE_SV_FTACV")
current_results1=results[2, 0::2]
voltage_results1=results[1, 0::2]
time_results1=results[0, 0::2]
param_list={
    "E_0":0.2,
    'E_start':  -300e-3, #(starting dc voltage - V)
    'E_reverse':700e-3,
    'omega':8.82, #8.88480830076,  #    (frequency Hz)
    'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 6e-3, #(capacitance parameters)
    'CdlE1': 0.05,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-9,
    "v":18.65e-3,
    "original_gamma":1e-9,        # (surface coverage per unit area)
    'k_0': 10000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "E0_skew":0.2,
    "cap_phase":3*math.pi/2,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :0,
    "time_end": -1,
    'num_peaks': 30,
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=2/(param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":16,
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[]
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(3,9,1)),
    "experiment_time": time_results1,
    "experiment_current": current_results1,
    "experiment_voltage":voltage_results1,
    "bounds_val":200,
}
param_bounds={
    'E_0':[param_list["E_start"], param_list["E_reverse"]],
    'omega':[0.99*param_list['omega'],1.01*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e5],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'CdlE1': [-0.1,0.1],#0.000653657774506,
    'CdlE2': [-0.05,0.05],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [0.01*param_list["original_gamma"],100*param_list["original_gamma"]],
    'k_0': [0.1, 1e4], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[0.1, 0.4],
    "E0_std": [1e-4,  0.2],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    'phase' : [math.pi, 2*math.pi],
    }
r_LPMO=single_electron(None, param_list, simulation_options, other_values, param_bounds)
time_results=r_LPMO.other_values["experiment_time"]
current_results=r_LPMO.other_values["experiment_current"]
voltage_results=r_LPMO.other_values["experiment_voltage"]

h_class=harmonics(range(2, 8), param_list["omega"], 0.04)
exp_harms=h_class.generate_harmonics(r_LPMO.t_nondim(time_results), r_LPMO.i_nondim(current_results), hanning=True)
SV_vals=[0.1323344839793261, 8.439305794165604, 32.129571420325696, 0.0038661162192220517, -0.03397713786397259, 0.0026026961458140083, 4.403070990011245e-09, 8.82, 0, 0.5999999979234201]
SV_vals=[0.23449030178378097, 17.309449466321066, 106.96763508342268, 0.004212514631028971, -0.09999998841572955, 5.998251906128016e-06, 4.378848258266178e-09, 8.82, 0, 0.5999999934553761]
r_LPMO.def_optim_list(["E_0","k_0","Ru","Cdl", "CdlE1", "CdlE2","gamma","omega","phase", "alpha"])
CMAES_time=r_LPMO.test_vals(SV_vals, "timeseries")
r_LPMO.simulation_options["method"]="dcv"
dcv_volt=r_LPMO.define_voltages()[r_LPMO.time_idx]
r_LPMO.simulation_options["method"]="ramped"
syn_harms=h_class.generate_harmonics(r_LPMO.t_nondim(time_results), r_LPMO.i_nondim(CMAES_time), hanning=True)
for i in range(0, len(exp_harms)):
    plt.subplot(len(exp_harms), 1, i+1)
    #plt.plot(time_results, abs(syn_harms[i,:]), label="Simulation")
    plt.plot(time_results, abs(exp_harms[i,:]), label="Experiment")
    if i==0:
        plt.legend()
plt.show()
