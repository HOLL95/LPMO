from Source.single_e_class_unified import single_electron
from Source.harmonics_plotter import harmonics
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import math
import pints
import time
experiment_type="SV"
current_dir=os.getcwd()

dec_amount=16
file_path=("/").join([current_dir, "experimental_data", "Extracted_data"])
def plot_harmonics(times, h_class,**kwargs):
    label_list=[]
    time_series_dict={}
    harm_dict={}
    if "hanning" not in kwargs:
        kwargs["hanning"]=False
    if "xaxis" not in kwargs:
        kwargs["xaxis"]=times
    label_counter=0
    for key in kwargs:
        if "time_series" in key:
            index=key.find("time_series")
            if key[index]=="_" or key[index]=="-":
                index-=1
            label_list.append(key[:index])
            time_series_dict[key[:index]]=kwargs[key]
            label_counter+=1
    if label_counter==0:
        return
    for label in label_list:
        harm_dict[label]=h_class.generate_harmonics(times, time_series_dict[label], hanning=kwargs["hanning"])
    num_harms=h_class.num_harmonics
    for i in range(0, num_harms):
        plt.subplot(num_harms, 1,i+1)
        for plot_name in label_list:
            plt.plot(kwargs["xaxis"], np.real(harm_dict[label][i,:]), label=plot_name)
        if i==0:
            plt.legend()
    plt.show()



def read_text_results(path, type, name):
    current_file_name=name+"_cv_current"
    voltage_file_name=name+"_cv_voltage"
    file_path=("/").join([path, type])
    current_data=np.loadtxt(file_path+"/"+current_file_name)
    voltage_data=np.loadtxt(file_path+"/"+voltage_file_name)
    current_results1=current_data[0::dec_amount, 1]
    time_results1=current_data[0::dec_amount,0]
    voltage_results1=voltage_data[0::dec_amount, 1]
    return time_results1, voltage_results1, current_results1
for i in range(1, 3):
    time_results1, voltage_results1, current_results1=read_text_results(file_path, "SV", "LPMO_SV_{}".format(str(i)))
    blank_time_results1, blank_voltage_results1, blank_current_results1=read_text_results(file_path, "BLANK", "Blank_sine")
    param_list={
        "E_0":0.2,
        'E_start':  min(voltage_results1[len(voltage_results1)//4:3*len(voltage_results1)//4]), #(starting dc voltage - V)
        'E_reverse':max(voltage_results1[len(voltage_results1)//4:3*len(voltage_results1)//4]),
        'omega':9.015036237186864, #8.88480830076,  #    (frequency Hz)
        "original_omega":9.015036237186864,
        'd_E': 302e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 1.0,  #     (uncompensated resistance ohms)
        'Cdl': 6e-3, #(capacitance parameters)
        'CdlE1': 0.05,#0.000653657774506,
        'CdlE2': 0,#0.000245772700637,
        "CdlE3":0,
        'gamma': 1e-8,
        "original_gamma":1e-8,        # (surface coverage per unit area)
        'k_0': 10000, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":0.2,
        "E0_std": 0.09,
        "E0_skew":0.2,
        "cap_phase":3*math.pi/2,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/200),
        'phase' :3*math.pi/2,
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
        "method": "sinusoidal",
        "phase_only":False,
        "likelihood":likelihood_options[1],
        "numerical_method": solver_list[1],
        "label": "MCMC",
        "optim_list":[]
    }
    for init_harm in range(3, 7):
        other_values={
            "filter_val": 0.5,
            "harmonic_range":list(range(init_harm,9,1)),
            "experiment_time": time_results1,
            "experiment_current": current_results1,
            "experiment_voltage":voltage_results1,
            "bounds_val":20000,
        }
        param_bounds={
            'E_0':[param_list["E_start"], param_list["E_reverse"]],
            'omega':[0.99*param_list['omega'],1.01*param_list['omega']],#8.88480830076,  #    (frequency Hz)
            'Ru': [0, 1e5],  #     (uncompensated resistance ohms)
            'Cdl': [0,1e-1], #(capacitance parameters)
            'CdlE1': [-0.5,0.5],#0.000653657774506,
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
        LPMO=single_electron(None, param_list, simulation_options, other_values, param_bounds)
        blank_other_values=copy.deepcopy(other_values)
        blank_results_dict=dict(zip(["experiment_current", "experiment_voltage", "experiment_time"], [blank_current_results1, blank_voltage_results1, blank_time_results1]))
        for key in blank_results_dict.keys():
            blank_other_values[key]=blank_results_dict[key]
        Blank_nano=single_electron(None, param_list, simulation_options, blank_other_values, param_bounds)

        LPMO.define_boundaries(param_bounds)
        time_results=LPMO.other_values["experiment_time"]
        current_results=LPMO.other_values["experiment_current"]
        voltage_results=LPMO.other_values["experiment_voltage"]
        blank_time_results=Blank_nano.other_values["experiment_time"]
        blank_current_results=Blank_nano.other_values["experiment_current"]
        blank_voltage_results=Blank_nano.other_values["experiment_voltage"]
        LPMO.def_optim_list([])
        start=time.time()
        syn_time=LPMO.test_vals([], "timeseries")
        print(time.time()-start)
        syn_volt=LPMO.define_voltages()
        h_class=harmonics(range(1, 9), param_list["omega"], 0.04)
        exp_harms=h_class.generate_harmonics(LPMO.t_nondim(time_results), LPMO.i_nondim(current_results))
        exp_Y=h_class.exposed_Y
        exp_f=h_class.exposed_f
        LPMO.simulation_options["label"]="cmaes"
        LPMO.simulation_options["likelihood"]="fourier"
        LPMO.simulation_options["test"]=False
        LPMO.simulation_options["adaptive_ru"]=False
        LPMO.simulation_options["dispersion_bins"]=[8]
        LPMO.simulation_options["GH_quadrature"]=True
        LPMO.simulation_options["numerical_method"]="Brent minimisation"
        LPMO.def_optim_list(["E_0","k_0","Ru","Cdl", "CdlE1", "CdlE2","gamma","omega","cap_phase","phase", "alpha"])
        cmaes_results=[0.1323344839793261, 8.439305794165604, 32.129571420325696, 0.0038661162192220517, -0.03397713786397259, 0.0026026961458140083, 4.403070990011245e-09, 9.015215428806911, 4.866870933553302, 5.331933302017616, 0.5999999979234201]
        #cmaes_results=[0.23449030178378097, 17.309449466321066, 106.96763508342268, 0.04212514631028971, -0.09999998841572955, 5.998251906128016e-06, 4.378848258266178e-08, 9.017197576723236, 6.283185304254533, 5.1973311117845435, 0.5999999934553761]
        cmaes_time=LPMO.test_vals(cmaes_results, likelihood="timeseries", test=False)

        plt.plot(voltage_results, cmaes_time, label="Simulation")
        plt.plot(voltage_results, current_results, alpha=0.5, label="Experiment")
        plt.plot(blank_voltage_results, blank_current_results, alpha=0.5, label="Blank")
        plt.legend()
        plt.show()
        syn_harms=h_class.generate_harmonics(LPMO.t_nondim(time_results), LPMO.i_nondim(cmaes_time))
        blank_harms=h_class.generate_harmonics(Blank_nano.t_nondim(blank_time_results), Blank_nano.i_nondim(blank_current_results))
        for i in range(0, len(syn_harms)):
            plt.subplot(len(syn_harms), 1, i+1)
            plt.plot(voltage_results, syn_harms[i,:]*1e6,label="Simulation")
            plt.plot(voltage_results, exp_harms[i,:]*1e6, alpha=0.7,label="Experiment")
            plt.plot(voltage_results, blank_harms[i,:]*1e6, alpha=0.7,label="Blank")
            if i==0:
                plt.legend()
        plt.show()
        
