from Source.single_e_class_unified import single_electron
from Source.harmonics_plotter import harmonics
import numpy as np
import matplotlib.pyplot as plt
import os
experiment_type="SV"
current_dir=os.getcwd()
for i in range(1, 11):
    current_file_name="LPMO_{}_{}_cv_current".format(experiment_type, str(i))
    voltage_file_name="LPMO_{}_{}_cv_voltage".format(experiment_type, str(i))
    file_path=("/").join([current_dir, "experimental_data", "Extracted_data", experiment_type])
    current_data=np.loadtxt(file_path+"/"+current_file_name)
    voltage_data=np.loadtxt(file_path+"/"+voltage_file_name)
    current_results=current_data[:, 1]
    time_results=current_data[:,0]
    voltage_results=voltage_data[:, 1]
    plt.plot(voltage_results, current_results)
    plt.show()
