import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read data
path = 'path_to_solution_files'

data=pd.read_csv(path+'/v_c.csv',sep='\t')

fs=14

plt.rcParams["font.family"] = "Arial"

fig = plt.figure(figsize=(4,10))
ax1 = fig.add_subplot(311)

fig.tight_layout()
plt.setp(ax1.get_xticklabels(), fontsize=fs)
plt.setp(ax1.get_yticklabels(), fontsize=fs)

ax1.set_ylabel('Voltage',fontsize=fs)
ax1.set_xlabel('Li concentration',fontsize=fs)

ax1.set_title(r'k/k$_0$=1', fontsize=fs)

plt.xlim([0,1])
ax1.set_xlim([0,1])
ax1.tick_params(axis="y",direction="in",length=6)
ax1.tick_params(axis="x",direction="in",length=6)

Vm=4.38e-5
sigma=0.072
Wc=1e-9
H=sigma/Wc
F=96485.33

VHf=Vm*H/F #V

V_Li=-3.04 #V

def OCV(data,delta_phi):
    V_LFP=(data['mu_tot']*VHf)+delta_phi
    OCV = V_LFP-V_Li
    return OCV[1:]

delta_phi=1e-3 #V

ax1.plot(data['conc'][1:],OCV(data, delta_phi),label='1 mV',linestyle='solid', color='deepskyblue')

ax1.legend(loc='upper right',fontsize=14)
