import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read data
path = 'path_to_solution_files'

data_vc=pd.read_csv(path+'/v_c.csv',sep='\t')
data_ene=pd.read_csv(path+'/ene.csv',sep='\t')

fs=14

plt.rcParams["font.family"] = "Arial"

fig = plt.figure(figsize=(4,8))
ax0 = fig.add_subplot(311)
ax1 = fig.add_subplot(312)
ax2 = fig.add_subplot(313)

plt.setp(ax1.get_xticklabels(), fontsize=fs)
plt.setp(ax1.get_yticklabels(), fontsize=fs)
plt.setp(ax2.get_xticklabels(), fontsize=fs)
plt.setp(ax2.get_yticklabels(), fontsize=fs)

plt.setp(ax0.get_xticklabels(), fontsize=fs)
plt.setp(ax0.get_yticklabels(), fontsize=fs)

ax0.set_title(r'k/k$_0$=10$^5$', fontsize=fs)

fig.tight_layout()
fig.align_ylabels()

ax1.set_ylabel('Voltage',fontsize=fs)
ax2.set_ylabel(r'Current I/F/N$_A$ [s$^{-1}$]',fontsize=fs)
ax0.set_xlabel('Time [s]',fontsize=fs)
ax0.set_ylabel('Li concentration',fontsize=fs)

plt.xlim([0,1])
ax1.set_xlim([0,1])
ax1.tick_params(axis="y",direction="in",length=6)
ax1.tick_params(axis="x",direction="in",length=6)

ax1.set_xlim([0,1])
ax2.set_xlim([0,1])
ax0.set_ylim([0,1])
ax0.set_xlim([0,20])
ax1.set_xticklabels([])

fig.align_ylabels()

fig.tight_layout()

pos = ax2.get_position()
new_pos = [pos.x0, pos.y0+0.05, pos.width, pos.height]
ax2.set_position(new_pos)

pos = ax0.get_position()
new_pos = [pos.x0, pos.y0, pos.width, pos.height+0.2]
ax0.set_position(new_pos)

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

#Li concentration vs time
ax0.plot(data_ene['time'],data_ene['c'],label='1 mV',linestyle='solid', color='deepskyblue')

#Voltage vs Li concentration
ax1.plot(data_vc['conc'][1:],OCV(data_vc, delta_phi),label='1 mV',linestyle='solid', color='deepskyblue')

#Current vs Li concentration
ax2.plot(data_vc['conc'][1:],data_vc['current'][1:],label='1 mV',linestyle='solid', color='deepskyblue')

plt.savefig('test.png', bbox_inches='tight')
