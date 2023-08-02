import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update(plt.rcParamsDefault)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})


#plt.rcParams['figure.constrained_layout.use'] = True

struct_mass = 163.82
batt_mass = 74.8
powertrain_mass = 102.3
payload_mass = 50
optics = 52.9

data = [struct_mass,batt_mass,powertrain_mass,payload_mass,optics]
explode = [0.,0.,0.,0.,0.]

labels = ['structural mass', 'battery mass', 'powertrain mass', 'payload mass', 'power beaming\nsystems mass']

colors = ['slategrey','tan','mediumseagreen','coral','plum']


plt.pie(data, labels=labels, explode=explode, autopct='%1.1f%%', wedgeprops={ 'linewidth' : 2, 'edgecolor' : 'white' }, colors=colors)
centre_circle = plt.Circle((0,0),0.375,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
#plt.legend()



plt.savefig('mass.png',format='png',dpi=1000,transparent=True,bbox_inches="tight")

plt.show()