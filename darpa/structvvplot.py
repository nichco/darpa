import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})

# ucsd beam, Aurora beam, Femap, Aurora Shell
data = {'UCSD Beam':8592.9, 'Aurora Beam':7318., 'FEmap':8655., 'Aurora Shell':9405.}

models = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize = (7, 4))
 
# creating the bar plot
plt.bar(models, values, color =['red','blue','orange','green'], width = 0.75)

plt.axhline(y=8592.9, color='black', linestyle='dashed', linewidth=2, alpha=1, zorder=10, label = '_nolegend_') 

plt.ylabel('Stress (psi)')
plt.title('Structural Verification and Validation')


plt.savefig('structvv.png', dpi=500, transparent=True, bbox_inches="tight")
plt.show()