
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

nc_1 = []
nc_3 = []
nc_5 = []
nc_7 = []
nc_9 = []
tknc = []
tknp = []
kmnc = []
nbc = []
snac = []

for i in range(0, 10):
    with open("./coverage_results/mnist/lenet1/improve/coverage_result_" + str(i) + ".txt") as f:
        results = f.read()
        nc_1.append(float(results.split("\n")[3].split(" ")[1]))
        nc_3.append(float(results.split("\n")[4].split(" ")[1]))
        nc_5.append(float(results.split("\n")[5].split(" ")[1]))
        nc_7.append(float(results.split("\n")[6].split(" ")[1]))
        nc_9.append(float(results.split("\n")[7].split(" ")[1]))
        tknc.append(float(results.split("\n")[8].split(" ")[1]))
        tknp.append(float(results.split("\n")[9].split(" ")[1]))
        kmnc.append(float(results.split("\n")[10].split(" ")[1]))
        nbc.append(float(results.split("\n")[11].split(" ")[1]))
        snac.append(float(results.split("\n")[12].split(" ")[1]))

norm_nc_1 = [num/nc_1[0] for num in nc_1]
norm_nc_3 = [num/nc_3[0] for num in nc_3]
norm_nc_5 = [num/nc_5[0] for num in nc_5]
norm_nc_7 = [num/nc_7[0] for num in nc_7]
norm_nc_9 = [num/nc_9[0] for num in nc_9]
norm_tknc = [num/tknc[0] for num in tknc]
norm_tknp = [num/tknp[0] for num in tknp]
norm_kmnc = [num/kmnc[0] for num in kmnc]
norm_nbc = [num/nbc[0] for num in nbc]
norm_snac = [num/snac[0] for num in snac]
# print(norm_nc_1, norm_nc_3, norm_nc_5, norm_nc_7, norm_nc_9, norm_tknc, norm_tknp, norm_kmnc, norm_nbc, norm_snac)

# Define a list of markevery cases and color cases to plot
cases = ["NC(0.1)",
         "NC(0.1)",
         "NC(0.1)",
         "NC(0.1)",
         "NC(0.1)",
         "TKNC",
         "TKNP",
         "KMNC",
         "NBC",
         "SNAC"]

colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf']

# Set the plot curve with markers and a title
mpl.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(10, 7.5), tight_layout=True)

plt.rcParams['font.serif']=['Times']
plt.ylim((0.995, 1.090))
ax.set_xticks(range(0, 11))
ax.set_xtickslabels(["$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$"])
ax.plot(norm_nc_1, marker='.', label=str(cases[0]))
ax.plot(norm_nc_3, marker='*', label=str(cases[1]))
ax.plot(norm_nc_5, marker='o', label=str(cases[2]))
ax.plot(norm_nc_7, marker='+', label=str(cases[3]))
ax.plot(norm_nc_9, marker='v', label=str(cases[4]))
ax.plot(norm_tknc, marker='^', label=str(cases[5]))
ax.plot(norm_tknp, marker='<', label=str(cases[6]))
ax.plot(norm_kmnc, marker='>', label=str(cases[7]))
ax.plot(norm_nbc, marker='s', label=str(cases[8]))
ax.plot(norm_snac, marker='D', label=str(cases[9]))
ax.legend(bbox_to_anchor=(0.05, 0.95), loc='upper left', borderaxespad=0.)

plt.title('Support for axes.prop_cycle cycler with markevery')
plt.show()
plt.savefig("./fig.png")
