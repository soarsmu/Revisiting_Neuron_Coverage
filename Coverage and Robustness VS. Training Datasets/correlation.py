from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.pyplot as plt

def func(a, b):
    Lens = len(a)

    ties_onlyin_x = 0
    ties_onlyin_y = 0
    con_pair = 0
    dis_pair = 0
    for i in range(Lens - 1):
        for j in range(i + 1, Lens):
            test_tying_x = np.sign(a[i] - a[j])
            test_tying_y = np.sign(b[i] - b[j])
            panduan = test_tying_x * test_tying_y
            if panduan == 1:
                con_pair += 1
            elif panduan == -1:
                dis_pair += 1

            if test_tying_y == 0 and test_tying_x != 0:
                ties_onlyin_y += 1
            elif test_tying_x == 0 and test_tying_y != 0:
                ties_onlyin_x += 1

    if (con_pair + dis_pair + ties_onlyin_x) * (dis_pair + con_pair + ties_onlyin_y) == 0:
        k = 10**-1
    else:
        k = (con_pair + dis_pair + ties_onlyin_x) * (dis_pair + con_pair + ties_onlyin_y)

    Kendallta1 = (con_pair - dis_pair) / np.sqrt(k)

    return Kendallta1

import numpy as np


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

for i in range(0, 11):
    with open("./RQ2_results/coverage_results/mnist/lenet1/improve/coverage_result_" + str(i) + ".txt") as f:
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
mnist_data = [norm_nc_1, norm_nc_3, norm_nc_5, norm_nc_7, norm_nc_9, norm_tknc, norm_tknp, norm_kmnc, norm_nbc, norm_snac]



mr = []
acac = []
actc = []
alp_l0 = []
alp_l2 = []
alp_li = []
ass = []
psd = []
nte = []
rgb = []
ric = []

for i in range(0, 11):
    with open("./RQ2_results/robustness_results/mnist/lenet1/improve/robustness_metrics_" + str(i) + ".txt") as f:
        results = f.read()
        mr.append(float(results.split("\n")[3].split(" ")[1]))
        acac.append(float(results.split("\n")[4].split(" ")[1]))
        actc.append(float(results.split("\n")[5].split(" ")[1]))
        alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
        alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
        alp_li.append(float(results.split("\n")[8].split(" ")[1]))
        ass.append(float(results.split("\n")[9].split(" ")[1]))
        psd.append(float(results.split("\n")[10].split(" ")[1]))
        nte.append(float(results.split("\n")[11].split(" ")[1]))
        rgb.append(float(results.split("\n")[12].split(" ")[1]))
        ric.append(float(results.split("\n")[13].split(" ")[1]))

norm_mr = [num/mr[0] for num in mr]
norm_acac = [num/acac[0] for num in acac]
norm_actc = [num/actc[0] for num in actc]
norm_alp_l0 = [num/alp_l0[0] for num in alp_l0]
norm_alp_l2 = [num/alp_l2[0] for num in alp_l2]
norm_alp_li = [num/alp_li[0] for num in alp_li]
norm_ass = [num/ass[0] for num in ass]
norm_psd = [num/psd[0] for num in psd]
norm_nte = [num/nte[0] for num in nte]
norm_rgb = [num/rgb[0] for num in rgb]
norm_ric = [num/ric[0] for num in ric]
mnist_data_rb = [norm_mr, norm_acac, norm_actc, norm_alp_l0, norm_alp_l2, norm_alp_li, norm_ass, norm_psd, norm_nte, norm_rgb, norm_ric]

results = []
# calculate Kendallta
for metric_1 in mnist_data:
    tmp = []
    for metric_2 in mnist_data_rb:
        Kendallta, p_value = kendalltau(np.array(metric_1), np.array(metric_2))
        v = func(np.array(metric_1), np.array(metric_2))
        tmp.append(round(v, 2))
        print('The Kendallta between is {:.2f}'.format(v))
    results.append(tmp)

print(results)


# create hot picture in Seaborn
f, ax = plt.subplots(figsize=(10, 11))
ax.set_xticklabels(np.array(results), rotation='horizontal')

heatmap = sns.heatmap(np.array(results),
                        square=True,
                        cmap='coolwarm',
                        cbar_kws={'shrink': 0.7, 'ticks': [-1, -.5, 0, 0.5, 1]},
                        vmin=-1,
                        vmax=1,
                        annot=True,
                        annot_kws={'size': 16})

label_y = ax.get_yticklabels()
plt.setp(label_y, rotation = 45)

label_x = ax.get_xticklabels()
plt.setp(label_x, rotation = 45)


# cb = heatmap.figure.colorbar(heatmap.collections[0])
# cb.ax.tick_params(length = 0.001, width=2,  labelsize=10)

# sns.set_style({'yticklabels.top': True})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.savefig('ALL_ALL.eps', format='eps', bbox_inches='tight')

plt.show()





