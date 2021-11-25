from scipy.stats import ttest_ind, kendalltau, pearsonr, spearmanr, mannwhitneyu, wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import mean
from numpy import var, std
from math import sqrt

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


# nc_1 = []
# nc_3 = []
# nc_5 = []
# nc_7 = []
# nc_9 = []
# tknc = []
# tknp = []
# kmnc = []
# nbc = []
# snac = []

# for i in range(0, 11):
#     with open("./mnist/lenet1/improve/coverage_result_" + str(i) + ".txt") as f:
#         results = f.read()
#         nc_1.append(float(results.split("\n")[3].split(" ")[1]))
#         nc_3.append(float(results.split("\n")[4].split(" ")[1]))
#         nc_5.append(float(results.split("\n")[5].split(" ")[1]))
#         nc_7.append(float(results.split("\n")[6].split(" ")[1]))
#         nc_9.append(float(results.split("\n")[7].split(" ")[1]))
#         tknc.append(float(results.split("\n")[8].split(" ")[1]))
#         tknp.append(float(results.split("\n")[9].split(" ")[1]))
#         kmnc.append(float(results.split("\n")[10].split(" ")[1]))
#         nbc.append(float(results.split("\n")[11].split(" ")[1]))
#         snac.append(float(results.split("\n")[12].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./mnist/lenet1/no_improve/coverage_result_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         nc_1.append(float(results.split("\n")[3].split(" ")[1]))
# #         nc_3.append(float(results.split("\n")[4].split(" ")[1]))
# #         nc_5.append(float(results.split("\n")[5].split(" ")[1]))
# #         nc_7.append(float(results.split("\n")[6].split(" ")[1]))
# #         nc_9.append(float(results.split("\n")[7].split(" ")[1]))
# #         tknc.append(float(results.split("\n")[8].split(" ")[1]))
# #         tknp.append(float(results.split("\n")[9].split(" ")[1]))
# #         kmnc.append(float(results.split("\n")[10].split(" ")[1]))
# #         nbc.append(float(results.split("\n")[11].split(" ")[1]))
# #         snac.append(float(results.split("\n")[12].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./mnist/lenet4/improve/coverage_result_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         nc_1.append(float(results.split("\n")[3].split(" ")[1]))
# #         nc_3.append(float(results.split("\n")[4].split(" ")[1]))
# #         nc_5.append(float(results.split("\n")[5].split(" ")[1]))
# #         nc_7.append(float(results.split("\n")[6].split(" ")[1]))
# #         nc_9.append(float(results.split("\n")[7].split(" ")[1]))
# #         tknc.append(float(results.split("\n")[8].split(" ")[1]))
# #         tknp.append(float(results.split("\n")[9].split(" ")[1]))
# #         kmnc.append(float(results.split("\n")[10].split(" ")[1]))
# #         nbc.append(float(results.split("\n")[11].split(" ")[1]))
# #         snac.append(float(results.split("\n")[12].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./mnist/lenet5/improve/coverage_result_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         nc_1.append(float(results.split("\n")[3].split(" ")[1]))
# #         nc_3.append(float(results.split("\n")[4].split(" ")[1]))
# #         nc_5.append(float(results.split("\n")[5].split(" ")[1]))
# #         nc_7.append(float(results.split("\n")[6].split(" ")[1]))
# #         nc_9.append(float(results.split("\n")[7].split(" ")[1]))
# #         tknc.append(float(results.split("\n")[8].split(" ")[1]))
# #         tknp.append(float(results.split("\n")[9].split(" ")[1]))
# #         kmnc.append(float(results.split("\n")[10].split(" ")[1]))
# #         nbc.append(float(results.split("\n")[11].split(" ")[1]))
# #         snac.append(float(results.split("\n")[12].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./svhn/svhn_first/improve/coverage_result_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         nc_1.append(float(results.split("\n")[3].split(" ")[1]))
# #         nc_3.append(float(results.split("\n")[4].split(" ")[1]))
# #         nc_5.append(float(results.split("\n")[5].split(" ")[1]))
# #         nc_7.append(float(results.split("\n")[6].split(" ")[1]))
# #         nc_9.append(float(results.split("\n")[7].split(" ")[1]))
# #         tknc.append(float(results.split("\n")[8].split(" ")[1]))
# #         tknp.append(float(results.split("\n")[9].split(" ")[1]))
# #         kmnc.append(float(results.split("\n")[10].split(" ")[1]))
# #         nbc.append(float(results.split("\n")[11].split(" ")[1]))
# #         snac.append(float(results.split("\n")[12].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./svhn/svhn_model/improve/coverage_result_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         nc_1.append(float(results.split("\n")[3].split(" ")[1]))
# #         nc_3.append(float(results.split("\n")[4].split(" ")[1]))
# #         nc_5.append(float(results.split("\n")[5].split(" ")[1]))
# #         nc_7.append(float(results.split("\n")[6].split(" ")[1]))
# #         nc_9.append(float(results.split("\n")[7].split(" ")[1]))
# #         tknc.append(float(results.split("\n")[8].split(" ")[1]))
# #         tknp.append(float(results.split("\n")[9].split(" ")[1]))
# #         kmnc.append(float(results.split("\n")[10].split(" ")[1]))
# #         nbc.append(float(results.split("\n")[11].split(" ")[1]))
# #         snac.append(float(results.split("\n")[12].split(" ")[1]))


# # for i in [0, 10]:
# #     with open("./svhn/svhn_model/no_improve/coverage_result_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         nc_1.append(float(results.split("\n")[3].split(" ")[1]))
# #         nc_3.append(float(results.split("\n")[4].split(" ")[1]))
# #         nc_5.append(float(results.split("\n")[5].split(" ")[1]))
# #         nc_7.append(float(results.split("\n")[6].split(" ")[1]))
# #         nc_9.append(float(results.split("\n")[7].split(" ")[1]))
# #         tknc.append(float(results.split("\n")[8].split(" ")[1]))
# #         tknp.append(float(results.split("\n")[9].split(" ")[1]))
# #         kmnc.append(float(results.split("\n")[10].split(" ")[1]))
# #         nbc.append(float(results.split("\n")[11].split(" ")[1]))
# #         snac.append(float(results.split("\n")[12].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./svhn/svhn_second/no_improve/coverage_result_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         nc_1.append(float(results.split("\n")[3].split(" ")[1]))
# #         nc_3.append(float(results.split("\n")[4].split(" ")[1]))
# #         nc_5.append(float(results.split("\n")[5].split(" ")[1]))
# #         nc_7.append(float(results.split("\n")[6].split(" ")[1]))
# #         nc_9.append(float(results.split("\n")[7].split(" ")[1]))
# #         tknc.append(float(results.split("\n")[8].split(" ")[1]))
# #         tknp.append(float(results.split("\n")[9].split(" ")[1]))
# #         kmnc.append(float(results.split("\n")[10].split(" ")[1]))
# #         nbc.append(float(results.split("\n")[11].split(" ")[1]))
# #         snac.append(float(results.split("\n")[12].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./cifar/resnet20/no_improve/coverage_result_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         nc_1.append(float(results.split("\n")[3].split(" ")[1]))
# #         nc_3.append(float(results.split("\n")[4].split(" ")[1]))
# #         nc_5.append(float(results.split("\n")[5].split(" ")[1]))
# #         nc_7.append(float(results.split("\n")[6].split(" ")[1]))
# #         nc_9.append(float(results.split("\n")[7].split(" ")[1]))
# #         tknc.append(float(results.split("\n")[8].split(" ")[1]))
# #         tknp.append(float(results.split("\n")[9].split(" ")[1]))
# #         kmnc.append(float(results.split("\n")[10].split(" ")[1]))
# #         nbc.append(float(results.split("\n")[11].split(" ")[1]))
# #         snac.append(float(results.split("\n")[12].split(" ")[1]))

# # for i in [0, 8]:
# #     with open("./cifar/resnet20/improve/coverage_result_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         nc_1.append(float(results.split("\n")[3].split(" ")[1]))
# #         nc_3.append(float(results.split("\n")[4].split(" ")[1]))
# #         nc_5.append(float(results.split("\n")[5].split(" ")[1]))
# #         nc_7.append(float(results.split("\n")[6].split(" ")[1]))
# #         nc_9.append(float(results.split("\n")[7].split(" ")[1]))
# #         tknc.append(float(results.split("\n")[8].split(" ")[1]))
# #         tknp.append(float(results.split("\n")[9].split(" ")[1]))
# #         kmnc.append(float(results.split("\n")[10].split(" ")[1]))
# #         nbc.append(float(results.split("\n")[11].split(" ")[1]))
# #         snac.append(float(results.split("\n")[12].split(" ")[1]))

# # norm_nc_1 = [num/nc_1[0] for num in nc_1]
# # norm_nc_3 = [num/nc_3[0] for num in nc_3]
# # norm_nc_5 = [num/nc_5[0] for num in nc_5]
# # norm_nc_7 = [num/nc_7[0] for num in nc_7]
# # norm_nc_9 = [num/nc_9[0] for num in nc_9]
# # norm_tknc = [num/tknc[0] for num in tknc]
# # norm_tknp = [num/tknp[0] for num in tknp]
# # norm_kmnc = [num/kmnc[0] for num in kmnc]
# # norm_nbc = [num/nbc[0] for num in nbc]
# # norm_snac = [num/snac[0] for num in snac]
# # mnist_data_1 = [norm_nc_1, norm_nc_3, norm_nc_5, norm_nc_7, norm_nc_9, norm_tknc, norm_tknp, norm_kmnc, norm_nbc, norm_snac]

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
mnist_data_1 = [norm_nc_1, norm_nc_3, norm_nc_5, norm_nc_7, norm_nc_9, norm_tknc, norm_tknp, norm_kmnc, norm_nbc, norm_snac]



# mr = []
# acac = []
# actc = []
# alp_l0 = []
# alp_l2 = []
# alp_li = []
# ass = []
# psd = []
# nte = []
# rgb = []
# ric = []

# for i in range(0, 11):
#     with open("./RQ2_results/robustness_results/mnist/lenet1/improve/robustness_metrics_" + str(i) + ".txt") as f:
#         results = f.read()
#         mr.append(float(results.split("\n")[3].split(" ")[1]))
#         acac.append(float(results.split("\n")[4].split(" ")[1]))
#         actc.append(float(results.split("\n")[5].split(" ")[1]))
#         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
#         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
#         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
#         ass.append(float(results.split("\n")[9].split(" ")[1]))
#         psd.append(float(results.split("\n")[10].split(" ")[1]))
#         nte.append(float(results.split("\n")[11].split(" ")[1]))
#         rgb.append(float(results.split("\n")[12].split(" ")[1]))
#         ric.append(float(results.split("\n")[13].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./robustness_results/mnist/lenet1/no_improve/robustness_metrics_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         mr.append(float(results.split("\n")[3].split(" ")[1]))
# #         acac.append(float(results.split("\n")[4].split(" ")[1]))
# #         actc.append(float(results.split("\n")[5].split(" ")[1]))
# #         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
# #         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
# #         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
# #         ass.append(float(results.split("\n")[9].split(" ")[1]))
# #         psd.append(float(results.split("\n")[10].split(" ")[1]))
# #         nte.append(float(results.split("\n")[11].split(" ")[1]))
# #         rgb.append(float(results.split("\n")[12].split(" ")[1]))
# #         ric.append(float(results.split("\n")[13].split(" ")[1]))

# # for i in range(0, 11):
# #     with open("./robustness_results/rb/Untitled/mnist/lenet4/improve/robustness_metrics_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         mr.append(float(results.split("\n")[3].split(" ")[1]))
# #         acac.append(float(results.split("\n")[4].split(" ")[1]))
# #         actc.append(float(results.split("\n")[5].split(" ")[1]))
# #         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
# #         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
# #         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
# #         ass.append(float(results.split("\n")[9].split(" ")[1]))
# #         psd.append(float(results.split("\n")[10].split(" ")[1]))
# #         nte.append(float(results.split("\n")[11].split(" ")[1]))
# #         rgb.append(float(results.split("\n")[12].split(" ")[1]))
# #         ric.append(float(results.split("\n")[13].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./robustness_results/rb/Untitled/mnist/lenet5/improve/robustness_metrics_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         mr.append(float(results.split("\n")[3].split(" ")[1]))
# #         acac.append(float(results.split("\n")[4].split(" ")[1]))
# #         actc.append(float(results.split("\n")[5].split(" ")[1]))
# #         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
# #         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
# #         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
# #         ass.append(float(results.split("\n")[9].split(" ")[1]))
# #         psd.append(float(results.split("\n")[10].split(" ")[1]))
# #         nte.append(float(results.split("\n")[11].split(" ")[1]))
# #         rgb.append(float(results.split("\n")[12].split(" ")[1]))
# #         ric.append(float(results.split("\n")[13].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./robustness_results/rb/Untitled/cifar/resnet20/no_improve/robustness_metrics_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         mr.append(float(results.split("\n")[3].split(" ")[1]))
# #         acac.append(float(results.split("\n")[4].split(" ")[1]))
# #         actc.append(float(results.split("\n")[5].split(" ")[1]))
# #         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
# #         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
# #         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
# #         ass.append(float(results.split("\n")[9].split(" ")[1]))
# #         psd.append(float(results.split("\n")[10].split(" ")[1]))
# #         nte.append(float(results.split("\n")[11].split(" ")[1]))
# #         rgb.append(float(results.split("\n")[12].split(" ")[1]))
# #         ric.append(float(results.split("\n")[13].split(" ")[1]))

# for i in range(0, 11):
#     with open("./robustness_results/rb/Untitled/cifar/resnet20/improve/robustness_metrics_" + str(i) + ".txt") as f:
#         results = f.read()
#         mr.append(float(results.split("\n")[3].split(" ")[1]))
#         acac.append(float(results.split("\n")[4].split(" ")[1]))
#         actc.append(float(results.split("\n")[5].split(" ")[1]))
#         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
#         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
#         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
#         ass.append(float(results.split("\n")[9].split(" ")[1]))
#         psd.append(float(results.split("\n")[10].split(" ")[1]))
#         nte.append(float(results.split("\n")[11].split(" ")[1]))
#         rgb.append(float(results.split("\n")[12].split(" ")[1]))
#         ric.append(float(results.split("\n")[13].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./robustness_results/rb/Untitled/svhn/svhn_first/improve/robustness_metrics_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         mr.append(float(results.split("\n")[3].split(" ")[1]))
# #         acac.append(float(results.split("\n")[4].split(" ")[1]))
# #         actc.append(float(results.split("\n")[5].split(" ")[1]))
# #         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
# #         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
# #         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
# #         ass.append(float(results.split("\n")[9].split(" ")[1]))
# #         psd.append(float(results.split("\n")[10].split(" ")[1]))
# #         nte.append(float(results.split("\n")[11].split(" ")[1]))
# #         rgb.append(float(results.split("\n")[12].split(" ")[1]))
# #         ric.append(float(results.split("\n")[13].split(" ")[1]))

# for i in range(0, 11):
#     with open("./robustness_results/rb/Untitled/svhn/svhn_model/improve/robustness_metrics_" + str(i) + ".txt") as f:
#         results = f.read()
#         mr.append(float(results.split("\n")[3].split(" ")[1]))
#         acac.append(float(results.split("\n")[4].split(" ")[1]))
#         actc.append(float(results.split("\n")[5].split(" ")[1]))
#         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
#         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
#         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
#         ass.append(float(results.split("\n")[9].split(" ")[1]))
#         psd.append(float(results.split("\n")[10].split(" ")[1]))
#         nte.append(float(results.split("\n")[11].split(" ")[1]))
#         rgb.append(float(results.split("\n")[12].split(" ")[1]))
#         ric.append(float(results.split("\n")[13].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./robustness_results/rb/Untitled/svhn/svhn_model/no_improve/robustness_metrics_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         mr.append(float(results.split("\n")[3].split(" ")[1]))
# #         acac.append(float(results.split("\n")[4].split(" ")[1]))
# #         actc.append(float(results.split("\n")[5].split(" ")[1]))
# #         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
# #         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
# #         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
# #         ass.append(float(results.split("\n")[9].split(" ")[1]))
# #         psd.append(float(results.split("\n")[10].split(" ")[1]))
# #         nte.append(float(results.split("\n")[11].split(" ")[1]))
# #         rgb.append(float(results.split("\n")[12].split(" ")[1]))
# #         ric.append(float(results.split("\n")[13].split(" ")[1]))

# # for i in [0, 10]:
# #     with open("./robustness_results/rb/Untitled/svhn/svhn_second/no_improve/robustness_metrics_" + str(i) + ".txt") as f:
# #         results = f.read()
# #         mr.append(float(results.split("\n")[3].split(" ")[1]))
# #         acac.append(float(results.split("\n")[4].split(" ")[1]))
# #         actc.append(float(results.split("\n")[5].split(" ")[1]))
# #         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
# #         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
# #         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
# #         ass.append(float(results.split("\n")[9].split(" ")[1]))
# #         psd.append(float(results.split("\n")[10].split(" ")[1]))
# #         nte.append(float(results.split("\n")[11].split(" ")[1]))
# #         rgb.append(float(results.split("\n")[12].split(" ")[1]))
# #         ric.append(float(results.split("\n")[13].split(" ")[1]))


# norm_mr = [num/mr[5] for num in mr]
# norm_acac = [num/acac[5] for num in acac]
# norm_actc = [num/actc[5] for num in actc]
# norm_alp_l0 = [num/alp_l0[5] for num in alp_l0]
# norm_alp_l2 = [num/alp_l2[5] for num in alp_l2]
# norm_alp_li = [num/alp_li[5] for num in alp_li]
# norm_ass = [num/ass[5] for num in ass]
# norm_psd = [num/psd[5] for num in psd]
# norm_nte = [num/nte[5] for num in nte]
# norm_rgb = [num/rgb[5] for num in rgb]
# norm_ric = [num/ric[5] for num in ric]
# mnist_data_rb = [norm_mr, norm_acac, norm_actc, norm_alp_l0, norm_alp_l2, norm_alp_li, norm_ass, norm_psd, norm_nte, norm_rgb, norm_ric]



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
    with open("./robustness_results/mnist/lenet1/improve/robustness_metrics_" + str(i) + ".txt") as f:
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

# for i in range(0, 11):
#     with open("./robustness_results/rb/Untitled/svhn/svhn_model/no_improve/robustness_metrics_" + str(i) + ".txt") as f:
#         results = f.read()
#         mr.append(float(results.split("\n")[3].split(" ")[1]))
#         acac.append(float(results.split("\n")[4].split(" ")[1]))
#         actc.append(float(results.split("\n")[5].split(" ")[1]))
#         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
#         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
#         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
#         ass.append(float(results.split("\n")[9].split(" ")[1]))
#         psd.append(float(results.split("\n")[10].split(" ")[1]))
#         nte.append(float(results.split("\n")[11].split(" ")[1]))
#         rgb.append(float(results.split("\n")[12].split(" ")[1]))
#         ric.append(float(results.split("\n")[13].split(" ")[1]))

# for i in range(0, 11):
#     with open("./robustness_results/rb/Untitled/cifar/resnet20/no_improve/robustness_metrics_" + str(i) + ".txt") as f:
#         results = f.read()
#         mr.append(float(results.split("\n")[3].split(" ")[1]))
#         acac.append(float(results.split("\n")[4].split(" ")[1]))
#         actc.append(float(results.split("\n")[5].split(" ")[1]))
#         alp_l0.append(float(results.split("\n")[6].split(" ")[1]))
#         alp_l2.append(float(results.split("\n")[7].split(" ")[1]))
#         alp_li.append(float(results.split("\n")[8].split(" ")[1]))
#         ass.append(float(results.split("\n")[9].split(" ")[1]))
#         psd.append(float(results.split("\n")[10].split(" ")[1]))
#         nte.append(float(results.split("\n")[11].split(" ")[1]))
#         rgb.append(float(results.split("\n")[12].split(" ")[1]))
#         ric.append(float(results.split("\n")[13].split(" ")[1]))

norm_mr = [num/mr[5] for num in mr]
norm_acac = [num/acac[5] for num in acac]
norm_actc = [num/actc[5] for num in actc]
norm_alp_l0 = [num/alp_l0[5] for num in alp_l0]
norm_alp_l2 = [num/alp_l2[5] for num in alp_l2]
norm_alp_li = [num/alp_li[5] for num in alp_li]
norm_ass = [num/ass[5] for num in ass]
norm_psd = [num/psd[5] for num in psd]
norm_nte = [num/nte[5] for num in nte]
norm_rgb = [num/rgb[5] for num in rgb]
norm_ric = [num/ric[5] for num in ric]
mnist_data_rb_1 = [norm_mr, norm_acac, norm_actc, norm_alp_l0, norm_alp_l2, norm_alp_li, norm_ass, norm_psd, norm_nte, norm_rgb, norm_ric]

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

plt.rcParams['font.serif'] = 'Times'
mpl.rcParams['text.usetex'] = True
# Set the plot curve with markers and a title
fig, ax = plt.subplots(figsize=(11, 8), tight_layout=True)


ax.plot(norm_nc_1, marker='.', label=str(cases[0]), linewidth=3)
ax.plot(norm_nc_3, marker='*', label=str(cases[1]), linewidth=3)
ax.plot(norm_nc_5, marker='o', label=str(cases[2]), linewidth=3)
ax.plot(norm_nc_7, marker='+', label=str(cases[3]), linewidth=3)
ax.plot(norm_nc_9, marker='v', label=str(cases[4]), linewidth=3)
ax.plot(norm_tknc, marker='^', label=str(cases[5]), linewidth=3)
ax.plot(norm_tknp, marker='<', label=str(cases[6]), linewidth=3)
ax.plot(norm_kmnc, marker='>', label=str(cases[7]), linewidth=3)
ax.plot(norm_nbc, marker='s', label=str(cases[8]), linewidth=3)
ax.plot(norm_snac, marker='D', label=str(cases[9]), linewidth=3)
plt.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left', borderaxespad=0., fontsize=21, ncol=2)

plt.xticks(np.arange(0, 11), ["$D_0$", "$D_1$", "$D_2$", "$D_3$", "$D_4$", "$D_5$", "$D_6$", "$D_7$", "$D_8$", "$D_9$", "$D_{10}$"], fontsize=25)
plt.yticks(np.arange(0.985, 1.120, 0.015), fontsize=25)
# plt.xlabels(["$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$"])
plt.ylabel('Relative Coverage', fontsize=31)
plt.savefig("./fig1.pdf")
plt.show()


# Define a list of markevery cases and color cases to plot
cases = ["MR", "ACAC", 
"ACTC",
"ALP\_L0",
"ALP\_L2", 
"ALP\_Li",
"ASS",
"PSD",
"NTE",
"RGB",
"RIC"]

colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#17bce2']
# import matplotlib
mpl.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(11, 8), tight_layout=True)

ax.plot(norm_mr, marker='.', label=str(cases[0]), linewidth=3)
ax.plot(norm_acac, marker='*', label=str(cases[1]), linewidth=3)
ax.plot(norm_actc, marker='o', label=str(cases[2]), linewidth=3)
ax.plot(norm_alp_l0, marker='+', label=str(cases[3]), linewidth=3)
ax.plot(norm_alp_l2, marker='v', label=str(cases[4]), linewidth=3)
ax.plot(norm_alp_li, marker='^', label=str(cases[5]), linewidth=3)
ax.plot(norm_ass, marker='<', label=str(cases[6]), linewidth=3)
ax.plot(norm_psd, marker='>', label=str(cases[7]), linewidth=3)
ax.plot(norm_nte, marker='s', label=str(cases[8]), linewidth=3)
ax.plot(norm_rgb, marker='D', label=str(cases[9]), linewidth=3)
ax.plot(norm_ric, marker='D', label=str(cases[10]), linewidth=3)
plt.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right', borderaxespad=0., fontsize=21, ncol=2)

plt.xticks(np.arange(0, 11), ["$D_0$", "$D_1$", "$D_2$", "$D_3$", "$D_4$", "$D_5$", "$D_6$", "$D_7$", "$D_8$", "$D_9$", "$D_{10}$"], fontsize=25)
plt.yticks(np.arange(0.875, 1.385, 0.05), fontsize=25)
# plt.xlabels(["$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$", "$-1$"])
plt.ylabel('Model Quality', fontsize=31)
plt.savefig("./fig2.pdf")
plt.show()
exit()

# # print(len(mnist_data_1[0]), len(mnist_data_rb[0]))
# exit()
# results = []
# # calculate Kendallta
# # for metric_1 in mnist_data_1:
# #     tmp = []
# #     print("\n")
# #     for metric_2 in mnist_data_rb:
# #         Kendallta, p_value = kendalltau(np.array(metric_1), np.array(metric_2))
# #         # p_value = ttest_ind(np.array(metric_1), np.array(metric_2)).pvalue
# #         # print(round(p_value,4), end="  ")
# #         # print(metric_1, metric_2)
# #         v = func(np.array(metric_1), np.array(metric_2))
# #         tmp.append(round(v, 2))
# #     results.append(tmp)

def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # print(n2, n1)
    # calculate the variance of the samples
    s1, s2 = std(d1), std(d2)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 **2 + (n2 - 1) * s2 **2) / (n1 + n2 - 2))
    
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # print(u2, u1)
    # calculate the effect size
    return (u2 - u1) / s

results = []
for metric_1, metric_2 in zip(mnist_data_rb, mnist_data_rb_1):
    p_value = mannwhitneyu(metric_1, metric_2).pvalue
    cd = cohend(metric_1, metric_2)
    results.append(p_value)
print(results)

# exit()
# # create hot picture in Seaborn
# f, ax = plt.subplots(figsize=(12, 12.5))
# mpl.rcParams['text.usetex'] = True
# plt.rcParams['font.serif'] = 'Times'

# label_y = ["MR", "ACAC", 
# "ACTC",
# "ALP-L0",
# "ALP-L2", 
# "ALP-Li",
# "ASS",
# "PSD",
# "NTE",
# "RGB",
# "RIC"]

# # label_y = ["NC(0.1)",
# #          "NC(0.3)",
# #          "NC(0.5)",
# #          "NC(0.7)",
# #          "NC(0.9)",
# #          "TKNC",
# #          "TKNP",
# #          "KMNC",
# #          "NBC",
# #          "SNAC"]

# label_x = ["NC(0.1)",
#          "NC(0.3)",
#          "NC(0.5)",
#          "NC(0.7)",
#          "NC(0.9)",
#          "TKNC",
#          "TKNP",
#          "KMNC",
#          "NBC",
#          "SNAC"]

# # mask = np.zeros_like(np.array(results), dtype=np.bool)
# # mask[np.triu_indices_from(mask)] = True
# # print(type(mask))

# heatmap = sns.heatmap(np.array(results),
#                         square=True,
#                         # mask = mask,
#                         cmap='coolwarm',
#                         cbar_kws={'shrink': 0.7, 'ticks': [-1, -.5, 0, 0.5, 1]},
#                         vmin=-1,
#                         vmax=1,
#                         annot=True,
#                         annot_kws={'size': 20},
#                         xticklabels = label_y,
#                         yticklabels = label_x)

# # ax.set_xticks(np.arange(len(label_y)), labels=label_y)
# # ax.set_yticks(np.arange(len(label_x)), labels=label_x)
# cax = plt.gcf().axes[-1]
# cax.tick_params(labelsize=18)
# # # plt.setp(label_y, rotation = 45)
# # # plt.setp(label_x, rotation = 45)

# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
# plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
# # cb = heatmap.figure.colorbar(heatmap.collections[0])
# # cb.ax.tick_params(length = 0.001, width=2,  labelsize=10)

# # sns.set_style({'yticklabels.top': True})
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.savefig('./fig6.pdf', bbox_inches='tight')

# plt.show()

# \begin{table*}[!t]
#     \caption{Misclassifcation rates for different test generation methods.}
#     \begin{center}
#     \begin{tabular}{|c|c|cccccc|}
#     \hline
#     \multirow{2}{*}{Datasets} & \multirow{2}{*}{Models}       & \multicolumn{6}{c|}{Misclassification Rate }  \\ \cline{3-8} 
#                               &                               & \multicolumn{1}{c|}{Benign} & \multicolumn{1}{c|}{Diff} & \multicolumn{1}{c|}{Non-Diff}   & \multicolumn{1}{c|}{DH} & \multicolumn{1}{c|}{FGSM} & \multicolumn{1}{c|}{PGD}  \\ \hline
#     \hline\multirow{3}{*}{MNIST}            & LeNet-1         & 1.33\% & 49.26\% & 48.45\% & 48.65\% & 90.59\% & 100.00\% \\
#                                             & LeNet-4         & 1.38\% & 45.51\% & 43.11\% & 44.50\% & 80.78\% & 99.98\% \\
#                                             & LeNet-5         & 1.02\% & 42.97\% & 42.10\% & 43.05\% & 64.52\% & 99.70\% \\
#     \hline\multirow{3}{*}{CIFAR}            & VGG-16          & 7.20\% & 79.05\% & 80.78\% & 79.77\% & 68.24\% & 98.90\% \\
#                                             & ResNet-20       & 8.26\% & 68.13\% & 65.22\% & 66.37\% & 82.68\% & 100.00\% \\
#                                             & ResNet-56       & 7.24\% & 62.64\% & 65.34\% & 64.08\% & 70.71\% & 100.00\% \\
#     \hline\multirow{3}{*}{SVHN}             & SADL-1          & 10.30\% & 23.97\% & 30.61\% & 27.27\% & 83.96\% & 99.99\% \\
#                                             & SADL-2          & 12.28\% & 24.17\% & 22.04\% & 22.89\% & 91.73\% & 99.94\% \\
#                                             & SADL-3          & 7.43\% & 20.90\% & 16.71\% & 18.45\% & 88.80\% & 100.00\% \\
#     \hline \multirow{3}{*}{EuroSAT}         & VGG-16          & 3.50\% & 50.09\% & 40.93\% & 45.15\% & 88.76\% & 97.72\% \\
#                                             & ResNet-20       & 2.89\% & 48.69\% & 54.06\% & 51.61\% & 88.74\% & 99.41\% \\
#                                             & ResNet-32       & 2.78\% & 46.94\% & 48.87\% & 48.11\% & 88.22\% & 99.17\% \\
#                                             & ResNet-56       & 3.65\% & 44.72\% & 34.89\% & 39.26\% & 85.50\% & 99.72\% \\
#     \hline
#     \end{tabular}
#     \label{tab:mr_on_attack}
# \end{center}
# \end{table*}

# fgsm = [90.59, 80.78, 64.52, 68.24, 82.68, 70.71, 83.96, 91.73, 80.80, 88.76, 88.74, 88.22, 85.50]
# pgd = [100.00, 99.98, 99.70, 98.90, 100.00, 100.00, 99.99, 99.94, 100.00, 97.72, 99.41, 99.17, 99.72]
# diff = [49.26, 45.51, 42.97, 79.05, 68.13, 62.64, 23.97, 24.17, 20.90, 50.09, 48.69, 46.94, 44.72]
# non_diff = [48.45, 43.11, 42.10, 80.78, 65.22, 65.34, 30.61, 22.04, 16.71, 40.93, 54.06, 48.87, 34.89]
# dh = [48.65, 44.50, 43.05, 79.77, 66.37, 64.08, 27.27, 22.89, 18.45, 45.15, 51.61, 48.11, 39.26]

# # for metric_1, metric_2 in zip(diff, non_diff):
# #     # for i in metric_1:
#     #     metric_2.append(i)
#     # print(metric_2)
# p_value_1 = mannwhitneyu(fgsm+fgsm+fgsm+pgd+pgd+pgd, diff+non_diff+dh+diff+non_diff+dh).pvalue
# # p_value_2 = mannwhitneyu(, diff+non_diff+dh).pvalue
# # p_value_1 = mannwhitneyu(dh, non_diff).pvalue
# # p_value_2 = ttest_ind(pgd+pgd+pgd, diff+non_diff+dh).pvalue
# # cd = cohend(metric_1, metric_2)

# print(p_value_1)

# \begin{table*}[]
# %     \caption{Accuracy of defended models.}
# %     \begin{center}
# %     \begin{tabular}{|c|c|cccccc|}
# %     \hline
# %     \multirow{2}{*}{Datasets} & \multirow{2}{*}{Models}       & \multicolumn{6}{c|}{Accuracy }                                                                                               \\ \cline{3-8} 
# %                               &                               &  \multicolumn{1}{c|}{Benign}  & \multicolumn{1}{c|}{D}  & \multicolumn{1}{c|}{N} & \multicolumn{1}{c|}{DH} & \multicolumn{1}{c|}{FGSM} & \multicolumn{1}{c|}{PGD}  \\ \hline
# %     \hline\multirow{6}{*}{MNIST}    & LN5            &  98.98\%(+0.00\%) & 57.03\%(+0.00\%) & 57.90\%(+0.00\%) & 56.95\%(+0.00\%) & 35.48\%(+0.00\%) & 0.30\%(0.00\%) \\
# %                                     & LN5-D       &  98.69\%(-0.29\%) & 55.23\%(-1.80\%) & 55.16\%(-2.74\%) & 55.12\%(-1.83\%) & 42.24\%(+6.76\%) & 0.89\%(+0.59\%) \\
# %                                     & LN5-N    &  98.60\%(-0.38\%) & 54.71\%(-2.32\%) & 50.30\%(-7.60\%) & 52.38\%(-4.57\%) & 30.41\%(-5.07\%) & 0.02\%(-0.28\%) \\
# %                                     & LN5-DH         &  98.80\%(-0.18\%) & 55.63\%(-1.40\%) & 52.93\%(-4.97\%) & 53.95\%(-3.00\%) & 36.41\%(+0.93\%) & 0.01\%(-0.29\%) \\
# %                                     & LN5-FGSM       &  97.77\%(-1.21\%) & 59.35\%(+2.32\%) & 63.48\%(+5.58\%) & 61.71\%(+4.76\%) & 86.03\%(+50.55\%) & 22.22\%(+21.92\%) \\
# %                                     & LN5-PGD        &  97.04\%(-1.94\%) & 60.37\%(+3.34\%) & 61.55\%(+3.65\%) & 61.03\%(+4.08\%) & 68.70\%(+33.22\%) & 33.75\%(+33.45\%) \\
# %     \hline\multirow{6}{*}{CIFAR}    & V16             &  92.80\%(+0.00\%) & 20.95\%(+0.00\%) & 19.22\%(+0.00\%) & 20.23\%(+0.00\%) & 31.76\%(+0.00\%) & 1.10\%(0.00\%) \\
# %                                     & V16-D        &  86.34\%(-6.46\%) & 30.76\%(+9.81\%) & 27.31\%(+8.09\%) & 28.53\%(+8.30\%) & 14.50\%(-17.26\%) & 0.15\%(-0.95\%) \\
# %                                     & V16-N     &  77.72\%(-15.08\%) & 23.67\%(+2.72\%) & 26.85\%(+7.63\%) & 25.44\%(+5.21\%) & 13.32\%(-18.44\%) & 0.45\%(-0.65\%) \\
# %                                     & V16-DH          &  88.22\%(-4.58\%) & 28.35\%(+7.40\%) & 28.84\%(+9.62\%) & 28.87\%(+8.64\%) & 17.39\%(-14.37\%) & 0.48\%(-0.62\%) \\
# %                                     & V16-FGSM        &  84.28\%(-8.52\%) & 10.00\%(-10.95\%) & 9.97\%(-9.25\%) & 9.99\%(-10.24\%) & 30.56\%(-1.20\%) & 8.35\%(+7.25\%) \\
# %                                     & V16-PGD         &  87.83\%(-4.97\%) & 10.50\%(-10.45\%) & 10.18\%(-9.04\%) & 10.72\%(-9.51\%) & 54.67\%(+22.91\%) & 40.78\%(+39.68\%) \\
# %     \hline\multirow{6}{*}{SVHN}     & S3             &  92.57\%(+0.00\%) & 79.10\%(+0.00\%) & 83.29\%(+0.00\%) & 81.55\%(+0.00\%) & 11.20\%(+0.00\%) & 0.00\%(0.00\%) \\
# %                                     & S3-D        &  93.77\%(+1.20\%) & 77.92\%(-1.18\%) & 77.87\%(-5.43\%) & 78.20\%(-3.35\%) & 30.60\%(+19.40\%) & 3.88\%(+3.88\%) \\
# %                                     & S3-N     &  94.49\%(+1.92\%) & 79.49\%(+0.40\%) & 81.21\%(-2.08\%) & 80.35\%(-1.19\%) & 38.64\%(+27.44\%) & 3.33\%(+3.33\%) \\
# %                                     & S3-DH          &  94.24\%(+1.67\%) & 79.19\%(+0.10\%) & 81.13\%(-2.16\%) & 79.99\%(-1.56\%) & 29.49\%(+18.29\%) & 4.48\%(+4.48\%) \\
# %                                     & S3-FGSM        &  92.28\%(-0.29\%) & 78.35\%(-0.75\%) & 82.63\%(-0.67\%) & 80.81\%(-0.74\%) & 84.30\%(+73.09\%) & 13.53\%(+13.53\%) \\
# %                                     & S3-PGD         &  87.74\%(-4.83\%) & 74.90\%(-4.20\%) & 76.49\%(-6.80\%) & 75.80\%(-5.75\%) & 59.67\%(+48.46\%) & 50.57\%(+50.57\%) \\
# %     \hline\multirow{6}{*}{EuroSAT}  & RN20          &  97.11\%(+0.00\%) & 51.31\%(+0.00\%) & 45.94\%(+0.00\%) & 48.39\%(+0.00\%) & 11.26\%(+0.00\%) & 0.59\%(0.00\%) \\
# %                                     & RN20-D     &  89.46\%(-7.65\%) & 50.00\%(-1.31\%) & 45.52\%(-0.43\%) & 48.52\%(+0.13\%) & 10.69\%(-0.57\%) & 1.96\%(+1.37\%) \\
# %                                     & RN20-N  &  93.76\%(-3.35\%) & 52.24\%(+0.93\%) & 44.54\%(-1.41\%) & 48.19\%(-0.20\%) & 14.69\%(+3.43\%) & 1.83\%(+1.24\%) \\
# %                                     & RN20-DH       &  93.28\%(-3.83\%) & 53.07\%(+1.76\%) & 45.41\%(-0.54\%) & 49.20\%(+0.81\%) & 7.85\%(-3.41\%) & 0.22\%(-0.37\%) \\
# %                                     & RN20-FGSM     &  22.70\%(-74.41\%) & 14.65\%(-36.67\%) & 14.28\%(-31.67\%) & 14.59\%(-33.80\%) & 25.81\%(+14.56\%) & 0.02\%(-0.57\%) \\
# %                                     & RN20-PGD     &  26.26\%(-70.85\%) & 15.44\%(-35.87\%) & 15.30\%(-30.65\%) & 15.30\%(-33.09\%) & 22.87\%(+11.61\%) & 22.57\%(+21.98\%) \\
# %     \hline
# %     \end{tabular}
# %     \end{center}
# %     \label{tab:rq4}
# % \end{table*}

# adv = [98.69, 98.60, 98.80, 97.77, 97.04, 86.34, 77.72, 88.22, 84.28, 87.83, 93.77, 94.49, 94.24, 92.28, 87.74, 89.46, 93.76, 93.28, 22.70, 33.87, 93.91, 93.33, 85.70, 42.69, 39.19]
# n_a = [98.98, 98.98, 98.98, 98.98, 98.98, 92.80, 92.80, 92.80, 92.80, 92.80, 92.57, 92.57, 92.57, 92.57, 92.57, 97.11, 97.11, 97.11, 97.11, 97.11, 96.35, 96.35, 96.35, 96.35, 96.35]

# print(mannwhitneyu(adv, n_a).pvalue, mean(adv) - mean(n_a), cohend(adv, n_a))




