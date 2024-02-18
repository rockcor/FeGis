import matplotlib.pyplot as plt
import matplotlib

# Placeholder data

acm_ut = [88.58, 91.9, 90.8, 88.9, 88.1, 87.7]
acm_auc = [85.42, 83.8, 83.7, 83.1, 81.6, 76.3]
acm_gcn_auc = [86.8, 86.4, 86.5, 85.8, 82.2, 80.7]
acm_gcn_ut = [88.6, 89.90, 89.35, 89.37, 89.8, 89.42]
acm_auc_list = [100 - i for i in acm_auc]
acm_gcn_auc_list = [100 - i for i in acm_gcn_auc]

cora_ut = [77.63, 76.3, 75.9, 77.2, 76.9, 75.3]
cora_gcn_ut = [78.30, 75.83, 74.83, 74.53, 75.63, 75.73]
cora_gcn_auc = [93.23, 81.17, 75.39, 74.37, 71.11, 60.30]
cora_auc = [92.25, 78.7, 65.4, 56.1, 51.6, 50.8]
cora_auc_list = [100 - i for i in cora_auc]
cora_gcn_auc_list = [100 - i for i in cora_gcn_auc]

plate = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6']
# acm,cora
vf_ut = [84.5, 77.8]
vf_auc = [91.1, 96.8]

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(acm_auc_list, acm_ut, 'o-', linewidth=3, markersize=10, color=plate[2], label='FeGis')
axes[0].scatter(acm_gcn_auc_list, acm_gcn_ut, marker='x',s=100 ,color=plate[1], label='FeGis-homo')
axes[0].scatter(100 - vf_auc[0], vf_ut[0], marker='^', s=200, color=plate[0], label='VFGNN')
axes[0].set_ylim([83, 93])
axes[0].set_title('ACM',fontsize=15)

axes[1].plot(cora_auc_list, cora_ut, 'o-', linewidth=3, markersize=10, color=plate[2], label='FeGis')
axes[1].scatter(cora_gcn_auc_list, cora_gcn_ut, marker='x',s=100,  color=plate[1], label='FeGis-homo')
axes[1].scatter(100 - vf_auc[1], vf_ut[1], marker='^', s=200, color=plate[0], label='VFGNN')
axes[1].set_ylim([73, 78])
axes[1].set_title('Cora',fontsize=15)

# Labels and Title
fig.text(0.5, -0.03, 'Privacy protection performance (1 – AUC score)', ha='center', fontsize=18)
fig.text(-0.02, 0.5, 'Utility on node classification (Accuracy)', va='center', rotation='vertical', fontsize=18)

# Legend
values = [1, 8, 16, 32, 64, 128]
axes[0].legend(fontsize=14)
axes[1].legend(fontsize=14)
for i, v in enumerate(values):
    axes[0].text(acm_auc_list[i], acm_ut[i] + 0.3, "%d" % v, ha="center", fontsize=18)
    axes[1].text(cora_auc_list[i], cora_ut[i] + 0.1, "%d" % v, ha="center", fontsize=18)
axes[0].text(100 - vf_auc[0], vf_ut[0] + 0.4, "%d" % 1, ha="center", fontsize=18)
axes[1].text(100 - vf_auc[1], vf_ut[1] - 0.5, "%d" % 1, ha="center", fontsize=18)
# Show plot
plt.tight_layout()
plt.savefig('utility.pdf', format='pdf', bbox_inches='tight')
plt.show()
