import matplotlib.pyplot as plt
import numpy as np

# Since the actual data values are not provided, I will use placeholder values.
# The user should replace these with the actual data.

# Categories
categories = ['DBLP', 'ACM', 'Cora']
plate = ['#858796', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6']

# Sample data for each model type
DBLP = [87.2, 76.0, 77.7, 83.9, 79.8]
ACM = [90.8, 81.5, 87.9, 87.1, 86.8]
Cora = [75.9, 75.4, 75.5, 73.7, 75.0]

# The width of the bars: can also be len(x) sequence
bar_width = 0.15

# Set position of bar on X axis
r1 = np.arange(3)
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

# Make the plot
plt.bar(r1, [DBLP[0], ACM[0], Cora[0]], color=plate[0], width=bar_width,  label='FeGis')
plt.bar(r2, [DBLP[1], ACM[1], Cora[1]], color=plate[1], width=bar_width, label=r'w/o unique')
plt.bar(r3, [DBLP[2], ACM[2], Cora[2]], color=plate[2], width=bar_width, label=r'w/o rec')
plt.bar(r4, [DBLP[3], ACM[3], Cora[3]], color=plate[3], width=bar_width, label=r'w/o max_MI')
plt.bar(r5, [DBLP[4], ACM[4], Cora[4]], color=plate[4], width=bar_width,  label=r'w/o min_MI')

# Add xticks on the middle of the group bars
# plt.xlabel('Model', fontweight='bold')
plt.ylabel('Accuracy', fontsize=20)
plt.xticks([r + 0.3 for r in range(3)], ['DBLP', 'ACM', 'Cora'], fontsize=15)

# Create legend & Show graphic
plt.legend(loc='upper right', ncol=1, prop={'size': 10})
plt.ylim(60, 100)
plt.yticks([60, 70, 80, 90, 100], fontsize=15)

# Show the plot
plt.savefig('alba_acc.pdf', bbox_inches='tight')
plt.show()
