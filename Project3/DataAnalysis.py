import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""
Data Analysis
"""

plt.style.use("seaborn")
#sns.set(font_scale=1.5)
plt.rcParams["font.family"] = "Times New Roman"; plt.rcParams['axes.titlesize'] = 21; plt.rcParams['axes.labelsize'] = 21; plt.rcParams["xtick.labelsize"] = 21; plt.rcParams["ytick.labelsize"] = 21; plt.rcParams["legend.fontsize"] = 18


df = pd.read_csv("water_potability.csv")
df = df.dropna()

#Pie chart to show percentage of potability
plt.pie([df['Potability'].value_counts()[0], df['Potability'].value_counts()[1]], labels = ["Non-potable", "Potable"], autopct='%1.1f%%')
plt.grid()
plt.savefig("potable_pie.pdf", bbox_inches='tight')
plt.show()

#Boxplots
sns.boxplot(data = df.drop(["Potability"], axis=1), orient='h')

plt.savefig("potable_box1.pdf", bbox_inches='tight')
plt.show()

sns.boxplot(data = df.drop(["Solids", "Potability"], axis=1), orient='h')
plt.savefig("potable_box2.pdf", bbox_inches='tight')
plt.show()

sns.boxplot(data = df.drop(["Solids", "Hardness", "Sulfate", "Trihalomethanes", "Conductivity", "Potability"], axis=1), orient='h')
plt.savefig("potable_box3.pdf", bbox_inches='tight')
plt.show()

#Correlation Matrix
df2 = df.drop(["Potability"], axis=1) # design matrix
corr=df2.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(250, 15, s=75, l=40,
                            n=9, center="light", as_cmap=True)
f, ax = plt.subplots(figsize=(18, 14))
sns.set(font_scale=3)
sns.heatmap(corr, mask=mask, center=0, annot=True,
            fmt='.2f', square=True, cmap=cmap, linewidths=3.1)
plt.xticks(fontsize=37)
plt.yticks(fontsize=37)
plt.title("Correlation Matrix", fontsize=37)
plt.savefig("potable_corr.pdf", bbox_inches='tight')
plt.show();

#Histogram
plt.clf()
f, ax = plt.subplots(figsize=(18, 14))
#f, ax = plt.subplots()
df2.hist(ax=ax, xlabelsize=30, ylabelsize=30)
plt.tight_layout()
plt.savefig("potable_hist.pdf", dpi=400, bbox_inches='tight')
plt.show()
