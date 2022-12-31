import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import matplotlib.colors
import matplotlib.ticker
import matplotlib as mpl
from scipy.stats import pearsonr

def visualize3D():
    data = pd.read_csv('data/trainval.csv')
    x = data['A1']
    y = data['A2']
    z = data['A3']
    f = data['Activity']
    min_f = min(f)
    max_f = max(f)
    color = [plt.get_cmap("rainbow", 100)(int(float(i-min_f)/(max_f-min_f)*100)) for i in f]
    fig = plt.figure(figsize=(15,10))
    ax = plt.axes(projection ="3d")
    plt.set_cmap(plt.get_cmap("rainbow", 100))
    im = ax.scatter(x, y, z, s=100,c=color,marker='.')
    fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*(max_f-min_f)+min_f)))
    ax.set_xlabel('A1')
    ax.set_ylabel('A2')
    ax.set_zlabel('A3')
    plt.savefig("data.png")

def visualize2D():
    data = pd.read_csv('data/trainval.csv')
    pool = ['A1','A2','A3','A4']
    f = data['Activity']
    min_f = min(f)
    max_f = max(f)
    color = [plt.get_cmap("rainbow", 100)(int(float(i-min_f)/(max_f-min_f)*100)) for i in f]
    for ax1 in range(4):
        for ax2 in range(ax1+1,4):
            x = data[pool[ax1]]
            y = data[pool[ax2]]
            fig = plt.figure(figsize=(15,10))
            ax = plt.axes()
            plt.set_cmap(plt.get_cmap("rainbow", 100))
            im = ax.scatter(x, y, s=100,c=color,marker='.')
            fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*(max_f-min_f)+min_f)))
            ax.set_xlabel(pool[ax1])
            ax.set_ylabel(pool[ax2])
            figname = pool[ax1] + pool[ax2] + '.png'
            plt.savefig(figname)

def pearsonfig():
    data = pd.read_csv('data/trainval.csv')
    data['weighted_A1'] = 80 * data['A1']
    data['weighted_A2'] = 113.3 * data['A2']
    data['weighted_A3'] = 54.7 * data['A3']
    data['weighted_A4'] = 84.7 * data['A4']
    data['A1A1'] = data['A1'] * data['A1']
    data['A2A2'] = data['A2'] * data['A2']
    data['A3A3'] = data['A3'] * data['A3']
    data['A4A4'] = data['A4'] * data['A4']
    data["A1A2"] = data["A1"] * data["A2"]
    data["A1A3"] = data["A1"] * data["A3"]
    data['A1A4'] = data['A1'] * data['A4']
    data["A2A3"] = data["A2"] * data["A3"]
    data['A2A4'] = data['A2'] * data['A4']
    data['A3A4'] = data['A3'] * data['A4']
    data["A1A2A3"] = data["A1"] * data["A2"] * data["A3"]
    data['A1A2A4'] = data['A1'] * data['A2'] * data['A4']
    data['A1A3A4'] = data['A1'] * data['A3'] * data['A4']
    data['A2A3A4'] = data['A2'] * data['A3'] * data['A4']
    data['A1A2A3A4'] = data['A1'] * data['A2'] * data['A3'] * data['A4']
    data['weighted'] = 80 * data['A1'] + 113.3 * data['A2'] + 54.7 * data['A3'] + 84.7 * data['A4']
    data['weighted_A1A1'] = data['weighted_A1'] * data['weighted_A1']
    data['weighted_A2A2'] = data['weighted_A2'] * data['weighted_A2']
    data['weighted_A3A3'] = data['weighted_A3'] * data['weighted_A3']
    data['weighted_A4A4'] = data['weighted_A4'] * data['weighted_A4']
    data['weighted_A1A2'] = data['weighted_A1'] * data['weighted_A2']
    data['weighted_A1A3'] = data['weighted_A1'] * data['weighted_A3']
    data['weighted_A1A4'] = data['weighted_A1'] * data['weighted_A4']
    data['weighted_A2A3'] = data['weighted_A2'] * data['weighted_A3']
    data['weighted_A2A4'] = data['weighted_A2'] * data['weighted_A4']
    data['weighted_A3A4'] = data['weighted_A3'] * data['weighted_A4']
    data['weighted_A1A2A3'] = data['weighted_A1'] * data['weighted_A2'] * data['weighted_A3']
    data['weighted_A1A2A4'] = data['weighted_A1'] * data['weighted_A2'] * data['weighted_A4']
    data['weighted_A1A3A4'] = data['weighted_A1'] * data['weighted_A3'] * data['weighted_A4']
    data['weighted_A2A3A4'] = data['weighted_A2'] * data['weighted_A3'] * data['weighted_A4']
    data['weighted_A1A2A3A4'] = data['weighted_A1'] * data['weighted_A2'] * data['weighted_A3'] * data['weighted_A4']
    X = data.drop(['Activity'], axis=1).values
    data['Activity_var'] = data['Activity'] - data['weighted']
    y = data['Activity'].values
    y_var = data['Activity_var'].values

    temp = np.hstack([X, y.reshape(-1, 1)])
    nfeature = temp.shape[1]
    corrmat = np.ones((nfeature, nfeature), dtype = float)
    for i in range(nfeature):
        for j in range(i + 1, nfeature):
            corrmat[j, i] = corrmat[i, j] = pearsonr(temp[i], temp[j])[0]
    fig, pearsonfig = plt.subplots()
    pearsonfig.matshow(corrmat, cmap = mpl.cm.spring)
    fig.colorbar(plt.cm.ScalarMappable(cmap=mpl.cm.spring), ax = pearsonfig)
    plt.savefig("Pearson.png")

if __name__ == '__main__':
    visualize3D()
    visualize2D()
