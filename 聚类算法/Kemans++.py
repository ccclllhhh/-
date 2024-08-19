import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def select_center(first_center, df, k, colmap):
    center = {}
    center[0] = first_center

    for i in range(1, k):
        df = assignment(df, center, colmap)
        sum_closest_d = df.loc[:, 'cd'].sum()  # cd = 最近中心点的距离。把所有样本点对应最近中心点的距离都加在一起
        df["p"] = df.loc[:, 'cd'] / sum_closest_d
        sum_p = df["p"].cumsum()

        # 下面是轮盘法取新的聚类中心点
        next_center = random.random()
        for index, j in enumerate(sum_p):
            if j > next_center:
                break
        center[i] = list(df.iloc[index].values)[0:2]

    return center


def assignment(df, center, colmap):
    # 计算所有样本分别对K个类别中心点的距离
    for i in center.keys():
        df["distance_from_{}".format(i)] = np.sqrt((df["x"] - center[i][0]) ** 2 + (df["y"] - center[i][1]) ** 2)

    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in center.keys()]
    df["closest"] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)  # "closest"列表示每个样本点离哪个类别的中心点距离最近
    df["cd"] = df.loc[:, distance_from_centroid_id].min(axis=1)
    df["closest"] = df["closest"].map(lambda x: int(x.lstrip('distance_from_')))
    df["color"] = df['closest'].map(lambda x: colmap[x])
    return df


def update(df, centroids):
    # 更新K个类别的中心点
    for i in centroids.keys():
        # 每个类别的中心点为 属于该类别的点的x、y坐标的平均值
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids


def main():
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })
    k = 3
    colomap = {0: "r", 1: "g", 2: "b"}
    first_center_index = random.randint(0, len(df) - 1)
    first_center = [df['x'][first_center_index], df['y'][first_center_index]]
    center = select_center(first_center, df, k, colomap)

    df = assignment(df, center, colomap)

    for i in range(10):  # 迭代10次
        closest_center = df['closest'].copy(deep=True)
        center = update(df, center)  # 更新K个类的中心点
        df = assignment(df, center, colomap)  # 类别中心点更新后，重新计算所有样本点到K个类别中心点的距离
        if closest_center.equals(df['closest']):  # 若各个样本点对应的聚类类别不再变化，则结束聚类
            break

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='b')
    for j in center.keys():
        plt.scatter(*center[j], color=colomap[j], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()


if __name__ == '__main__':
    main()