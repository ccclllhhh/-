import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def assignment(df, center, colmap):
    # 计算所有样本分别对K个类别中心点的距离
    for i in center.keys():
        df["distance_from_{}".format(i)] = np.sqrt((df["x"] - center[i][0]) ** 2 + (df["y"] - center[i][1]) ** 2)

    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in center.keys()]
    df["closest"] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)  # "closest"列表示每个样本点离哪个类别的中心点距离最近
    print(df["closest"])
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

    # 一开始随机指定 K个类的中心点
    center = {
        i: [np.random.randint(0, 80), np.random.randint(0, 80)]
        for i in range(k)
    }

    colmap = {0: "r", 1: "g", 2: "b"}
    df = assignment(df, center, colmap)

    for i in range(10):  # 迭代10次
        closest_center = df['closest'].copy(deep=True)
        center = update(df, center)  # 更新K个类的中心点
        df = assignment(df, center, colmap)  # 类别中心点更新后，重新计算所有样本点到K个类别中心点的距离
        if closest_center.equals(df['closest']):  # 若各个样本点对应的聚类类别不再变化，则结束聚类
            break

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='b')
    for j in center.keys():
        plt.scatter(*center[j], color=colmap[j], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()


if __name__ == '__main__':
    main()