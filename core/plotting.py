import pandas as pd
from core.cluster_set import ClusterSet
from core.point import Point
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
from sklearn.decomposition import PCA


def create_color_it(labels):
    k = len(labels)
    color_range = list(np.linspace(0, 1, k, endpoint=False))
    colors = iter([to_hex(cm.Paired(x)) for x in color_range])
    return colors


def plot_clusters(df,
                  directory,
                  file_name_prefix):

    mem = create_cluster_membership_map(df)

    cluster_plot_file = '{0}/{1}_cluster_plt.png'.format(directory, file_name_prefix)

    #Plot the raw data
    plt.style.use('default')

    # Plot the data
    plt.figure(figsize=(8, 6))

    df['cluster_count'] = df['label'].map(mem)
    # print(df.head())

    labels = df.iloc[:, -1].unique()
    labels.sort()
    # print(labels)

    colors = create_color_it(labels)

    i = 0
    for label in labels:
        color = next(colors)
        sub_df = df[df['cluster_count'] == label]
        plt.scatter(
            sub_df.iloc[:,0],
            sub_df.iloc[:,1],
            color=color,
            alpha=0.8,
            label=label,
            s=9
        )

        i = i+1

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(file_name_prefix)
    plt.legend(loc='lower right')
    plt.savefig(cluster_plot_file)
    df.drop(['cluster_count'], axis = 1, inplace=True)
    plt.show()


def data_frame_from_cluster_memberships(val, perform_pca=False):
    if type(val) != ClusterSet:
        raise Exception('Expecting val type to be ClusterSet')

    cols = np.arange(0, Point.dimension + 1, 1).tolist()

    df = pd.DataFrame(columns=cols)

    for c in range(0, len(val)):
        for p in range(0, len(val.clusters[c].points)):
            point_to_append = [val.clusters[c].points[p].featureSet]
            cluster_no = np.array([[c]])
            point_to_append = np.append(point_to_append, cluster_no, axis=1)
            current_df = pd.DataFrame(data=point_to_append)

            df = pd.concat([df, current_df], axis=0)

    df.index = range(1, len(df)+1)
    df.columns = [*df.columns[:-1], 'label']

    if perform_pca:
        #PCA is used to to visualize the high-dimensional data.
        X = df.iloc[:, 0:-1].values
        vis_res = PCA(n_components=2).fit_transform(X)
        df['X'] = vis_res[:, 0]
        df['Y'] = vis_res[:, 1]
        df = df[['X', 'Y', 'label']]

    return df


def create_cluster_membership_map(i_df):
    mem_dict = i_df['label'].value_counts().to_dict()
    for key, value in mem_dict.items():
        mem_dict[key] = "#{0} (n={1})".format(key, str(value))
    return mem_dict
