import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

def show_pcd(pcd, labels, exclude_labels=[]):

    mask = np.where(~np.isin(labels, exclude_labels))[0]
    pcd_ = pcd.select_by_index(mask)
    labels_ = labels[mask]

    # mix
    mask_noise = labels_ != -1
    un_labels = np.unique(labels_[mask_noise])
    shuffle_ = np.random.choice(len(un_labels), len(un_labels),replace=False)
    inv = np.unique(labels_[mask_noise], return_inverse=True)[1]
    labels_[mask_noise] = shuffle_[inv]

    max_label = labels_.max()
    colors = plt.get_cmap("gist_rainbow")(labels_ / (max_label if max_label > 0 else 1))
    colors[labels_ < 0] = 0
    pcd_.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd_])

def show_pcd_floors(pcd, floors):
    pcds_ = []
    for i in range(len(floors)):
        mask = floors[i]
        pcd_ = pcd.select_by_index(np.where(mask)[0])
        color = plt.get_cmap("tab20")(i / len(floors))
        pcd_.paint_uniform_color(color[:3])
        pcds_.append(pcd_)

    o3d.visualization.draw_geometries(pcds_)