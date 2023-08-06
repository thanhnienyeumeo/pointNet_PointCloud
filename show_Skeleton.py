import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import matplotlib.animation as animation
def test():
    data_pcs = np.load("D:\Documents\AIProject\HAR\pointnet2\Pointnet_Pointnet2_pytorch\log\classification\data_196\X_test.npy")
    data_labels = np.load('D:\Documents\AIProject\HAR\pointnet2\Pointnet_Pointnet2_pytorch\log\classification\data_196\y_test.npy')

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), subplot_kw={'projection': '3d'})

    def update_frame(idx):
        axs[0].clear()
        axs[0].scatter(data_pcs[idx,:, 0], data_pcs[idx,:,2], data_pcs[idx,:,1])  
        axs[0].set_xlim(-2, 2)
        axs[0].set_ylim(0, 6)
        axs[0].set_zlim(-2, 2)
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].set_zlabel('Z')
        axs[0].view_init(10, -100)
        axs[0].set_xticks([-2, -1, 0, 1, 2])
        axs[0].set_yticks([0, 2, 4, 6])
        axs[0].set_zticks([-2, -1, 0, 1, 2])

        axs[1].clear()
        axs[1].scatter(data_labels[idx,:, 0], data_labels[idx,:,2], data_labels[idx,:,1])  
        axs[1].set_xlim(-2, 2)
        axs[1].set_ylim(0, 6)
        axs[1].set_zlim(-2, 2)
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].set_zlabel('Z')
        axs[1].view_init(10, -100)
        axs[1].set_xticks([-2, -1, 0, 1, 2])
        axs[1].set_yticks([0, 2, 4, 6])
        axs[1].set_zticks([-2, -1, 0, 1, 2])
        
    ani = animation.FuncAnimation(fig, update_frame, frames=data_labels.shape[0], interval=50)
    plt.show()

def visualize(data_pc, pred_label, true_label):
        #data_pcs = np.load("D:\Documents\AIProject\HAR\pointnet2\Pointnet_Pointnet2_pytorch\log\classification\data_196\X_test.npy")
        #pred_labels = np.load('D:\Documents\AIProject\HAR\pointnet2\Pointnet_Pointnet2_pytorch\log\classification\data_196\y_test.npy')

        fig, axs = plt.subplots(1, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
        axs[0].clear()
        axs[0].scatter(data_pc[:, 0], data_pc[:,2], data_pc[:,1])  
        axs[0].set_xlim(-2, 2)
        axs[0].set_ylim(0, 6)
        axs[0].set_zlim(-2, 2)
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].set_zlabel('Z')
        axs[0].view_init(10, -100)
        axs[0].set_xticks([-2, -1, 0, 1, 2])
        axs[0].set_yticks([0, 2, 4, 6])
        axs[0].set_zticks([-2, -1, 0, 1, 2])

        axs[1].clear()
        axs[1].scatter(pred_label[:, 0], pred_label[:,2], pred_label[:,1])  
        axs[1].set_xlim(-2, 2)
        axs[1].set_ylim(0, 6)
        axs[1].set_zlim(-2, 2)
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].set_zlabel('Z')
        axs[1].view_init(10, -100)
        axs[1].set_xticks([-2, -1, 0, 1, 2])
        axs[1].set_yticks([0, 2, 4, 6])
        axs[1].set_zticks([-2, -1, 0, 1, 2])

        axs[2].clear()
        axs[2].scatter(true_label[:, 0], true_label[:,2], true_label[:,1])
        axs[2].set_xlim(-2, 2)
        axs[2].set_ylim(0, 6)
        axs[2].set_zlim(-2, 2)
        axs[2].set_xlabel('X')
        axs[2].set_ylabel('Y')
        axs[2].set_zlabel('Z')
        axs[2].view_init(10, -100)
        axs[2].set_xticks([-2, -1, 0, 1, 2])
        axs[2].set_yticks([0, 2, 4, 6])
        axs[2].set_zticks([-2, -1, 0, 1, 2])

        plt.show()
