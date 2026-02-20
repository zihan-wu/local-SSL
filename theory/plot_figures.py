import numpy as np
import matplotlib.pyplot as plt
import os   
import sys


def plot_figure_2():
    """Plot Figure 2"""
    def collect_layer(dict, key, ep):
        return np.array([dict[key.format(i)][ep] for i in range(6)])

    grad_align_lw_linear_ortho = np.load('./stats/grad_align_128x6_linear_lw_fastb_orthogonal.npy', allow_pickle=True).item()
    grad_align_lw_linear_default = np.load('./stats/grad_align_128x6_linear_lw_fastb_default.npy', allow_pickle=True).item()
    grad_align_lw_relu_default_logsig= np.load('./stats/grad_align_128x6_relu_lw_fastb_default.npy', allow_pickle=True).item()
    grad_align_lw_relu_orthogonal_logsig= np.load('./stats/grad_align_128x6_relu_lw_fastb_orthogonal.npy', allow_pickle=True).item()
    layer_list = np.arange(1, 7)
    scale = 1.97/np.sqrt(32)
    plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2, 'legend.fontsize': 11, 'axes.spines.right': False, 'axes.spines.top':False, 'legend.framealpha': 0.6, 'xtick.labelsize':14, 'ytick.labelsize':14, 'figure.figsize': (7, 4)})

    plt.errorbar(layer_list, collect_layer(grad_align_lw_linear_ortho, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_lw_linear_ortho, 'layer_{}_weight_std', -1), color = 'C2', alpha=0.7, label='With all assumptions (Theorem 1)')
    plt.errorbar(layer_list, collect_layer(grad_align_lw_linear_ortho, 'layer_{}_weight', 0), yerr=scale*collect_layer(grad_align_lw_linear_ortho, 'layer_{}_weight_std', 0), color='C3', alpha=0.7, label='Random fixed B')
    plt.errorbar(layer_list, collect_layer(grad_align_lw_linear_default, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_lw_linear_default, 'layer_{}_weight_std', -1), alpha=0.7, label='Non-orthogonal W')
    plt.errorbar(layer_list, collect_layer(grad_align_lw_relu_orthogonal_logsig, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_lw_relu_orthogonal_logsig, 'layer_{}_weight_std', -1), alpha=0.7, label='ReLU MLP')
    plt.errorbar(layer_list, collect_layer(grad_align_lw_relu_default_logsig, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_lw_relu_default_logsig, 'layer_{}_weight_std', -1), alpha=0.7, color = 'C4', label='Non-orthogonal W, ReLU MLP')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.94), fontsize=10)
    plt.xlabel('Layer', fontsize=16)
    plt.ylabel('cosine similarity of gradient update')
    plt.ylim([-0.2, 1.05])

    plt.savefig('Figure_2.svg', transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
    pass


def plot_figure_3a():
    """Plot Figure 3a"""
    def collect_layer(dict, key, ep):
        return np.array([dict[key.format(i)][ep] for i in range(6)])

    grad_align_lw_lowrank = np.load('./stats/grad_align_128to4_linear_lw_fastb_orthogonal_l2.npy', allow_pickle=True).item()
    grad_align_fb_lowrank = np.load('./stats/grad_align_128to4_linear_fb_fastb_orthogonal_l2.npy', allow_pickle=True).item()

    grad_align_lw_lowrank_logsig = np.load('./stats/grad_align_128to4_linear_lw_fastb_orthogonal_logsigl2.npy', allow_pickle=True).item()
    grad_align_fb_lowrank_logsig = np.load('./stats/grad_align_128to4_linear_fb_fastb_orthogonal_logsigl2.npy', allow_pickle=True).item()

    layer_list = np.arange(1, 7)
    plt.rcParams.update({'font.size': 14, 'lines.linewidth': 2, 'legend.fontsize': 12, 'axes.spines.right': False, 'axes.spines.top':False, 'legend.framealpha': 0.6, 'xtick.labelsize':16, 'ytick.labelsize':16, 'figure.figsize':(7,5)})
    scale = 1.97/np.sqrt(32)
    plt.errorbar(layer_list, collect_layer(grad_align_lw_lowrank, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_lw_lowrank, 'layer_{}_weight_std', -1), alpha=0.7, label='Linear f')
    plt.errorbar(layer_list, collect_layer(grad_align_fb_lowrank, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_fb_lowrank, 'layer_{}_weight_std', -1), alpha=0.7, label='Linear f, DFB')

    plt.errorbar(layer_list, collect_layer(grad_align_lw_lowrank_logsig, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_lw_lowrank_logsig, 'layer_{}_weight_std', -1), alpha=0.7, color='C0', linestyle='-.', label='Softplus f')
    plt.errorbar(layer_list, collect_layer(grad_align_fb_lowrank_logsig, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_fb_lowrank_logsig, 'layer_{}_weight_std', -1), alpha=0.7, color='C1', linestyle='-.', label='Softplus f, DFB')

    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=13)
    plt.xlabel('Layer', fontsize=20)
    plt.ylabel('Weight Update Cos-Similarity', fontsize=16)
    plt.ylim([0, 1.05])

    plt.savefig('Figure_3a.svg', transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
    return


def plot_figure_3b():
    """Plot Figure 3b"""
    def collect_layer(dict, key, ep):
        return np.array([dict[key.format(i)][ep] for i in range(6)])
    
    clapp = np.load('stats/grad_align_clapp_lw_512x6_logsigl2_seed42.npy', allow_pickle=True).item()
    clapp_fb = np.load('stats/grad_align_clapp_fb_512x6_logsigl2_seed42.npy', allow_pickle=True).item()
    clapp_fb_train_fb_grad = np.load('stats/grad_align_clapp_fb_512x6_train_fb_with_grad_logsigl2_seed42.npy', allow_pickle=True).item()
    clapp_fb_frozen = np.load('stats/grad_align_clapp_fb_512x6_frozen_fb_logsigl2_seed42.npy', allow_pickle=True).item()

    plt.rcParams.update({'font.size': 14, 'lines.linewidth': 2, 'axes.spines.right': False, 'axes.spines.top':False, 'legend.framealpha': 0.6, 'mathtext.default': 'regular', 'xtick.labelsize':16, 'ytick.labelsize':16, 'legend.fontsize': 11, 'figure.figsize': (7, 5)})
    layer_list = np.arange(1, 7)
    scale = 1.96/np.sqrt(312)  # 95% confidence interval for 312 samples
    plt.errorbar(layer_list, collect_layer(clapp, 'layer_{}_weight', -1), yerr=collect_layer(clapp, 'layer_{}_weight_std', -1)*scale, alpha=0.7, label='local-SSL')
    plt.errorbar(layer_list, collect_layer(clapp_fb, 'layer_{}_weight', -1), yerr=collect_layer(clapp_fb, 'layer_{}_weight_std', -1)*scale, alpha=0.7, label='local-SSL, DFB')

    plt.errorbar(layer_list, collect_layer(clapp_fb_train_fb_grad, 'layer_{}_weight', -1), yerr=collect_layer(clapp_fb_train_fb_grad, 'layer_{}_weight_std', -1)*scale, alpha=0.7, color='C4', label='local-SSL, theoretical optimal update')
    # plt.errorbar(layer_list, collect_layer(clapp_fb_train_fb_grad_cos, 'layer_{}_weight', -1), yerr=collect_layer(clapp_fb_train_fb_grad_cos, 'layer_{}_weight_std', -1)*scale, alpha=0.7, label='Train fb to match bp grad (cos)')
    plt.errorbar(layer_list, collect_layer(clapp_fb_frozen, 'layer_{}_weight', -1), yerr=collect_layer(clapp_fb_frozen, 'layer_{}_weight_std', -1)*scale, alpha=0.7, color = 'gray', label='local-SSL, random fixed feedback')

    plt.legend(bbox_to_anchor=(0, 1.0), loc='upper left', fontsize=13)
    plt.ylabel('Weight Update Cos-Similarity', fontsize=16)
    plt.xlabel('Layer', fontsize=20)
    plt.ylim([0, 1.1])
    plt.savefig('Figure_3b.svg', transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
    return



def plot_figure_4b():
    """Plot Figure 4b"""
    def collect_layer(dict, key, ep):
        return np.array([dict[key.format(i)][ep] for i in range(4)])

    grad_align_cnn_lwp_ortho = np.load('./stats/grad_align_cnnlinear_32x4_lwp_fastb_orthogonal_logsigl2.npy', allow_pickle=True).item()
    grad_align_cnn_slw_ortho = np.load('./stats/grad_align_cnnlinear_32x4_lw_fastb_orthogonal_logsigl2.npy', allow_pickle=True).item()
    grad_align_cnn_sfb_ortho = np.load('./stats/grad_align_cnnlinear_32x4_dfb_fastb_orthogonal_logsigl2.npy', allow_pickle=True).item()

    plt.rcParams.update({'font.size': 14, 'lines.linewidth': 2, 'legend.fontsize': 14, 'axes.spines.right': False, 'axes.spines.top':False, 'legend.framealpha': 0.6, 'figure.figsize': (7, 6), 'mathtext.default': 'regular', 'xtick.labelsize':16, 'ytick.labelsize':16})
    layer_list = np.arange(1, 5)
    def collect_layer(dict, key, ep):
        return np.array([dict[key.format(i)][ep] for i in range(4)])
    scale = 1.97/np.sqrt(32)

    plt.errorbar(layer_list, collect_layer(grad_align_cnn_lwp_ortho, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_cnn_lwp_ortho, 'layer_{}_weight_std', -1), alpha=0.7, color='C3', label='CNN, without 2D spatial dependence')
    plt.errorbar(layer_list, collect_layer(grad_align_cnn_slw_ortho, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_cnn_slw_ortho, 'layer_{}_weight_std', -1), alpha=0.7, color='C0', label='CNN, with 2D 2D spatial dependence')
    plt.errorbar(layer_list, collect_layer(grad_align_cnn_sfb_ortho, 'layer_{}_weight', -1), yerr=scale*collect_layer(grad_align_cnn_sfb_ortho, 'layer_{}_weight_std', -1), alpha=0.7, color='C1', label='CNN, with 2D spatial dependence and DFB')

    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97), fontsize=12)
    plt.xlabel('Layer', fontsize=20)
    plt.xticks(layer_list)
    plt.ylabel('Weight Update Cos-Similarity', fontsize=16)
    plt.ylim([-0.1, 1.05])
    plt.savefig('Figure_4b.svg', transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
    return


def plot_figure_4c():
    """Plot Figure 4c"""
    pass


if __name__ == "__main__":
    
    figures = {
        "2": plot_figure_2,
        "3a": plot_figure_3a,
        "3b": plot_figure_3b,
        "4b": plot_figure_4b,
        "4c": plot_figure_4c,
    }
    
    fig_num = sys.argv[1]
    if fig_num in figures:
        figures[fig_num]()
    else:
        print(f"Unknown figure: {fig_num}")
        print(f"Available: {', '.join(figures.keys())}")