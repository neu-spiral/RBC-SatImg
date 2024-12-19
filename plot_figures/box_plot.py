import os
from configuration import Visual
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import pickle


def plot_qa_boxplot(Config):
    """ Main function
    Results must be saved with the qa_settings['save'] = True option, and then they can be used for this boxplot.
    """
    # --------------------------------------------------------------------
    # TO BE CHANGED BY USER
    # --------------------------------------------------------------------
    # TODO: select the accuracy metric
    accuracy_metric = "balanced_accuracy"
    savefig = True
    hatch = False
    conf_id = {'2': "00", '1a': "00", '1b': "00", '3': "00"}

    # --------------------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------------------
    plt.rc('text', usetex=True)
    plt.rc('font', family='times new roman')

    test_sites = ['1a', '1b', '2', '3']
    # test_sites = ['2','3']
    scenarios = ['oroville_dam', 'oroville_dam', 'charles_river', 'multiearth']
    # scenarios = ['charles_river',"multiearth"]
    # test_sites = [3]
    # scenarios = ['charles_river']
    colors = dict()
    colors['recursive_face'] = "#a0d0d0"
    colors['recursive'] = "black"
    colors['nonrecursive'] = "black"
    # colors['nonrecursive_face'] = "#ed9c84"
    colors['nonrecursive_face'] = "#ffd95d"
    # colors['nonrecursive_face'] = "white"
    # colors['recursive_face'] = "white"
    colors['models'] = ["#a7b4ee", "#38761d", "#85595f", "#867d8c", "magenta"]

    # --------------------------------------------------------------------
    # Create figure
    # --------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(11, 5))

    # --------------------------------------------------------------------
    # Loop over test sites
    # --------------------------------------------------------------------
    # Iterate over test sites
    for i, test_site_i in enumerate(test_sites):

        axs_i = int(i > 1)
        axs_j = i % 2

        #
        #
        # Load results from test site
        path_results_metrics = os.path.join(Config.path_evaluation_results, "classification",
                                            f'{scenarios[i]}_{test_site_i}', f'accuracy',
                                            f'conf_{conf_id[test_site_i]}')
        results_qa = dict()
        models = Visual.qa_fig_settings['legend_models'][test_site_i]
        # models = pickle.load(open(os.path.join(path_results_metrics, f'models.pkl'), "rb"))
        # if test_site_i in ['3']:
        #     models = ['SIC', 'GMM', "LR", "RSIC", "RGMM", "RLR"]
        # else:
        #     models = ['SIC', 'GMM', "LR",'DWM', 'WN', "RSIC", "RGMM", "RLR",'RDWM', 'RWN',]
        for acc in Config.qa_settings['metrics']:
            path_i = os.path.join(path_results_metrics, f'{acc}.pkl')
            results_qa[acc] = pickle.load(open(path_i, "rb"))

        #
        #
        # Sort Results for Box Plot
        results_qa_sorted = np.ndarray(shape=results_qa[accuracy_metric][:, :-1].shape)
        models_sorted = models.copy()
        for model_idx, model in enumerate(models[:int(len(models) / 2)]):
            results_qa_sorted[:, model_idx * 2] = results_qa[accuracy_metric][:, model_idx]
            models_sorted[model_idx * 2] = model
            # in the sorted version we have: model - recursive model, model2 - recursive model2
            results_qa_sorted[:, model_idx * 2 + 1] = results_qa[accuracy_metric][:,
                                                      model_idx + int(len(models) / 2)]
            # models_sorted[model_idx * 2 + 1] = models[model_idx + int(len(models) / 2)]
            models_sorted[model_idx * 2 + 1] = ""

        if test_site_i in ['3']:
            positions = np.arange(0, 10)
        else:
            positions = np.arange(0, 16)
        positions = positions[(positions % 3) != 0] / 1.5
        #
        #
        # Box Plot
        # Create Box Plot and its Layout/Colors
        bp = axs[axs_i, axs_j].boxplot(results_qa_sorted, positions=positions, labels=models_sorted, notch=False,
                                       patch_artist=True, vert=True)
        set_style_bp(bp=bp, colors=colors, hatch=hatch)
        # Move xticks to right
        set_layout_bp(axs=axs, axs_i=axs_i, axs_j=axs_j, fig=fig)
    #
    #
    # Set bp legend
    set_legend_bp(colors=colors, hatch=hatch, fig=fig)

    # --------------------------------------------------------------------
    # Save figure
    # --------------------------------------------------------------------
    if Visual.qa_fig_settings['save']:
        path_store = os.path.join(Config.path_evaluation_results, "classification",
                                  f"{Config.scenario}_{Config.test_site}", "figures", "boxplot.svg")
        plt.savefig(path_store, format='svg', dpi=1000)


def set_style_bp(bp, colors, hatch):
    # Colors Box Plot
    for element in ['medians']:
        for patch_idx, patch in enumerate(bp[element]):
            print(patch)
            if patch_idx % 2 == 0:
                plt.setp(patch, color=colors['nonrecursive'])
            else:
                plt.setp(patch, color=colors['recursive'])
    for element in ['fliers']:  # Outliers (fliers)
        for patch_idx, patch in enumerate(bp[element]):
            if patch_idx % 2 == 0:
                patch.set(marker='o', markeredgecolor=colors['nonrecursive'],
                          markerfacecolor=colors['nonrecursive_face'])
            else:
                patch.set(marker='o', markeredgecolor=colors['recursive'], markerfacecolor=colors['recursive_face'])
    for element in ['whiskers', 'caps']:
        for patch_idx, patch in enumerate(bp[element]):
            if patch_idx % 4 in [0, 1]:
                plt.setp(patch, color=colors['nonrecursive'])
            else:
                plt.setp(patch, color=colors['recursive'])
    for patch_idx, patch in enumerate(bp['boxes']):
        if patch_idx % 2 == 0:
            plt.setp(patch, color=colors['nonrecursive'])
            patch.set(facecolor=colors['nonrecursive_face'])
        else:
            if hatch:
                plt.setp(patch, color=colors['recursive'], hatch='/')
            else:
                plt.setp(patch, color=colors['recursive'])
            patch.set(facecolor=colors['recursive_face'])


def set_layout_bp(axs, axs_i, axs_j, fig):
    # Move xticks to right
    current_ticks = axs[axs_i, axs_j].get_xticks()
    current_labels = axs[axs_i, axs_j].get_xticklabels()

    # Shift the x-axis ticks slightly to the right
    new_ticks = [tick + 0.35 for tick in current_ticks]  # Adjust the value as needed for your plot

    # Set the new x-axis ticks and labels
    axs[axs_i, axs_j].set_xticks(new_ticks)
    axs[axs_i, axs_j].set_xticklabels(current_labels, fontsize=12)
    current_labels_y = axs[axs_i, axs_j].get_yticklabels()
    axs[axs_i, axs_j].set_yticklabels(current_labels_y, fontsize=12)
    axs[axs_i, axs_j].tick_params(axis='x', which='both', bottom=False, top=False)

    for j in [0, 1]:
        # axs[axs_i, axs_j].set_ylabel('Accuracy [%]')
        axs[axs_i, axs_j].grid(alpha=0.2)

    axs[0, 0].set_title('Test site 1a', fontsize=16)
    axs[0, 1].set_title('Test site 1b', fontsize=16)
    axs[1, 0].set_title('Test site 2', fontsize=16)
    axs[1, 1].set_title('Test site 3', fontsize=16)
    fig.text(0, 0.5, 'Balanced Accuracy [\%]', va='center', rotation='vertical', fontsize=16)


def set_legend_bp(colors, hatch, fig):
    if hatch:
        legend_handles = [patches.Rectangle((0, 0), 1, 1, linewidth=0, facecolor=colors['nonrecursive_face']),
                          plt.Rectangle((0, 0), 1, 1, facecolor=colors['recursive_face'], hatch='///')]
    else:
        legend_handles = [patches.Rectangle((0, 0), 1, 1, linewidth=0, facecolor=colors['nonrecursive_face']),
                          plt.Rectangle((0, 0), 1, 1, facecolor=colors['recursive_face'])]
    legend_labels = ['Non-recursive', 'Recursive']

    # Displaying the legend with custom handles and labels
    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=14)
    plt.tight_layout(pad=3, w_pad=2, h_pad=2.0)
