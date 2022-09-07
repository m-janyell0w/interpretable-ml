# functions for plotting accross iml methods
from src.ebm.plot import plot_shape_function, make_plot_interaction


def ebm_plot_all(shape_data_dict, importance_data_df, n_features=10, run_id=''):
    """
    Plots shape functions for a given data dictionary

    Params:

        shape_data_dict (dict): dict containing x and y values of shape functions and some
                                meta data
        importance_data_df (pd.DataFrame): df containing effect name and importance score
        n_features (int): The number of features to plot
        run_id (str): wandb run-id for file saving 
    """

    for i in range(n_features):
        #data_names = ebm_global.data()
        feature_name = importance_data_df.iloc[i,'names']
        #shape_data = ebm_global.data(i) -> shape_data_dict

        ## TO DO: make_plot_interaction?? implement or find
        if shape_data_dict['type'] == 'interaction':
            # x_name, y_name = feature_name.split('x')
            # x_name = x_name.replace(' ', '')
            # y_name = y_name.replace(' ', '')
            # make_plot_interaction(shape_data['left_names'], shape_data['right_names'],
            #                       np.transpose(shape_data['scores']),
            #                       x_name, y_name, plot_name) # would be handy for plotting interactions!
            continue
        if len(shape_data_dict['names']) == 2:
            pass
            # make_one_hot_plot(shape_data['scores'][0], shape_data['scores'][1], feature_name, model_name, dataset_name)
        else:
            plot_shape_function(data_dict=shape_data_dict, feature_name=feature_name, run_id=run_id)


# from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html?highlight=heatmap, all rights reserved.

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar