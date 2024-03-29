import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random



def visualize_sample_prediction(y_test, y_pred, data):
    """
    Plots a random permno's predicted return vs actual returns
    
    Parameters
    ----------
        y_test (array-like): Actual returns
        y_pred (array-like): Pedicted returns
        data (pd.DataFrame): The sample to draw permnos from
    """
    # Picks a random stock, to visualize its real returns 
    # against the predicted returns
    permnos_total = data.index.droplevel(['date'])
    permnos_total = permnos_total.unique()
    
    sample_permno = random.choice(permnos_total)

    sample_test_y = y_test.xs(sample_permno, level='permno')
    
    sample_test_idx = sample_test_y.index.get_level_values('date')
    
    y_pred = pd.Series(y_pred, index=y_test.index)
    sample_pred = y_pred.xs(sample_permno, level='permno')
    
    fig,ax=plt.subplots(figsize=(18,6))

    ax.scatter(x=sample_test_idx, y= sample_test_y, color='r')
    ax.scatter(x=sample_test_idx, y= sample_pred, color='b')
    plt.title(f'Stock nr {sample_permno} predicted returns (blue) vs actual returns (red)')
    

def importance_bar_chart(feature_importance_df, effect_names, model_dir='./models/results/', model_name='EBM', save=False, id='00', ir=False):

    # sort unsorted df by importance
    if not ir:
        feature_importance_df.sort_values(by='importance score', ascending=False, axis=0, inplace=True)
        importance = feature_importance_df['importance score']
    else:
        feature_importance_df.sort_values(by='importance ratio', ascending=False, axis=0, inplace=True)
        importance = feature_importance_df['importance ratio']
    
    n_features = feature_importance_df.shape[0]
    figsize=(6, 0.25*n_features)

    #plt.rcdefaults()
    fig, ax = plt.subplots(figsize=figsize)
    #fig.text(0.001, 0.5, 'Variable', ha='center', va='center', rotation='vertical', fontsize='large')
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    # ylabels
    #features = effect_names
    features_latex = []
    for name in effect_names:
        name_latex = fr'${name}$'
        features_latex.append(name_latex)

    y_pos = np.arange(len(effect_names))
    
    ax.barh(y_pos, importance, align='center')
    ax.set_yticks(y_pos, labels=features_latex)
    ax.invert_yaxis()  # labels read top-to-bottom
    if not ir:
        ax.set_xlabel('Importance')
    else:
        ax.set_xlabel('Importance ratio')
    #ax.set_title('Average global feature importance')
    plt.show()
    
    if save:
        fig.savefig(f'{model_dir}{model_name}_feature_importances_{id}.png', bbox_inches='tight', dpi=200)

def plot_shape_function(data_dict, feature_name, dataset_name='sub', run_id='', debug=False, save=False):
    """
    Plot shape function of one given feature.
    Params: 
        :positionals: data_dict, feature_name, cum_cols, scaler_dict
        :defaults: dataset_name, run_id, debug
    """
    x_vals = data_dict["names"].copy()
    y_vals = data_dict["scores"].copy()

    # This is important since you do not plot plt.stairs with len(edges) == len(vals) + 1, which will have a drop to zero at the end
    y_vals = np.r_[y_vals, y_vals[np.newaxis, -1]] 
    x = np.array(x_vals)
    
    # if debug:
    #     print("Num cols:", num_cols)
    # if feature_name in num_cols:
    #     if debug:
    #         print("Feature to scale back:", feature_name)
        #x = scaler_dict[feature_name].inverse_transform(x.reshape(-1, 1)).squeeze()
    # else:
    #     if debug:
    #         print("Feature not to scale back:", feature_name)
    
    plt.xlim(left=-3, right=3) # temporary fix, better: rescale X
    plt.step(x, y_vals, where="post") #, color='black')
    # plt.fill_between(x, lower_bounds, mean, color='gray')
    # plt.fill_between(x, mean, upper_bounds, color='gray')
    plt.xlabel(f'Feature value')
    plt.ylabel('Feature effect on model output')
    plt.title(f'Feature:{feature_name}')
    if save:
        plt.savefig(f'figures/ebm/{run_id}_{dataset_name}_shape_{feature_name}_.png')
    plt.show()

def make_plot_interaction(shape_data_left, shape_data_right,
                          scores, x_name, y_name, plot_name='interaction_plot'):
    # To-Do: implement heatmap with x-y and score as hue!
    # plots
    return None