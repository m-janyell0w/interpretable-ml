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
    

def importance_bar_chart(feature_importance_df, model_dir='./models/ebm/' , save=False, id='00'):
    
    # sort unsorted df by importance
    feature_importance_df.sort_values(by='importance score', ascending=False, axis=0, inplace=True)
    
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # ylabels
    features = feature_importance_df.feature
    y_pos = np.arange(len(features))
    importance = feature_importance_df['importance score']
    
    ax.barh(y_pos, importance, align='center')
    ax.set_yticks(y_pos, labels=features)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title('Average global feature importance')

    plt.show()
    
    if save:
        plt.savefig(f'{model_dir}ebm_importances_{id}.png')