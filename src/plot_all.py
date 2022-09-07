# functions for plotting accross iml methods
from src.ebm.plot import plot_shape_function


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
            #                       x_name, y_name, model_name, dataset_name) # would be handy for plotting interactions!
            continue
        if len(shape_data_dict['names']) == 2:
            pass
            # make_one_hot_plot(shape_data['scores'][0], shape_data['scores'][1], feature_name, model_name, dataset_name)
        else:
            plot_shape_function(data_dict=shape_data_dict, feature_name=feature_name, run_id=run_id)
            