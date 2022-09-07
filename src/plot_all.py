# functions for plotting accross iml methods

def ebm_plot_all(shape_data_dict, num_cols, scaler_dict={}, run_id=''):

    for i in range(len(shape_data_dict['names'])):
        data_names = ebm_global.data()
        feature_name = data_names['names'][i]
        shape_data = ebm_global.data(i)

        ## TO DO: make_plot_interaction?? implement or find
        if shape_data['type'] == 'interaction':
            # x_name, y_name = feature_name.split('x')
            # x_name = x_name.replace(' ', '')
            # y_name = y_name.replace(' ', '')
            # make_plot_interaction(shape_data['left_names'], shape_data['right_names'],
            #                       np.transpose(shape_data['scores']),
            #                       x_name, y_name, model_name, dataset_name) # would be handy for plotting interactions!
            continue
        if len(shape_data['names']) == 2:
            pass
            # make_one_hot_plot(shape_data['scores'][0], shape_data['scores'][1], feature_name, model_name, dataset_name)
        else:
            plot_shape_function(data_dict=shape_data, feature_name=feature_name, num_cols=num_cols, scaler_dict={}, run_id=run_id)