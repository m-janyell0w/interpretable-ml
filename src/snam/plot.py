# plot shape functions and the like
import matplotlib.pyplot as plt


def plot_shape_functions(X_train, n_features, f_out_tr, nrows=2, ncols=4, save=False, run_id='00'):
    """
    Plots learned shape functions where the amount of plots depends the nrows and ncols
    """
    
    columns = X_train.columns
    features = n_features

    f_tr = f_out_tr
    
    plt.title('Shape functions for each feature learned by SNAM subnetworks')
    
    w = ncols*4
    h = nrows*4.5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(w, h))
    
    i = 0
    for row in axes:
        for col in row: 

            col.scatter(X_train.iloc[:, i], f_tr[:, [i]], label='SNAM')
            col.xaxis.set_tick_params(labelsize=12)
            col.yaxis.set_tick_params(labelsize=12)
            col.set_xlabel(f'{columns[i]}', fontsize='14')
            col.set_ylabel(r'$f_{}(x)$'.format(i+1), fontsize='14')
            i += 1
    
    
    fig.tight_layout()
    if save:
        fig.savefig(f'./figures/snam/shape_functions_{run_id}.png', bbox_inches='tight')
    plt.show()

# TO-DO: make it work!
def plot_shape_functions_xu(f_out_tr, save=False, run_id=""):
    feature_names = X_train.columns
    # Plot shape functions
    f_tr = f_out_tr
    for i in range(8):
        if i < y_train.shape[0]:
            f_i = y_train.to_numpy().reshape(-1,1) #if y is pd.series
        else: f_i = np.zeros_like(X_train[:, i])
        #print(f_i.shape, "-", f_tr.shape)
        c_i=np.mean(f_tr[:,[i]]-f_i)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 12))

    i = 0
    for row in axes:
        for col in row:
            if i < y_train.shape[0]:
                f_i = y_train.to_numpy().reshape(-1,1) #if y is pd.series
            else: 
                f_i = np.zeros_like(X_train[:, i])
            c_i=np.mean(f_tr[:,[i]]-f_i) 
            col.scatter(X_train.iloc[:, i], f_tr[:, [i]]-c_i, label='SNAM')
            #col.scatter(X_train, features[i](X_train))
            if i == 5:
                col.legend(fontsize='15')
            col.xaxis.set_tick_params(labelsize=15)
            col.yaxis.set_tick_params(labelsize=15)
            col.set_xlabel(f'{feature_names[i]}', fontsize='20')
            col.set_ylabel(r'$f_{}(x)$'.format(i+1), fontsize='20')
            i += 1

    fig.tight_layout()
    if save:
        fig.savefig(path+'shape_functions_snam_'+str(run_id)+'.png', bbox_inches='tight')
    plt.show() 