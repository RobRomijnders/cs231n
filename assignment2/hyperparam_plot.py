import numpy as np
import matplotlib.pyplot as plt

def poly_contour_plot(results, **kwargs):
    """
    Takes as an input the dict containing tuples 
    results[(lr,reg)] = (acc_train,acc_val)
    
    Functions fits a 2D-polynomial to the data in log10-space
    that is ax^2+by^2+cx+dy*exy+f
    
    Returns a contour plot for the hyperparameters
   
    """
    if  len(kwargs)>0:
        best = kwargs.pop('best',{})
        best_train_acc_hist = best['best_train_acc_hist']
        best_val_acc_hist = best['best_val_acc_hist']
        best_loss_hist = best['best_loss_hist']
        plt.plot(best_train_acc_hist)
        plt.plot(best_val_acc_hist)
        plt.legend(['train','val'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracies')
        plt.title('Accuracies for the best model')
        plt.show()
        plt.plot(best_loss_hist)
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.title('Loss function for the best model')
        plt.show()
    
    if len(results) < 6:
        print 'Function needs at least six points for polynomial interpolation'
        return
    N = len(results)
    data = np.zeros((N,3))
    
    # Load the data from the dictionary into a data matrix
    i=0
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        data[i] = np.array([np.log10(lr), np.log10(reg), val_accuracy])
        i+=1
    
    # generate columns for the given lr, reg and the validation accuracy
    lr_d = data[:,0]
    reg_d = data[:,1]
    val_d = data[:,2]
    lr_d = np.expand_dims(lr_d,axis=1)
    reg_d = np.expand_dims(reg_d,axis=1)
    
    #postfix "sp" abbreviates for spaced
    #postfix "st" abbreviates for straightened
    #capital letters denote a matrix
    pt = 50
    # Prepare ranges for the meshgrid
    lr_sp = np.linspace(lr_d.min(), lr_d.max(), pt)
    reg_sp = np.linspace(reg_d.min(), reg_d.max(), pt+10)
    
    # Construct a meshgrid according to 
    #  http://matplotlib.org/examples/pylab_examples/contour_demo.html
    LR_sp, REG_sp = np.meshgrid(lr_sp,reg_sp)
    
    
    if len(results) < 8:
        # Apply a Least Square fitting to fit a 2D-polynomial
        A = np.concatenate((lr_d**2,reg_d**2,lr_d,reg_d,lr_d*reg_d,np.ones((N,1))),axis=1)
        p,_,_,_ = np.linalg.lstsq(A,val_d)

        # Apply the polynomial coefficients
        VAL_sp_2 = p[0]*LR_sp**2 + p[1]*REG_sp**2 + p[2]*LR_sp + p[3]*REG_sp + p[4]*LR_sp*REG_sp + p[5]*np.ones(LR_sp.shape)
    else:
        # Apply a Least Square fitting to fit a 2D-polynomial
        A = np.concatenate((lr_d**2,reg_d**2,(lr_d**2)*reg_d,(reg_d**2)*lr_d,lr_d,reg_d,lr_d*reg_d,np.ones((N,1))),axis=1)
        p,_,_,_ = np.linalg.lstsq(A,val_d)

        # Apply the polynomial coefficients
        VAL_sp_2 = p[0]*LR_sp**2 + p[1]*REG_sp**2 +p[2]*(LR_sp**2)*REG_sp + p[3]*(REG_sp**2)*LR_sp+ p[4]*LR_sp + p[5]*REG_sp + p[6]*LR_sp*REG_sp + p[7]*np.ones(LR_sp.shape)
    
    #Plot the figure
    plt.figure()
    CS = plt.contour(LR_sp, REG_sp, VAL_sp_2)
    plt.clabel(CS, inline=1, fontsize=10)
    sc = plt.scatter(lr_d,reg_d,c=val_d,cmap = plt.cm.OrRd)
    cbar = plt.colorbar(sc, orientation='vertical')
    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar

    
    plt.xlabel('Learning rate in log10 space')
    plt.ylabel('Regularization strength in log10 space')
    plt.show()
    
    
    


def hyperparam_plot(results):
    """
    Takes as an input the dict containing tuples 
    results[(lr,reg)] = (acc_train,acc_val)
    """
    data = np.zeros((len(results),3))
    i=0
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        data[i] = np.array([np.log10(lr), np.log10(reg), val_accuracy])
        i+=1
    
    lr_d = data[:,0]
    reg_d = data[:,1]
    val_d = data[:,2]
    
    
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    #%pylab inline
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    surf = ax.plot_trisurf(lr_d, reg_d, val_d, cmap=cm.jet)
    plt.xlabel('Learnin rate')
    plt.ylabel('regularization strength')
    #ax.set_zlim(-1.01, 1.01)

    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
