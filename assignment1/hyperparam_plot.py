import numpy as np
import matplotlib.pyplot as plt

def poly_contour_plot(results):
    """
    Takes as an input the dict containing tuples 
    results[(lr,reg)] = (acc_train,acc_val)
    
    Functions fits a 2D-polynomial to the data in log10-space
    that is ax^2+by^2+cx+dy*exy+f
    
    Returns a contour plot for the hyperparameters
   
    """
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
    
    # Apply a Least Square fitting to fit a 2D-polynomial
    A = np.concatenate((lr_d**2,reg_d**2,lr_d,reg_d,lr_d*reg_d,np.ones((N,1))),axis=1)
    p,_,_,_ = np.linalg.lstsq(A,val_d)
    
    
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
    
    # Apply the polynomial coefficients
    
    VAL_sp_2 = p[0]*LR_sp**2 + p[1]*REG_sp**2 + p[2]*LR_sp + p[3]*REG_sp + p[4]*LR_sp*REG_sp + p[5]*np.ones(LR_sp.shape)
    
    #Plot the figure
    plt.figure()
    CS = plt.contour(LR_sp, REG_sp, VAL_sp_2)
    plt.clabel(CS, inline=1, fontsize=10)
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
