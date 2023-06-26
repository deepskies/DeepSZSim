def plot_img(image, z, opt = 0, path = None):
     '''
     Input: image, mode (option of 0.5/5 Mpc, default to 0.5), cmb (option of y/delta_T, default to y)
     Return: angular scale
     '''
     values = [0, 9, 18, 27, 36]
     x_label_list = [9, 4.5, 0, 4.5, 9]
     y_label_list = np.around(arcmin_to_Mpc(x_label_list, z), decimals = 2)
     if opt == 0 or opt == 3:
         option = 'hot'
         title = 'Y'
         cbar_label = r'$Y$'
     if opt == 1:
         option = 'ocean'
         title = 'T'
         cbar_label = r'$uK$'
     if opt == 2:
         option = 'viridis'
         title = 'Kernal'
         cbar_label = r' '

     fig, ax = plt.subplots(1,1)
     img = ax.imshow(image, cmap=option)
     ax.set_xticks(values)
     ax.set_xticklabels(x_label_list)
     ax.set_yticks(values)
     ax.set_yticklabels(y_label_list)
     cbar = fig.colorbar(img)
     cbar.ax.set_ylabel(cbar_label)
     plt.title(title)
     plt.xlabel('arcmin')
     plt.ylabel(r'Mpc')

     if opt == 3:
         circle_disk = plt.Circle((18, 18), radius_size(z,disk = True), color='green', fill=False, linewidth=2)
         circle_ring = plt.Circle((18, 18), radius_size(z,ring = True), color='black', fill=False, linewidth=2)
         ax.add_patch(circle_disk)
         ax.add_patch(circle_ring)

     plt.savefig(path)
     
def plot_y(r, y, z, path):
    '''
    Input: profile as function of radius
    Return: visulization (non-log & log scale)
    '''
    fig,ax = plt.subplots(1,2,figsize = (12,5))
    plt.subplots_adjust(wspace = 0.3)
    ax[0].plot(r, y, color = "red", label = "non-log")
    ax[0].set_xlabel("Mpc")
    ax[0].set_ylabel(r'Mpc$^{-1}$')
    ax[0].title.set_text("Y z="+str(z))
    ax[1].loglog(r, y, color = "blue", label = "log")
    ax[1].set_xlabel("Mpc")
    ax[1].set_ylabel(r'Mpc$^{-1}$')
    ax[1].title.set_text("Y(Log) z="+str(z))
    plt.savefig(path)


      #code to demonstrate the final result
def plot_pdf(xr, title, func=None, args=None, label='ITS method', 
            ylog=True,
            bins = 50, xlims=[], ylims=[], figsize=3):
    """
    by Andrey Kravtsov
    wrapper convenience function for plotting  histogram of a sequence of floats xr
    and comparing it to a theoretical distribution given by func
    
    Parameters:
    -----------
    xr: 1d numpy array of floats, values in the sequence to plot
    func: Python function object handle
    args: list of arguments to the function
    other parameters are used for plotting
    """
    plt.figure(figsize=(figsize,figsize))
    if ylog: plt.yscale('log') # plot y-values on a logarithmic scale
    if ylims: plt.ylim(ylims) # set axis limits 
    if xlims: plt.xlim(xlims)
    # compute histogram values 
    hist, bins, patches = plt.hist(xr, density=True, bins=bins, label=label)
    binc = 0.5*(bins[1:] + bins[:-1]) # compute bin centers
    plt.plot(binc, func(binc), lw=5, c='orangered', label='target pdf')
    plt.title(title, fontsize=3*figsize)
    plt.ylabel(r'$likelihood$') # label axis 
    plt.xlabel(r'$mass$')
    plt.legend(loc='best', frameon=False, fontsize=3*figsize)
    plt.show()
