#write a class with one method
# method should take a list of parameters and return a value

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Density:
    def __init__(self, r1=1, r2=2, obs_center=[0,0], goal=[5,5], alpha=0.2):
        """
        Inputs: 
        r1              : radius of the obstacle
        r2              : sensing radius for the obstalce
        obs_center      : list of (x,y) position the center of the obstacle
        goal            : (x,y) position of the goal
        alpha           : tuning parameter for the density function
        """
        self.r1 = r1
        self.r2 = r2
        self.alpha = alpha
        self.obs_center = obs_center
        self.goal = goal

    def distance_metric(self,x):
        """
        Form the distance function V
        Inputs:
        x               : the (x,y) position of the robot
        Outputs:
        dist            : the scalar distance metric for the robot
        """
        dist = np.linalg.norm(np.subtract(x, self.goal))
        return dist
    
    def inverse_bump(self,x):
        """
        Form the bump function as psi
        Inputs:
        x               : the (x,y) position of the robot
        Outputs:
        ivnerse_bump    : the circular inverse bump function evaltuated at x 
        """

        # form circluar shape for the obstalce and with linearly scaled sensing
        obs_dist = np.linalg.norm(np.subtract(x , self.obs_center))
        shape = (np.subtract(obs_dist**2, self.r1**2)) / np.subtract(self.r2**2, self.r1**2)

        # form inverse bump function for that shape
        f = np.piecewise(shape, [shape<=0, shape>0], [0, np.exp(-1/shape)])
        shape_shift = 1-shape
        f_shift = np.piecewise(shape_shift, [shape_shift<=0, shape_shift>0], [0, np.exp(-1/(shape_shift))])
        inverse_bump = f/(f+f_shift)

        return inverse_bump
    
    def eval_inverse_bump(self, x_domain=np.linspace(-10,10,100), y_domain=np.linspace(-10,10,100)):
        """
        Evalulate the inverse bump function
        Inputs:
        x_domain        : the x domain to evaluate
        y_domain        : the y domain to evaluate
        Outputs:
        f_inverse_bump  : the value of inverse bump function at each point in the domain
        """
        f_inverse_bump = []
        for x in x_domain:
            for y in y_domain:
                    f_inverse_bump.append(self.inverse_bump([x,y]))        
        return f_inverse_bump
    
    def density(self, x):
        """
        Form the density function as rho= (1/V^2*alpha)*psi
        Inputs:
        x               : the (x,y) position of the robot
        Outputs:
        rho             : the density function evaltuated at x 
        """
        V = self.distance_metric(x)
        psi = self.inverse_bump(x)
        rho = (1/(V**(2*self.alpha)))*psi
        return rho
    
    def eval_density(self, x_domain=np.linspace(-10,10,100), y_domain=np.linspace(-10,10,100)):
        """
        Evalulate the density function
        Inputs:
        x_domain        : the x domain to evaluate
        y_domain        : the y domain to evaluate
        Outputs:
        f_density       : the value of density function at each point in the domain
        """
        f_density = []
        for x in x_domain:
            for y in y_domain:
                if  self.density([x,y]) < 10:
                    f_density.append(self.density([x,y]))
                else:
                    f_density.append(10)
        return f_density
    
    def grad_inverse_bump(self, x):
        """
        Form the gradient of the inverse bump function
        """
        obs_dist = np.linalg.norm(np.subtract(x , self.obs_center))

        # set gradient inside the obstacle to be zero
        if np.subtract(obs_dist**2, self.r1**2) <= 0:
            grad_psi_x = 0
            grad_psi_y = 0

        elif np.subtract(obs_dist**2, self.r2**2) >= 0:
            grad_psi_x = 0
            grad_psi_y = 0

        # calculate gradient outside the obstacle
        else:
            psi0 = np.exp(1/np.subtract(obs_dist**2, self.r1**2)) /\
                (np.exp(1/np.subtract(obs_dist**2, self.r1**2)) + np.exp(1/np.subtract(obs_dist**2, self.r2**2)))
            psi1 = np.exp(1/np.subtract(obs_dist**2, self.r2**2)) /\
                (np.exp(1/np.subtract(obs_dist**2, self.r2**2)) + np.exp(1/np.subtract(obs_dist**2, self.r1**2)))
            term1 = (np.subtract(psi0,1))/(np.subtract(obs_dist**2, self.r1**2)**2)
            term2 = psi1/(np.subtract(obs_dist**2, self.r2**2)**2)

            grad_psi_x = 2*(x[0]-self.obs_center[0])*psi0*np.add(term1,term2)
            grad_psi_y = 2*(x[1]-self.obs_center[1])*psi0*np.add(term1,term2)

        return grad_psi_x, grad_psi_y
    
    def eval_grad_inverse_bump(self, x_domain=np.linspace(-10,10,100), y_domain=np.linspace(-10,10,100)):
        """
        Evalulate the density function
        Inputs:
        x_domain        : the x domain to evaluate
        y_domain        : the y domain to evaluate
        Outputs:
        f_grad_density  : the value of gradient of density function at each point in the domain
        """
        f_grad_psi_x = []
        f_grad_psi_y = []
        for x in x_domain:
            for y in y_domain:
                    grad_psi_x, grad_psi_y = self.grad_inverse_bump([x,y])
                    f_grad_psi_x.append(grad_psi_x)
                    f_grad_psi_y.append(grad_psi_y)
        return f_grad_psi_x, f_grad_psi_y
    
    def grad_density(self, x):
        """
        Form the gradient of the density function
        """
        obs_dist = np.linalg.norm(np.subtract(x , self.obs_center))

        # set gradient inside the obstacle to be zero
        if np.subtract(obs_dist**2, self.r1**2) <= 0:
            print('zero gradient')
            grad_rho_x = 0
            grad_rho_y = 0

        # calculate gradient outside the obstacle
        else:
            grad_psi_x, grad_psi_y = self.grad_inverse_bump(x)
            V = self.distance_metric(x)
            grad_rho_x = (1/(V**(2*self.alpha)))*grad_psi_x - (2*x[0]/(V**(2*self.alpha+1)))*self.inverse_bump(x)
            grad_rho_y = (1/(V**(2*self.alpha)))*grad_psi_y - (2*x[1]/(V**(2*self.alpha+1)))*self.inverse_bump(x)

        return grad_rho_x, grad_rho_y
    
    def eval_grad_density(self, x_domain=np.linspace(-10,10,100), y_domain=np.linspace(-10,10,100)):
        """
        Evalulate the density function
        Inputs:
        x_domain        : the x domain to evaluate
        y_domain        : the y domain to evaluate
        Outputs:
        f_grad_density  : the value of gradient of density function at each point in the domain
        """
        f_grad_density_x = []
        f_grad_density_y = []
        for x in x_domain:
            for y in y_domain:
                    grad_density_x, grad_density_y = self.grad_density([x,y])
                    f_grad_density_x.append(grad_density_x)
                    f_grad_density_y.append(grad_density_y)
        return f_grad_density_x, f_grad_density_y
    

########### utility functions ###########################################################
def symlog(x):
    """ Returns the symmetric log10 value """
    return np.sign(x) * np.log10(np.abs(x))


###################### main function ####################################################
def main():
    plots = True
    density = Density()    

    if(plots ==True):
        # surface plot of f_density
        X = np.linspace(-2,2,100)
        Y = np.linspace(-2,2,100)
        f_inverse_bump = density.eval_inverse_bump(X,Y)
        f_density = density.eval_density(X,Y) 
        #u,v = density.eval_grad_density(X,Y) 
        u,v = density.eval_grad_inverse_bump(X,Y)
        u = np.array(u).reshape(100,100)
        v = np.array(v).reshape(100,100)

        fig = plt.figure()
        X, Y = np.meshgrid(X, Y)
        # subplot 1
        ax = fig.add_subplot(2,2,1, projection='3d')
        Z = np.array(f_inverse_bump).reshape(100,100)
        surf1 = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax.set_zlim(0, 1)
        ax.zaxis.set_major_locator(plt.LinearLocator(10))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
        fig.colorbar(surf1, shrink=0.5, aspect=5)

        # subplot 2
        ax = fig.add_subplot(2,2,2, projection='3d')
        Z = np.array(f_density).reshape(100,100)
        #print(Z)
        surf2 = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax.set_zlim(0, 1)
        ax.zaxis.set_major_locator(plt.LinearLocator(10))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
        fig.colorbar(surf2, shrink=0.5, aspect=5)

        # subplot 3
        ax = fig.add_subplot(2,2,3)
        ax.set_aspect('equal', adjustable='box')
        #quiver_plot = ax.streamplot(X, Y, np.array(u).reshape(100,100), np.array(v).reshape(100,100), density=1, color='k')
        #quiver_plot = ax.quiver(X, Y, symlog(u), symlog(v))
        quiver_plot = ax.quiver(X, Y, u, v, scale=1)
        plt.show()
    

if __name__=="__main__":
    main()
