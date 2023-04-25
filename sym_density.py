#write a class with one method
# method should take a list of parameters and return a value

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy.vector import CoordSys3D

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

    def distance_metric(self):
        """
        Form the distance function V
        Outputs:
        dist            : the scalar distance metric for the robot
        """
        x, y = sp.symbols('x y')
        R = CoordSys3D('R')
        x_vec = x*R.i + y*R.j
        goal = self.goal[0]*R.i + self.goal[1]*R.j
        dist = sp.sqrt((x_vec-goal).dot(x_vec-goal))
        dist_fn = sp.lambdify([x,y]  , 1/(dist**(2*self.alpha)))
        return dist, dist_fn
    
    def eval_distance_fn(self, x_domain=np.linspace(-10,10,100), y_domain=np.linspace(-10,10,100)):
        """
        Evalulate the inverse bump function
        Inputs:
        x_domain        : the x domain to evaluate
        y_domain        : the y domain to evaluate
        Outputs:
        f_inverse_bump  : the value of inverse bump function at each point in the domain
        """
        f_distance = []
        dist, distance_fn = self.distance_metric()
        for x in x_domain:
            for y in y_domain:
                    f_distance.append(distance_fn(x,y))        
        return f_distance
    
    def inverse_bump(self):
        """
        Form the bump function as psi
        Outputs:
        ivnerse_bump    : the circular inverse bump function evaltuated at x 
        """
        x, y = sp.symbols('x y')
        R = CoordSys3D('R')
        x_vec = x*R.i + y*R.j
        obs = self.obs_center[0]*R.i + self.obs_center[1]*R.j
        obs_dist = sp.sqrt((x_vec-obs).dot(x_vec-obs))
        shape = (np.subtract(obs_dist**2, self.r1**2)) / np.subtract(self.r2**2, self.r1**2)
        
        f = sp.Piecewise((0,shape<=0), (sp.exp(-1/shape),shape>0))
        shape_shift = 1-shape
        f_shift = sp.Piecewise((0,shape_shift<=0), (sp.exp(-1/shape_shift),shape_shift>0))
        inverse_bump = f/(f+f_shift)
        inverse_bump_fn = sp.lambdify([x,y], inverse_bump)
        return inverse_bump, inverse_bump_fn
    
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
        inverse_bump, inverse_bump_fn = self.inverse_bump()
        for x in x_domain:
            for y in y_domain:
                    f_inverse_bump.append(inverse_bump_fn(x,y))        
        return f_inverse_bump
    
    def density(self):
        """
        Form the density function as rho= (1/V^2*alpha)*psi
        Inputs:
        x               : the (x,y) position of the robot
        Outputs:
        rho             : the density function evaltuated at x 
        """
        x, y = sp.symbols('x y')
        R = CoordSys3D('R')
        x_vec = x*R.i + y*R.j
        goal = self.goal[0]*R.i + self.goal[1]*R.j
        dist = sp.sqrt((x_vec-goal).dot(x_vec-goal))
        obs = self.obs_center[0]*R.i + self.obs_center[1]*R.j
        obs_dist = sp.sqrt((x_vec-obs).dot(x_vec-obs))
        shape = (np.subtract(obs_dist**2, self.r1**2)) / np.subtract(self.r2**2, self.r1**2)
        
        f = sp.Piecewise((0,shape<=0), (sp.exp(-1/shape),shape>0))
        shape_shift = 1-shape
        f_shift = sp.Piecewise((0,shape_shift<=0), (sp.exp(-1/shape_shift),shape_shift>0))
        inverse_bump = f/(f+f_shift)

        rho = 1/(dist**(2*self.alpha))*inverse_bump
        rho_fn = sp.lambdify([x,y], rho)

        return rho_fn
    
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
        rho_fn = self.density()
        for x in x_domain:
            for y in y_domain:
                if  self.density([x,y]) < 10:
                    f_density.append(rho_fn(x,y))
                else:
                    f_density.append(10)
        return f_density
    
    def grad_inverse_bump(self):
        """
        Form the gradient of the inverse bump function
        """
        x, y = sp.symbols('x y')
        inverse_bump, inverse_bump_fn = self.inverse_bump()
        grad_psi_x = sp.diff(inverse_bump, x)
        grad_psi_y = sp.diff(inverse_bump, y)
        grad_psi_fn_x = sp.lambdify([x,y], grad_psi_x)
        grad_psi_fn_y = sp.lambdify([x,y], grad_psi_y)
        return grad_psi_fn_x, grad_psi_fn_y
    
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
        #return grad_density_x, grad_density_y
    
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
    print("Running main function")
    plots = False
    density = Density()
    f_rho = density.density()
    print(f_rho(1,2))

    if(plots ==True):
        # surface plot of f_density
        X = np.linspace(-10,10,100)
        Y = np.linspace(-10,10,100)
        f_distance = density.eval_density(X,Y)

        fig = plt.figure()
        X, Y = np.meshgrid(X, Y)
        # subplot 1
        ax = fig.add_subplot(2,2,1, projection='3d')
        Z = np.array(f_distance).reshape(100,100)
        surf1 = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                            linewidth=0, antialiased=False)
        #ax.set_zlim(0, 10)
        ax.zaxis.set_major_locator(plt.LinearLocator(10))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
        fig.colorbar(surf1, shrink=0.5, aspect=5)
        plt.show()
    

if __name__=="__main__":
    main()
