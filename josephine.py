import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

imgR0 = plt.imread("robot_no_noise.jpg")
imgR = imgR0[:, :, 0] / 255
imgRB0 = plt.imread("robot_noise.jpg")
imgRB = imgRB0[:,:,0]/255

'''
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(imgR,cmap='gray')
axes[0].set_title("Image Originale")
axes[1].imshow(imgRB,cmap='gray')
axes[1].set_title("Image Bruitée")
'''


## Q4 grad, div et laplacien
def grad(u):
    n, m = u.shape[0], u.shape[1]

    dux = u[1:, :] - u[:-1, :]
    dux = np.concatenate((dux, np.zeros((1, m))), axis=0)
    duy = u[:, 1:] - u[:, :-1]
    duy = np.concatenate((duy, np.zeros((n, 1))), axis=1)

    return np.array([dux, duy])


def div(v):
    divx = v[0, 1:, :] - v[0, :-1, :]
    divx = np.concatenate((divx, np.array([-v[0, -1, :]])), axis=0)
    divy = v[1, :, 1:] - v[1, :, :-1]
    divy = np.concatenate((divy, np.array(-v[1, :, -1].reshape(-1, 1))), axis=1)
    return divx + divy


def laplacien(u):
    return div(grad(u))


'''
axes[1].imshow(laplacien(imgR),cmap='gray')
axes[1].set_title("Laplacien")
'''

## Q5 Méthode de descente de gradient à pas fixe


def optim_gradient_fixed_step(img, L, max_iter=1000, epsilon_grad_fun=1e-8):
    nb_iter = 0
    x = img
    while nb_iter <= max_iter and np.linalg.norm(grad(x)) > epsilon_grad_fun:
        nb_iter += 1
        x -= L * (grad(x)[0] + grad(x)[1])
    return x


imgRB_opti=optim_gradient_fixed_step(imgRB, 1e-5)
'''
axes[2].imshow(imgRB_opti,cmap='gray')
axes[2].set_title("Image après descente de gradient")

plt.show()
'''

## Q6 erreur RMSE

def RMSE(u,uVT):
    n,m=u.shape
    return np.linalg.norm(uVT-u)/(n*m)**0.5

print(f"RMSE de l'image trouvée par minimisation = {RMSE(imgRB_opti, imgR)}, et RMSE de l'image bruitée = {RMSE(imgRB, imgR)}")

## Q7 comparaison avec méthode scipy

def fun_img(img):
    pass

scipy_opti = optimize.minimize(fun_img,imgRB, method="BFGS").x

plt.imshow(scipy_opti)
plt.show()