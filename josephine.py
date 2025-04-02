import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

imgR0 = plt.imread("robot_no_noise.jpg")
imgR = imgR0[:, :, 0] / 255
imgRB0 = plt.imread("robot_noise.jpg")
imgRB = imgRB0[:, :, 0] / 255

'''
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
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


"""
axes[1].imshow(laplacien(imgR),cmap='gray')
axes[1].set_title("Laplacien")
"""

## Q5 Méthode de descente de gradient à pas fixe


def funR(u, ub):
    g = grad(u)
    return (np.linalg.norm(u - ub) ** 2)/2 + np.linalg.norm((g[0] ** 2 + g[1] ** 2)**0.5)


def grad_funR(u, ub):
    return u - ub - 2 * laplacien(u)


def approx_grad_funR(x, xb, eps, h):
    return (funR(x + eps * h, xb) - funR(x - eps * h, xb)) / (2 * eps)

u = np.random.random(imgRB.shape)
eps = 10 ** (-5)
h = np.ones(imgRB.shape)
print(approx_grad_funR(u,imgRB,eps,h)-np.dot(np.transpose(grad_funR(u, imgRB)),h))

def optim_gradient_fixed_step(grad_fun, x0, L, xb, max_iter=100, epsilon_grad_fun=1e-8):
    nb_iter = 0
    x = x0
    while nb_iter <= max_iter and np.linalg.norm(grad_fun(x,xb)) > epsilon_grad_fun:
        nb_iter += 1
        x -= L * grad_fun(x,xb)
    return x


x0, pas = np.zeros(imgR.shape), 1e-3
imgRB_opti = optim_gradient_fixed_step(grad_funR, x0, pas, imgRB)
'''
axes[2].imshow(imgRB_opti,cmap='gray')
axes[2].set_title("Image après descente de gradient")
'''

## Q6 erreur RMSE


def RMSE(u, uVT):
    n, m = u.shape
    return np.linalg.norm(uVT - u) / (n * m) ** 0.5


print(
    f"RMSE de l'image trouvée par minimisation = {RMSE(imgRB_opti, imgR)}, et RMSE de l'image bruitée = {RMSE(imgRB, imgR)}"
)

## Q7 comparaison avec méthode scipy


def scipy_opti(fun, grad_fun, x0):
    sol = optimize.minimize(fun, x0.flatten(), jac = grad_fun, method="BFGS").x
    return sol.reshape(x0.shape)

'''
axes[3].imshow(scipy_opti(lambda u : funR(u,imgRB), lambda u : grad_funR(u,imgRB), x0), cmap='gray')
axes[3].set_title("Image après min par scipy")
plt.show()
'''