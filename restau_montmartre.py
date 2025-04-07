import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

imgR0 = plt.imread("montmartre_no_noise.jpg")
imgR = imgR0[:, :, 0] / 255

imgRB = np.random.normal(loc=imgR, scale=0.2)

fig, axes = plt.subplots(1, 6, figsize=(15, 5))
axes[0].imshow(imgR, cmap="gray")
axes[0].set_title("Image Originale")
axes[1].imshow(imgRB, cmap="gray")
axes[1].set_title("Image Bruitée")

## Q4 grad, div et laplacien


def grad(u):
    dux = np.zeros(u.shape)
    duy = np.zeros(u.shape)

    dux[:-1, :] = u[1:, :] - u[:-1, :]
    duy[:, :-1] = u[:, 1:] - u[:, :-1]

    return np.array([dux, duy])


def div(v):
    n, m = v.shape[1], v.shape[2]
    divx = np.zeros((n, m))
    divy = np.zeros((n, m))

    divx[1:, :] = v[0, 1:, :] - v[0, :-1, :]
    divx[0, :] = v[0, 0, :]
    divx[-1, :] = -v[0, -2, :]

    divy[:, 1:] = v[1, :, 1:] - v[1, :, :-1]
    divy[:, 0] = v[1, :, 0]
    divy[:, -1] = -v[1, :, -2]

    return divx + divy


def laplacien(u):
    return div(grad(u))


"""
axes[1].imshow(laplacien(imgR),cmap='gray')
axes[1].set_title("Laplacien")
"""

## Q5 Méthode de descente de gradient à pas fixe


def funR(u, ub):  # ne sert pas
    g = grad(u)
    return (
        (np.linalg.norm(u - ub, ord=2) ** 2) / 2
        + np.linalg.norm(g[0], ord=2) ** 2
        + np.linalg.norm(g[1], ord=2) ** 2
    )


def grad_funR(u, ub):
    return u - ub - 2 * laplacien(u)


"""
# test pour vérifier les implémentations des fonctions précédentes

def approx_grad_funR(x, xb, eps, h):
    return (funR(x + eps * h, xb) - funR(x - eps * h, xb)) / (2 * eps)

u = np.random.random(imgRB.shape)
eps = 10 ** (-5)
h = np.random.random(imgRB.shape)
h = h/np.linalg.norm(h)
print(approx_grad_funR(u, imgRB, eps, h) - np.dot(np.transpose(grad_funR(u, imgRB)).ravel(), h.ravel()))
"""


def optim_gradient_fixed_step(grad_fun, x0, L, xb, max_iter=100, epsilon_grad_fun=1e-8):
    nb_iter = 0
    x = x0
    while nb_iter <= max_iter and np.linalg.norm(grad_fun(x, xb)) > epsilon_grad_fun:
        nb_iter += 1
        x -= L * grad_fun(x, xb)
    return x


x0, pas, m_iter = np.zeros(imgR.shape), 1e-2, 100
imgRB_opti = optim_gradient_fixed_step(grad_funR, x0, pas, imgRB, m_iter)

axes[2].imshow(imgRB_opti, cmap="gray")
axes[2].set_title("Après descente de gradient")


## Q6 erreur RMSE


def RMSE(u, uVT):
    n, m = u.shape
    return np.linalg.norm(uVT - u) / (n * m) ** 0.5


print(
    f"RMSE de l'image trouvée par minimisation = {RMSE(imgRB_opti, imgR)}, et RMSE de l'image bruitée = {RMSE(imgRB, imgR)}"
)

## Q7 comparaison avec méthode scipy


def funR_vect(u):
    v = u.reshape(imgRB.shape)
    g = grad(v)
    return (
        (np.linalg.norm(v - imgRB) ** 2) / 2
        + np.linalg.norm(g[0])
        + np.linalg.norm(g[1])
    )


def grad_funR_vect(u):
    v = u.reshape(imgRB.shape)
    g = v - imgRB - 2 * laplacien(v)
    return g.flatten()


def scipy_opti(fun, grad_fun, x0):
    sol = optimize.minimize(
        fun, x0.flatten(), jac=grad_fun, method="CG"
    ).x  # ou L-BFGS-B
    return sol.reshape(x0.shape)


axes[3].imshow(scipy_opti(funR_vect, grad_funR_vect, x0), cmap="gray")
axes[3].set_title("Après min par scipy")

# Q10 descente de gradient à pas fixe de TV-L2


def TVL2(u, ub):  # ne sert pas
    g = grad(u)
    return (
        (np.linalg.norm(u - ub, ord=2) ** 2) / 2
        + np.linalg.norm(g[0], ord=1)
        + np.linalg.norm(g[1], ord=1)
    )


def grad_TVL2(u, ub):
    return u - ub - div(np.sign(grad(u)))


x0, pas, m_iter = np.zeros(imgR.shape), 1e-2, 100
imgRB_opti_TVL2 = optim_gradient_fixed_step(grad_TVL2, x0, pas, imgRB, m_iter)

axes[4].imshow(imgRB_opti_TVL2, cmap="gray")
axes[4].set_title("Après descente de grad de TV-L2")

print(
    f"RMSE de l'image trouvée par minimisation de TV-L2 = {RMSE(imgRB_opti_TVL2, imgR)}, et RMSE de l'image bruitée = {RMSE(imgRB, imgR)}"
)

"""
def approx_grad_TVL2(x, xb, eps, h):
    return (TVL2(x + eps * h, xb) - TVL2(x - eps * h, xb)) / (2 * eps)


u = np.random.random(imgRB.shape)
eps = 10 ** (-5)
h = np.random.random(imgRB.shape)
h = h / np.linalg.norm(h)
print(
    approx_grad_TVL2(u, imgRB, eps, h)
    - np.dot(np.transpose(grad_TVL2(u, imgRB)).ravel(), h.ravel())
)"""

# Q11 descente avec momentum


beta = 0.9


def optim_gradient_momentum(grad_fun, x0, L, xb, max_iter=100, epsilon_grad_fun=1e-8):
    global beta
    nb_iter = 0
    x = x0
    p = -grad_fun(x, xb)
    while nb_iter <= max_iter and np.linalg.norm(grad_fun(x, xb)) > epsilon_grad_fun:
        nb_iter += 1
        p = beta * p + (1 - beta) * (-grad_fun(x, xb))
        x += L * p
    return x


x0, pas, m_iter = np.zeros(imgR.shape), 1e-2, 100
imgRB_opti_TVL2_mom = optim_gradient_momentum(grad_TVL2, x0, pas, imgRB, m_iter)

axes[5].imshow(imgRB_opti_TVL2_mom, cmap="gray")
axes[5].set_title("Après descente de gradient avec momentum de TV-L2")

plt.show()
