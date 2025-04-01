import matplotlib.pyplot as plt
import numpy as np

imgR0 = plt.imread("robot_no_noise.jpg")
imgR = imgR0[:, :, 0] / 255

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(imgR)
axes[0].set_title("Image Originale")


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


axes[1].imshow(laplacien(imgR))
axes[1].set_title("Laplacien")

## Q5 Méthode de descente de gradient à pas fixe


def optim_gradient_fixed_step(img, L, max_iter=1000, epsilon_grad_fun=1e-8):
    nb_iter = 0
    x = img
    while nb_iter <= max_iter and np.linalg.norm(grad(x)) > epsilon_grad_fun:
        nb_iter += 1
        x -= L * (grad(x)[0] + grad(x)[1])
    return x


axes[2].imshow(optim_gradient_fixed_step(imgR, 1e-3))
axes[2].set_title("Image après optimisation par descente de gradient")


plt.show()
