import matplotlib.pyplot as plt
import numpy as np

imgR0 = plt.imread("robot_no_noise.jpg")
imgR = imgR0[:, :, 0] / 255

# plt.imshow(imgR)


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

plt.imshow(laplacien(imgR))
plt.show()