import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, rho in zip(axes, [0, 0.5, 0.8, 0.99]):
    cov = np.array([
        [1.0, rho],
        [rho, 1.0]
    ])

    # Dos clases con diferentes medias
    X1 = rng.multivariate_normal([0, 0], cov, 500)
    X2 = rng.multivariate_normal([2, 2], cov, 500)

    # Graficar cada clase con un color
    ax.scatter(
        X1[:, 0], X1[:, 1],
        color='royalblue', alpha=0.4, s=10,
        label='Clase 0'
    )

    ax.scatter(
        X2[:, 0], X2[:, 1],
        color='tomato', alpha=0.4, s=10,
        label='Clase 1'
    )

    ax.set_title(f'Covarianza = {rho}')
    ax.set_xlim(-4, 7)
    ax.set_ylim(-4, 7)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.legend()

plt.tight_layout()
plt.show()