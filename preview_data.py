import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, level in zip(axes, ['well', 'semi', 'heavy']):
    data = np.load(f'data/dataset_{level}.npz')
    X, y = data['X'], data['y']

    for cls, color in zip([0, 1], ['blue', 'red']):
        sns.kdeplot(
            x=X[y == cls, 0],
            ax=ax,
            color=color,
            fill=True,
            alpha=0.25,
            label=f'Clase {cls}'
        )

    ax.set_title(f'Distribución: {level}')
    ax.set_xlabel('Característica 1')
    ax.set_ylabel('Densidad')
    ax.legend()

plt.tight_layout()
plt.savefig('data/gaussian_distributionsV2.png', dpi=300)
plt.show()