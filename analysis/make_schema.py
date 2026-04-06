import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 9), facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor='white')

grid = np.linspace(-5, 5, 250)
X, Y = np.meshgrid(grid, grid)
Z = np.zeros_like(X)

Z += 0.3 * np.sin(X * 0.3) * np.cos(Y * 0.4) + 0.1

# Basin 1: Deep common class basin
Z -= 3.0 * np.exp(-((X - 1.5)**2 + (Y + 0.5)**2) / (2.0**2))
# Basin 2: Medium common class basin
Z -= 2.2 * np.exp(-((X + 2.0)**2 + (Y - 1.5)**2) / (1.5**2))
# Basin 3: Shallow rare class basin
Z -= 1.0 * np.exp(-((X + 0.5)**2 + (Y + 2.5)**2) / (0.8**2))

# Ridges
Z += 0.8 * np.exp(-(X**2) / 0.5) * np.exp(-((Y - 0.5)**2) / 3.0)
Z += 0.5 * np.exp(-((Y + 0.5)**2) / 0.4) * np.exp(-((X + 1.0)**2) / 2.0)
Z += 0.02 * (X**2 + Y**2)

ax.plot_surface(X, Y, Z, cmap='bone', alpha=0.6, rstride=5, cstride=5,
                edgecolor='gray', linewidth=0.15, antialiased=True)

def get_z(px, py):
    z = 0.3 * np.sin(px * 0.3) * np.cos(py * 0.4) + 0.1
    z -= 3.0 * np.exp(-((px - 1.5)**2 + (py + 0.5)**2) / 4.0)
    z -= 2.2 * np.exp(-((px + 2.0)**2 + (py - 1.5)**2) / 2.25)
    z -= 1.0 * np.exp(-((px + 0.5)**2 + (py + 2.5)**2) / 0.64)
    z += 0.8 * np.exp(-(px**2) / 0.5) * np.exp(-((py - 0.5)**2) / 3.0)
    z += 0.5 * np.exp(-((py + 0.5)**2) / 0.4) * np.exp(-((px + 1.0)**2) / 2.0)
    z += 0.02 * (px**2 + py**2)
    return z


ax.view_init(elev=30, azim=-40)

ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white'); ax.yaxis.pane.set_edgecolor('white'); ax.zaxis.pane.set_edgecolor('white')
ax.grid(False)


fig.savefig('figures_v5es/fig_attractor_schema.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved figures_v5es/fig_attractor_schema.png')
