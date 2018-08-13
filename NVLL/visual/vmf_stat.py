from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np

# Load and format data

kl_z = [[24.060544967651367, 47.265777587890625, 63.639678955078125, 76.09759521484375],
        [18.745616912841797, 48.122337341308594, 73.41583251953125, 94.29776763916016],
        [14.36080265045166, 43.08722686767578, 72.1833267211914, 98.24507141113281],
        [11.424114227294922, 37.56840896606445, 67.44066619873047, 96.24412536621094]]
kl_z = np.asarray(kl_z)
nrows, ncols = kl_z.shape

x = np.linspace(100, 400, ncols)
y = np.linspace(100, 400, nrows)
x, y = np.meshgrid(x, y)

# region = np.s_[5:50, 5:50]
# x, y, z = x[region], y[region], z[region]

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(kl_z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, kl_z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

plt.show()
