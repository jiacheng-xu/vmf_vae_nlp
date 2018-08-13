from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import torch
from random import gauss, uniform


def make_rand_vector(dims):
    vec = [np.random.normal(0, 0.12) for i in range(dims)]

    return vec


def rand_vec(dim, z):
    vec = [uniform(0, 1) for i in range(dim - 1)]
    mag = 1 - z ** 2
    y = [x / mag for x in vec] + [z]
    z = sum([x ** 2 for x in y])
    print(z)


def draw_ball(z=np.linspace(-1, 1, 50)):
    bag = []
    for x in z:
        r = np.sqrt(1 - x ** 2)
        theta = np.linspace(-180, 180, 20)
        for t in theta:
            ys = r * np.sin(np.pi * t / 180.)
            xs = r * np.cos(np.pi * t / 180.)
            bag.append([x, xs, ys])
    return bag


def drawSphere(xCenter, yCenter, zCenter, r):
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r * x + xCenter
    y = r * y + yCenter
    z = r * z + zCenter
    return (x, y, z)


bag = []
# bag.append(draw_ball())
from NVLL.util.util import GVar

for n in range(5):
    x = make_rand_vector(3)
    x = np.asarray(x)
    tmp = []
    for _ in range(20):
        y = np.random.normal(x, 0.1)
        print(y)
        tmp.append(y)
    bag.append(tmp)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

(xs, ys, zs) = drawSphere(0, 0, 0, 1)
ax.plot_wireframe(xs, ys, zs, color="black", linestyle=":", lw=0.75)
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


a = Arrow3D([0, 0], [0, 0],
            [-1.2, 1.2], mutation_scale=7,
            lw=1, arrowstyle="-|>", color="black")
b = Arrow3D([-1.2, 1.2], [0, 0], [0, 0],
            mutation_scale=7,
            lw=1, arrowstyle="-|>", color="black")
c = Arrow3D([0, 0], [-1.2, 1.2], [0, 0],
            mutation_scale=7,
            lw=1, arrowstyle="-|>", color="black")
ax.add_artist(a)
ax.add_artist(b)
ax.add_artist(c)

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
bank_color = ['b', 'r', 'g', 'c', 'm']
bank_marker = ['.', 'o', 'v', '^', '<']
for idx, group in enumerate(bag):
    c = bank_color[idx]
    m = '.'
    mean_x = 0
    mean_y = 0
    mean_z = 0
    for point in group:
        xs, ys, zs = point
        mean_x += xs

        mean_y += ys
        mean_z += zs
        ax.scatter(xs, ys, zs, c=c, marker=m)
    mean_x /= len(group)
    mean_y /= len(group)
    mean_z /= len(group)
    line = Arrow3D([0, mean_x], [0, mean_y], [0, mean_z],
                   mutation_scale=10,
                   lw=1.5, arrowstyle="-|>", color=c)

    ax.add_artist(line)

# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)
# ax.set_axis_on()
ax.set_axis_off()

start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(-1, 1.5, 0.5))
ax.yaxis.set_ticks(np.arange(-1, 1.5, 0.5))
ax.zaxis.set_ticks(np.arange(-1, 1.5, 0.5))

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# fig.set_size_inches(5, 5)
fig.savefig('gauss.pdf', transparent=True)
plt.show()
