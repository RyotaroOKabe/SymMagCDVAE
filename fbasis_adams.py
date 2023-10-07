#%%
"""
Algorithm 1: construct orbit graph
referenncce: 
https://www.notion.so/Representing-and-Learning-Functions-Invariant-Under-Crystallographic-Groups-9c60b812ea0f49c68902ea75db076b27?pvs=4
https://www.notion.so/231002-crystal-representation-d2f8fffd5bfd489c8eef8bd54847dcbf?pvs=4#e6fc79d1bdfb4d4faf83ca3190eb02cf
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from pymatgen.symmetry.analyzer import *
import scipy as sp
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))   # I downloaded MP structures
mpids = sorted(list(mpdata.keys()))
#%%
# parameter
mpid = 'mp-149' # get A_Pi from cubic silicon
epsilon = 0.4  # Minimum distance between points in the epsilon-net
uniform_grid=True # initial e-net is uniform grid
N_grid = 5  # Num of unit form grid for x,y,z directions (uniform_grid=True)
N_pts = 100 # Num of vertices in initialized e-net  (uniform_grid=False)
delta = 0.4 # cut off radius for orbit graph
threshold = 1e-5
#%%

def generate_epsilon_net(points, epsilon):
    """
    Generate an epsilon-net in 3D within a unit cube [0, 1] x [0, 1] x [0, 1].
    N: Number of points in the cube.
    epsilon: Minimum distance between points in the epsilon-net.
    """
    epsilon_net = []
    
    # Generate random points in the unit cube
    # points = np.random.rand(N, 3)@lattice
    
    for p in points:
        is_valid = False
        
        # Check if the point is at least epsilon distance away from all previously selected points
        for q in points:
            if not np.allclose(p, q, atol=threshold):
                if np.linalg.norm(p - q) < epsilon:
                    is_valid = True
                    break
        # If the point is valid, add it to the epsilon-net
        if is_valid:
            epsilon_net.append(p)
    
    return np.array(epsilon_net)

def plot_epsilon_net(epsilon_net):
    """
    Plot the epsilon-net in 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(epsilon_net[:, 0], epsilon_net[:, 1], epsilon_net[:, 2], s=20, c='b', marker='o')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Epsilon-Net in 3D: {len(epsilon_net)} points')
    
    plt.show()


def distance_matrix(epsilon_net):
    """
    Calculate the distance matrix for points in the epsilon-net.
    """
    num_points = len(epsilon_net)
    # dist_matrix = np.zeros((num_points, num_points))
    # for i in range(num_points):
    #     for j in range(num_points):
    #         dist_matrix[i, j] = np.linalg.norm(epsilon_net[i] - epsilon_net[j])
    dist_matrix = sp.spatial.distance.cdist(epsilon_net, epsilon_net)
    
    return dist_matrix

#%%
# [Algorithm 1]  orbit graph (all fractional coordinates)
# 0. data preparation (structure, space group operations)
pstruct = mpdata[mpid]  # pymatgen structure
sga = SpacegroupAnalyzer(pstruct)
opes = sga.get_space_group_operations()
sg = [sga.get_space_group_number(), sga.get_space_group_symbol()]
print(mpid, sg, len(opes))
frac, cart = pstruct.frac_coords, pstruct.cart_coords   # cart == frac@lattice
lattice = pstruct.lattice.matrix


#%%
# 1-1. e-net
# initialize e-net
if uniform_grid: 
    # Create a grid of values between 0 and 1 (you can adjust the range if needed)
    x = np.linspace(0.05, 0.95, N_grid)
    y = np.linspace(0.05, 0.95, N_grid)
    z = np.linspace(0.05, 0.95, N_grid)
    # Generate a mesh grid
    xv, yv, zv = np.meshgrid(x, y, z)
    # Reshape and stack to get the desired output shape
    points = np.stack((xv.ravel(), yv.ravel(), zv.ravel()), axis=-1)

else: 
    points = np.random.rand(N_pts, 3)

epsilon_net = generate_epsilon_net(points, epsilon)
plot_epsilon_net(epsilon_net)
print(len(epsilon_net))
dist_matrix = distance_matrix(epsilon_net)
plt.imshow(dist_matrix<epsilon)

# 1-2. local group elements A_Pi (Algorithm2)
# we use opes

# 1-3. get d_G distances
xs0, ys0 = epsilon_net.copy(), epsilon_net.copy()
neps = len(xs0) # the number of vertices in e-net
nops = len(opes)    # the number of operations
ys1s = []
for ope in opes:
    opr = ope.rotation_matrix
    opt = ope.translation_vector
    ys1 = (ys0@opr.T + opt)#%1
    ys1s.append(ys1)
ys1s = np.stack(ys1s)
dmatrix = []
idx_mins = []
for i, x0 in enumerate(xs0):
    d_x_ys1s = np.linalg.norm(x0 - ys1s, axis=-1)
    idx_min = np.argmin(d_x_ys1s, axis=0)  # idx_min.shape: len(ys1s)
    idx_mins.append(idx_min)
    dmatrix.append(d_x_ys1s[idx_min, range(len(ys1))])
idx_mins = np.stack(idx_mins)
dmatrix = np.stack(dmatrix)
plt.imshow(dmatrix)
plt.title('Edge (x,y) length')
plt.show()
plt.close()

# 1-4. add edges to x,y in e-net \Gammma
mask = dmatrix<delta
plt.imshow(mask)
plt.title('Edge (x,y) length < delta')
plt.show()
plt.close()
idx_mx, idx_my = np.nonzero(mask)

# plot orbit graph (cartesian coordinate)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs0[:, 0], xs0[:, 1], xs0[:, 2], s=20, c='b', marker='o')
for ix, iy in zip(idx_mx, idx_my):
    im = idx_mins[ix, iy]
    y1 = ys1s[im, iy]
    x0 = xs0[ix]
    ax.scatter(y1[0], y1[1], y1[2], s=20, c='r', marker='o')
    ax.plot([x0[0], y1[0]], [x0[1], y1[1]], [x0[2], y1[2]], color='k')
    print(ix, iy)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Orbit Graph in 3D: {len(epsilon_net)} points')
plt.show()

#%%
#  [Algorithm 4]  Fourier basis
# 4.2 normalized laplacian L
W = np.exp(-dmatrix/(2*epsilon**2))
W[mask==0]=0
D = np.diag(np.sum(W, axis=-1))
L = np.diag([1 for _ in range(len(D))]) +  - np.linalg.inv(D)@W
# 4.3 eigenvalues, eigenvectors of L
e_val, e_vec = np.linalg.eig(L)    # e_val.shape, e_vec.shape = ((91,), (91, 91))
fig, axs = plt.subplots(1,3, figsize=(20, 6))
ax0 = axs[0]
ax0.plot(e_val)
ax0.set_title('eigenvalues')
ax1 = axs[1]
ax1.imshow(e_vec.real)
ax1.set_title('eigenvectors (real)')
ax2 = axs[2]
ax2.imshow(e_vec.imag)
ax2.set_title('eigenvectors (imag)')


#%% 
# interpolation (use 4D data)
e_i = e_vec*2/epsilon   # p.16 of the paper
lmd_i = e_val*2/epsilon
arg = (e_i * np.sqrt(lmd_i)).real
ematrix = np.zeros_like(arg)
ematrix[:, ::2] = np.sin(arg[:, ::2])   # Apply sine to odd columns
ematrix[:, 1::2] = np.cos(arg[:, 1::2]) # Apply cosine to even columns
plt.imshow(ematrix)
plt.title(f'$e_i$ (row, vertices x, col: index i)')
plt.show()

#%%
if uniform_grid:    # Get slice at z=0.5 if uniform grid
    z_slice = 0.5
    smask = np.abs(epsilon_net[:, 2]-z_slice)<threshold
    slice_net1 = epsilon_net[smask]
    sl_e_i_x = e_i[smask, :] # sliced e_i(x)
    plt.scatter(slice_net1[:, 0], slice_net1[:, 1])
    plt.title(f'vertices (z={z_slice})')
    plt.show()
    # Apply sine to odd columns
    arg = (sl_e_i_x * np.sqrt(lmd_i)).real
    ematrix = np.zeros_like(arg)
    ematrix[:, ::2] = np.sin(arg[:, ::2])   # Apply sine to odd columns
    ematrix[:, 1::2] = np.cos(arg[:, 1::2]) # Apply cosine to even columns
    plt.imshow(ematrix)
    plt.title(f'$e_i$ (z={z_slice})')
    plt.show()

    for idx in range(len(lmd_i))[:10]:
        x = slice_net1[:, 0]
        y = slice_net1[:, 1]
        z = ematrix[:, idx]

        # Create a uniform grid to interpolate to
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        # Create a new figure with 1 row and 2 columns of subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # The 3D surface plot
        ax1 = fig.add_subplot(121, projection='3d', sharex=axes[1], sharey=axes[1])
        surf = ax1.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')
        fig.colorbar(surf, ax=ax1)
        ax1.set_xlabel('X Label')
        ax1.set_ylabel('Y Label')
        ax1.set_zlabel('Z Label')
        ax1.set_title(f'$e_{idx}$ (fix z=0.5)')

        # The z-projection (contour plot)
        cnt = axes[1].tricontourf(x, y, z, levels=20)
        fig.colorbar(cnt, ax=axes[1])
        axes[1].set_title('Z Projection')

        # Adjust layout
        plt.tight_layout()
        plt.show()
        
# TODO: interpolation, visualization

#%%

























#%%