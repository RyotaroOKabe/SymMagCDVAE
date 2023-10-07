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
from utils.utils_plot import vis_structure
from utils.utils_material import MatSym, MatTrans, distance_sorted, Rx, Ry, Rz, rotate_cart, switch_latvecs
mpdata = pkl.load(open('data/mp_full.pkl', 'rb'))
mpids = sorted(list(mpdata.keys()))
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
    dist_matrix = np.zeros((num_points, num_points))
    
    for i in range(num_points):
        for j in range(num_points):
            dist_matrix[i, j] = np.linalg.norm(epsilon_net[i] - epsilon_net[j])
    
    return dist_matrix

#%%
# demonstrate epsilon net
N = 100  # Number of random points in the unit cube
epsilon = 0.1  # Minimum distance between points in the epsilon-net
lattice = np.diag([1,1,1])
points = np.random.rand(N, 3)@lattice
epsilon_net = generate_epsilon_net(points, epsilon)
plot_epsilon_net(epsilon_net)
print(len(epsilon_net))

# Calculate the distance matrix
dist_matrix = distance_matrix(epsilon_net)
plt.imshow(dist_matrix<epsilon)


#%%
# orbit graph (all fractional coordinates)
# 0. data preparation (structure, space group operations)
mpid = 'mp-149'
pstruct = mpdata[mpid]
ms = MatTrans(pstruct)
opes = list(set(ms.spgops))
print(mpid, ms.sg, len(opes))
frac, cart = pstruct.frac_coords, pstruct.cart_coords   # cart == frac@lattice
lattice = ms.lat


# 1. epsilon-net (frac)
N = 100  # Number of random points in the unit cube
epsilon = 0.4  # Minimum distance between points in the epsilon-net
lattice0 = np.diag([1,1,1])
points0 = np.random.rand(N, 3)@lattice0
epsilon_net = generate_epsilon_net(points0, epsilon)
plot_epsilon_net(epsilon_net)
print(len(epsilon_net))
dist_matrix = distance_matrix(epsilon_net)
plt.imshow(dist_matrix<epsilon)

# 2. local group elements
# opes

# 3. find all d_g
enet0, enet1 = epsilon_net.copy(), epsilon_net.copy()
xs0 = enet0
ys0 = enet1
neps = len(enet0)
nops = len(opes)
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
    # print(i, d_x_ys1s[idx_min, range(len(ys1))].shape)
    dmatrix.append(d_x_ys1s[idx_min, range(len(ys1))])
idx_mins = np.stack(idx_mins)
dmatrix = np.stack(dmatrix)
plt.imshow(dmatrix)
plt.show()
plt.close()

# 4. add edges to x,y \in Gammma
delta = 0.6
mask = dmatrix<delta
plt.imshow(mask)
plt.show()
plt.close()
idx_mx, idx_my = np.nonzero(mask)

#%%
# orbit graph (cartesian coordinate)
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
ax.set_title(f'Epsilon-Net in 3D: {len(epsilon_net)} points')

plt.show()


#%%
W = np.exp(-dmatrix/(2*epsilon**2))
W[mask==0]=0
D = np.diag(np.sum(W, axis=-1))
L = np.diag([1 for _ in range(len(D))]) +  - np.linalg.inv(D)@W
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
# interpolation (use gird data)
epsilon = 0.4  # Minimum distance between points in the epsilon-net
Num = 5  # As an example, you can adjust this as needed.
# Create a grid of values between 0 and 1 (you can adjust the range if needed)
x = np.linspace(0.05, 0.95, Num)
y = np.linspace(0.05, 0.95, Num)
z = np.linspace(0.05, 0.95, Num)
# Generate a mesh grid
xv, yv, zv = np.meshgrid(x, y, z)
# Reshape and stack to get the desired output shape
grid_points = np.stack((xv.ravel(), yv.ravel(), zv.ravel()), axis=-1)
print(grid_points.shape)  # This should print (N**3, 3)
epsilon_net1 = generate_epsilon_net(grid_points, epsilon)
plot_epsilon_net(epsilon_net1)
print(len(epsilon_net1))
dist_matrix1 = distance_matrix(epsilon_net1)
plt.imshow(dist_matrix1<epsilon)

enet0, enet1 = epsilon_net1.copy(), epsilon_net1.copy()
xs0 = enet0
ys0 = enet1
neps = len(enet0)
nops = len(opes)
ys1s = []
for ope in opes:
    opr = ope.rotation_matrix
    opt = ope.translation_vector
    ys1 = (ys0@opr.T + opt)#%1
    ys1s.append(ys1)
ys1s = np.stack(ys1s)
dmatrix1 = []
idx_mins = []
for i, x0 in enumerate(xs0):
    d_x_ys1s = np.linalg.norm(x0 - ys1s, axis=-1)
    idx_min = np.argmin(d_x_ys1s, axis=0)  # idx_min.shape: len(ys1s)
    idx_mins.append(idx_min)
    # print(i, d_x_ys1s[idx_min, range(len(ys1))].shape)
    dmatrix1.append(d_x_ys1s[idx_min, range(len(ys1))])
idx_mins = np.stack(idx_mins)
dmatrix1 = np.stack(dmatrix1)
plt.imshow(dmatrix1)
plt.title('Edge (x,y) length')
plt.show()
plt.close()

# 4. add edges to x,y \in Gammma
delta = 0.4
mask1 = dmatrix1<delta
plt.imshow(mask1)
plt.title('Edge (x,y) length < delta')
plt.show()
plt.close()
idx_mx, idx_my = np.nonzero(mask1)

# orbit graph (cartesian coordinate)
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
ax.set_title(f'Orbit Graph in 3D: {len(epsilon_net1)} points')

plt.show()

W = np.exp(-dmatrix1/(2*epsilon**2))
W[mask1==0]=0
D = np.diag(np.sum(W, axis=-1))
L = np.diag([1 for _ in range(len(D))]) +  - np.linalg.inv(D)@W
e_val, e_vec = np.linalg.eig(L)    # e_val.shape, e_vec.shape = ((91,), (91, 91))
fig, axs = plt.subplots(1,3, figsize=(20, 6))
ax0 = axs[0]
ax0.scatter(e_val)
ax0.set_title('eigenvalues')
ax1 = axs[1]
ax1.imshow(e_vec.real)
ax1.set_title('eigenvectors (real)')
ax2 = axs[2]
ax2.imshow(e_vec.imag)
ax2.set_title('eigenvectors (imag)')

import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Assuming data is your (N, 4) array
data = np.array(...)  # shape (N, 4), where each row is (x, y, z, a)

# Split data into points and values
points = data[:, :3]  # shape (N, 3), representing (x, y, z)
values = data[:, 3]   # shape (N,), representing a

# Create interpolator
interpolator = LinearNDInterpolator(points, values)

# Now, you can interpolate a value for any (x,y,z) point
x_new, y_new, z_new = 1.5, 2.5, 3.5
a_new = interpolator(x_new, y_new, z_new)

print(a_new)



#%% 
# interpolation (use 4D data)
e_i = e_vec*2/epsilon
lmd_i = e_val*2/epsilon

#%%

























#%%