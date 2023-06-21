import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def create_animation(frac0, frac_list, grad_list, filename):
    def plot_3d_vectors(ax, coord0, coord1, grad, title):
        N = coord1.shape[0]  # Number of points

        # Plot coord1 with labels
        for i in range(N):
            ax.scatter(coord0[i, 0], coord0[i, 1], coord0[i, 2], c='red', label='Coord0')
            ax.text(coord0[i, 0], coord0[i, 1], coord0[i, 2], f'{i+1}', color='red')
        # Plot coord1 with labels
        for i in range(N):
            ax.scatter(coord1[i, 0], coord1[i, 1], coord1[i, 2], c='blue', label='Coord1')
            ax.text(coord1[i, 0], coord1[i, 1], coord1[i, 2], f'{i+1}', color='blue')

        # Normalize vectors
        norm_grad = np.linalg.norm(grad)
        norm_grad = 1 if norm_grad < 1e-6 else norm_grad
        normalized_grad = grad / norm_grad

        vectors = coord0 - coord1
        norm_vectors = np.linalg.norm(vectors, axis=1)
        norm_vectors[norm_vectors < 1e-6] = 1  # Avoid division by zero
        normalized_vectors = vectors / norm_vectors[:, None]

        # Define the desired length of the normalized vector
        vlength = 0.2

        # Plot normalized vectors
        ax.quiver(coord1[:, 0], coord1[:, 1], coord1[:, 2], normalized_grad[:, 0], normalized_grad[:, 1],
                  normalized_grad[:, 2], length=vlength, color='green', label='Vector')
        ax.quiver(coord1[:, 0], coord1[:, 1], coord1[:, 2], normalized_vectors[:, 0], normalized_vectors[:, 1],
                  normalized_vectors[:, 2], length=vlength, color='gray', label='Vector')

        # Set labels, title, and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Panel: {title}')

        # Set the range of the plot to 0 to 1 for all dimensions
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        return ax

    # Create the figure and axis objects
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize an empty quiver plot
    quiver = ax.quiver([], [], [], [], [], [], length=0.2, color='gray', label='Vector')

    def update(frame):
        # Clear the axis
        ax.clear()

        # Get the current frame's coordinates and gradient
        coord1 = frac_list[frame]
        grad = grad_list[frame]

        # Update the quiver plot
        quiver.set_segments([coord1, coord1 + grad])

        # Call the plot_3d_vectors function
        plot_3d_vectors(ax, frac0, coord1, grad, title=frame)

        return quiver,

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(frac_list), interval=300, blit=True)

    # Display the animation
    plt.show()
    animation.save(f'{filename}.gif', writer='imagemagick')

if __name__=='__main__':
    # Example usage
    frac_coord0 = np.random.rand(10, 3)
    frac_list = [np.random.rand(10, 3) for _ in range(10)]
    grad_list = [np.random.rand(10, 3) for _ in range(10)]

    create_animation(frac_coord0, frac_list, grad_list, 'anime0')