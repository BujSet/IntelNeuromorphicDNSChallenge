import matplotlib.pyplot as plt
from cipic_db import CipicDatabase

def plot_coordinates(coords, title):
    x0 = coords
    n0 = coords
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], n0[:, 0],
                  n0[:, 1], n0[:, 2], length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.savefig("source_positions.png", bbox_inches='tight')
    return q

poses = CipicDatabase.subjects[12].getCartesianPositions()
plot_coordinates(poses, "Source Positions")
