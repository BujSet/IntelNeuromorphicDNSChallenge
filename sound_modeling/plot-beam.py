import matplotlib.pyplot as plt

def simulate_and_plot_beam_path(x_dim, y_dim, pos, angle, sink):
    """
    Simulate the beam's path and plot it.
    
    Parameters:
    - x_dim, y_dim: dimensions of the rectangular box
    - pos: initial position of the beam as a tensor
    - angle: initial angle of the beam as a tensor, in radians
    - sink: position of the sink as a tensor
    """
    positions = [pos.numpy().copy()]  # Store initial position
    hit_count = 0
    max_steps = 1000  # Limit the number of steps to prevent infinite loops
    reached_sink = False

    for _ in range(max_steps):
        dx = torch.where(torch.cos(angle) > 0, x_dim - pos[0], -pos[0]) / torch.cos(angle)
        dy = torch.where(torch.sin(angle) > 0, y_dim - pos[1], -pos[1]) / torch.sin(angle)

        distance = torch.min(torch.abs(dx), torch.abs(dy))
        pos += distance * torch.tensor([torch.cos(angle), torch.sin(angle)], dtype=torch.float32)
        positions.append(pos.numpy().copy())  # Store current position

        if torch.norm(pos - sink) < 0.1:
            reached_sink = True
            break

        if torch.abs(dx) < torch.abs(dy):
            angle = torch.pi - angle
        else:
            angle = -angle

        hit_count += 1

    # Plotting
    positions = torch.tensor(positions)
    plt.figure(figsize=(8, 4))
    plt.plot(positions[:, 0], positions[:, 1], '-o', label='Beam Path')
    plt.plot(sink[0], sink[1], 'rx', label='Sink', markersize=10)  # Mark the sink
    plt.xlim(0, x_dim)
    plt.ylim(0, y_dim)
    plt.xlabel('X dimension')
    plt.ylabel('Y dimension')
    plt.title('Beam Path in Rectangular Box')
    plt.legend()
    plt.grid(True)
    plt.show()

    return hit_count, reached_sink

# Re-initialize to reset starting position and angle
pos, angle, sink = initialize_beam_corrected(x_dim, y_dim, initial_pos, theta_deg, sink_pos)

# Now simulate the beam path and plot it
simulate_and_plot_beam_path(x_dim, y_dim, pos, angle, sink)