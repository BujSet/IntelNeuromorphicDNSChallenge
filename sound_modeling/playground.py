import torch
import matplotlib.pyplot as plt
import numpy as np

def initialize_beam(x_dim, y_dim, initial_pos, theta_deg, sink_pos):
    """
    Initialize the beam's position and angle in the box.

    Parameters:
    - x_dim, y_dim: dimensions of the rectangular box
    - initial_pos: (x, y) position where the beam enters the box
    - theta_deg: angle of the beam entering the box, in degrees
    - sink_pos: (x, y) position of the sink inside the box

    Returns:
    - pos: current position of the beam as a tensor
    - angle: current angle of the beam as a tensor, in radians
    - sink: position of the sink as a tensor
    """
    pos = torch.tensor(initial_pos, dtype=torch.float32)
    angle = torch.tensor([theta_deg], dtype=torch.float32) * (torch.pi / 180)
    sink = torch.tensor(sink_pos, dtype=torch.float32)
    return pos, angle, sink

def simulate_and_plot_beam_path(x_dim, y_dim, beams, initial_pos):
    """
    Simulate the beam's path and plot it.
    
    Parameters:
    - x_dim, y_dim: dimensions of the rectangular box
    - pos: initial position of the beam as a tensor
    - angle: initial angle of the beam as a tensor, in radians
    - sink: position of the sink as a tensor
    """
    plt.figure(figsize=(8, 4))
    for beam in beams:
        pos = beam[0]
        angle = beam[1]
        sink = beam[2]

        positions = [pos.numpy().copy()]  # Store initial position
        hit_count = 0
        max_steps = 2000  # Limit the number of steps to prevent infinite loops
        reached_sink = False

        for _ in range(max_steps):
            # Movement of the beam using horizontal/vertical component and SOHCAHTOAH
            dx = torch.where(torch.cos(angle) > 0, x_dim - pos[0], -pos[0]) / torch.cos(angle)
            dy = torch.where(torch.sin(angle) > 0, y_dim - pos[1], -pos[1]) / torch.sin(angle)

            distance = torch.min(torch.abs(dx), torch.abs(dy))
            pos += distance * torch.tensor([torch.cos(angle), torch.sin(angle)], dtype=torch.float32)
            positions.append(pos.numpy().copy())  # Store current position
            # Add some form of decay for each reflection within the canal
            # It should be a vector multiplication because 
            # Using a smaller tolerance for reaching the sink
            # Fix the two source and sink positions
            # Move the mic to the position that has the best SNR score 
            # cos(theta) = x/h 
            # that tell you h, then # of reflections is floor(height of triangle/(half height of tube? or full height of tube))
            # phase shift impact is dependent on h
            # magnitude is a factor of distance 
            # work with frequencies too
            # get the factors (coefficients) for a perfect acoustic reflector (get them from the Pyroom acoustics source)
            # subwavelength question. 
            # OTHER TODOs:
            # look at utilization on GPU (A100s) on Condor
            # explore mp3s instead of wav files in ters of transer times 

            if torch.norm(pos - sink) < 0.5:
                reached_sink = True
                break

            if torch.abs(dx) < torch.abs(dy):
                angle = torch.pi - angle
            else:
                angle = -angle

            hit_count += 1

        # Plotting
        positions = torch.tensor(np.array(positions))
       
        plt.plot(positions[:30, 0], positions[:30, 1], '-o', label='Beam Path')
    plt.plot(sink[0], sink[1], 'rx', label='Sink', markersize=10)  # Mark the sink
    plt.plot(initial_pos[0], initial_pos[1], 'bo', label='pinhole', markersize=10)  # Mark the sink
    plt.xlim(0, x_dim)
    plt.ylim(0, y_dim)
    plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Beam Path in Rectangular Box')
    # plt.legend()
    plt.grid(True)
    plt.show()

    return hit_count, reached_sink

x_dim = 10.0  # width of the box
y_dim = 5.0   # height of the box
initial_pos = (0, 2.5)  # initial position of the beam
theta_deg = 0  # angle of the beam entering the box, in degrees
sink_pos = (8.0, 3.0)  # position of the sink
num_beams = 2
beams = []
angles = [80,20]
d_theta = 0
for i in range(num_beams):    
    # if i % 2 == 0:
    #     pos, angle, sink = initialize_beam(x_dim, y_dim, initial_pos, theta_deg + d_theta, sink_pos)
    # else:
    #     pos, angle, sink = initialize_beam(x_dim, y_dim, initial_pos, theta_deg - d_theta, sink_pos)
    pos, angle, sink = initialize_beam(x_dim, y_dim, initial_pos, angles[i], sink_pos)
    beams.append([pos,angle,sink])
    # print(f"angles: {theta_deg+d_theta} {theta_deg-d_theta}")
   
    # d_theta += 30/num_beams

# Now simulate the beam path and plot it
simulate_and_plot_beam_path(x_dim, y_dim, beams, initial_pos) 
