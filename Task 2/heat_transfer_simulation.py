import numpy as np
import matplotlib.pyplot as plt

def initialize_temperature(Nx, Ny, T_left, T_right, T_top, T_bottom):
    """
    Initialize the temperature field with boundary conditions.
    """
    T = np.zeros((Nx, Ny))
    T[:, 0] = T_left  # Left boundary
    T[:, -1] = T_right  # Right boundary
    T[0, :] = T_top  # Top boundary
    T[-1, :] = T_bottom  # Bottom boundary
    return T

def heat_equation_fd(T, alpha, dx, dy, dt, Nt):
    """
    Solve the 2D heat equation using finite difference method.
    """
    Nx, Ny = T.shape
    T_new = T.copy()
    for n in range(Nt):
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt * (
            (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
            (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
        )
        T = T_new.copy()
    return T

def visualize_temperature(T, Lx, Ly):
    """
    Create a heatmap visualization of the temperature distribution.
    """
    plt.imshow(T, extent=[0, Lx, 0, Ly], origin='lower', cmap='hot')
    plt.colorbar(label='Temperature (°C)')
    plt.title('2D Temperature Distribution')
    plt.xlabel('X-axis (m)')
    plt.ylabel('Y-axis (m)')
    plt.show()

# Parameters
Lx, Ly = 1.0, 1.0  # Domain dimensions in meters
Nx, Ny = 50, 50  # Grid points
dx, dy = Lx/(Nx-1), Ly/(Ny-1)  # Grid spacing
alpha = 0.01  # Thermal diffusivity (m^2/s)
dt = 0.0001  # Time step (s)
Nt = 500  # Number of time steps

# Boundary conditions
T_left = 100.0  # Temperature on the left boundary (°C)
T_right = 50.0  # Temperature on the right boundary (°C)
T_top = 75.0  # Temperature on the top boundary (°C)
T_bottom = 25.0  # Temperature on the bottom boundary (°C)

# Initialize and solve
T_initial = initialize_temperature(Nx, Ny, T_left, T_right, T_top, T_bottom)
T_final = heat_equation_fd(T_initial, alpha, dx, dy, dt, Nt)

# Visualize results
visualize_temperature(T_final, Lx, Ly)
