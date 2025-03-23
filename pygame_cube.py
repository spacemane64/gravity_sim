import pygame
import math

# Initialize pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pygame 2D Cube")

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Cube properties
cube_size = 100
center_x, center_y = width // 2, height // 2

# Define cube vertices in 3D space (x, y, z)
vertices = [
    [-1, -1, -1],  # 0: back bottom left
    [1, -1, -1],   # 1: back bottom right
    [1, 1, -1],    # 2: back top right
    [-1, 1, -1],   # 3: back top left
    [-1, -1, 1],   # 4: front bottom left
    [1, -1, 1],    # 5: front bottom right
    [1, 1, 1],     # 6: front top right
    [-1, 1, 1]     # 7: front top left
]

# Define cube edges as pairs of vertex indices
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # back face
    (4, 5), (5, 6), (6, 7), (7, 4),  # front face
    (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
]

# Define cube faces as vertex indices
faces = [
    [0, 1, 2, 3],  # back
    [4, 5, 6, 7],  # front
    [0, 1, 5, 4],  # bottom
    [2, 3, 7, 6],  # top
    [0, 3, 7, 4],  # left
    [1, 2, 6, 5]   # right
]

def rotate_point(point, angle_x, angle_y):
    """Rotate a point around the X and Y axes"""
    # Original coordinates
    x, y, z = point
    
    # Rotate around Y-axis
    cosa = math.cos(angle_y)
    sina = math.sin(angle_y)
    x, z = x * cosa - z * sina, z * cosa + x * sina
    
    # Rotate around X-axis
    cosb = math.cos(angle_x)
    sinb = math.sin(angle_x)
    y, z = y * cosb - z * sinb, z * cosb + y * sinb
    
    return [x, y, z]

def project_point(point):
    """Project 3D point to 2D screen coordinates"""
    # Simple perspective projection
    factor = 3
    x, y, z = point
    z += 5  # Move away from camera
    
    # Apply perspective
    if z != 0:
        f = factor / z
    else:
        f = factor
    
    # Convert to screen coordinates
    x = x * f * cube_size + center_x
    y = -y * f * cube_size + center_y  # Negative because screen Y increases downward
    
    return (int(x), int(y))

# Rotation angles
angle_x = 0.5
angle_y = 0.5

# Init for rotation
rotation_speed = 0.02
dragging = False
prev_mouse_pos = (0, 0)

# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                dragging = True
                prev_mouse_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                # Calculate the change in mouse position
                dx = event.pos[0] - prev_mouse_pos[0]
                dy = event.pos[1] - prev_mouse_pos[1]
                
                # Update rotation angles
                angle_y += dx * 0.01
                angle_x += dy * 0.01
                
                prev_mouse_pos = event.pos
    
    # Clear screen
    screen.fill(WHITE)
    
    # Rotate and project vertices
    projected_vertices = []
    for vertex in vertices:
        # Apply rotation
        rotated = rotate_point(vertex, angle_x, angle_y)
        
        # Project to 2D
        projected = project_point(rotated)
        projected_vertices.append(projected)
    
    # Draw faces (filled polygons)
    for i, face in enumerate(faces):
        points = [projected_vertices[vertex] for vertex in face]
        
        # Simple face color variation
        color_value = 150 + i * 20
        if i == 0:  # Make one face red
            face_color = RED
        else:
            face_color = (color_value, color_value, color_value)
            
        pygame.draw.polygon(screen, face_color, points)
    
    # Draw edges
    for edge in edges:
        start = projected_vertices[edge[0]]
        end = projected_vertices[edge[1]]
        pygame.draw.line(screen, BLACK, start, end, 2)
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)

pygame.quit() 