import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Cube vertices
vertices = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)

# Cube edges
edges = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 7), (3, 6)
)

# Cube faces
faces = (
    (0, 1, 2, 3),  # Back face
    (4, 5, 7, 6),  # Front face
    (0, 1, 5, 4),  # Right face
    (2, 3, 6, 7),  # Left face
    (1, 2, 7, 5),  # Top face
    (0, 3, 6, 4)   # Bottom face
)

# Face colors
colors = (
    (0.0, 1.0, 0.0),  # Green (Back face)
    (1.0, 0.0, 0.0),  # Red (Front face)
    (1.0, 0.0, 1.0),  # Magenta (Right face)
    (0.0, 1.0, 1.0),  # Cyan (Left face)
    (0.0, 0.0, 1.0),  # Blue (Top face)
    (1.0, 1.0, 0.0)   # Yellow (Bottom face)
)

def draw_cube():
    # Draw faces with different colors
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3fv(colors[i])  # Apply the color for this face
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()
    
    # Draw edges in black
    glBegin(GL_LINES)
    glColor3f(0.0, 0.0, 0.0)  # Black
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Interactive 3D Cube')

    # Print OpenGL info for debugging
    print("OpenGL Version:", glGetString(GL_VERSION).decode())
    print("OpenGL Renderer:", glGetString(GL_RENDERER).decode())

    # Set the clear color (white background)
    glClearColor(1.0, 1.0, 1.0, 1.0)

    # Set up the viewport to match the window size
    glViewport(0, 0, display[0], display[1])
    
    # Setup the projection matrix with a wider field of view
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Use a 60 degree FOV for a wider view
    gluPerspective(60, (display[0] / display[1]), 0.1, 100.0)
    
    # Switch to modelview matrix for object transformations
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Move the cube back further to ensure it's in view
    glTranslatef(0.0, 0.0, -6.0)
    
    # Enable depth testing for proper 3D rendering
    glEnable(GL_DEPTH_TEST)
    
    # Initial rotation for better visibility
    x_rotation = 30
    y_rotation = 45
    
    # Mouse variables
    last_mouse_pos = None
    mouse_sensitivity = 1.0

    # Main loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    last_mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    last_mouse_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if last_mouse_pos is not None:  # If dragging
                    current_mouse_pos = pygame.mouse.get_pos()
                    delta_x = current_mouse_pos[0] - last_mouse_pos[0]
                    delta_y = current_mouse_pos[1] - last_mouse_pos[1]
                    
                    # Update rotation angles
                    y_rotation += delta_x * mouse_sensitivity * 0.5
                    x_rotation += delta_y * mouse_sensitivity * 0.5
                    
                    last_mouse_pos = current_mouse_pos

        # Clear the screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Reset the modelview matrix
        glLoadIdentity()
        
        # Move back to see the cube
        glTranslatef(0.0, 0.0, -6.0)

        # Apply rotations
        glRotatef(x_rotation, 1, 0, 0)
        glRotatef(y_rotation, 0, 1, 0)

        # Draw the cube
        draw_cube()
        
        # Swap buffers (update the display)
        pygame.display.flip()
        
        # Control the framerate
        clock.tick(60)

    # Clean up
    pygame.quit()

if __name__ == "__main__":
    main() 