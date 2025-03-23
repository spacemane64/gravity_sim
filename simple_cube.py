import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def draw_cube():
    # We'll define each face separately with explicit vertices
    # Red: Front face
    glBegin(GL_QUADS)
    glColor3f(1.0, 0.0, 0.0)  # Red
    glVertex3f(-1.0, -1.0, 1.0)  # Bottom left
    glVertex3f(1.0, -1.0, 1.0)   # Bottom right
    glVertex3f(1.0, 1.0, 1.0)    # Top right
    glVertex3f(-1.0, 1.0, 1.0)   # Top left
    glEnd()

    # Green: Back face
    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)  # Green
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(1.0, -1.0, -1.0)
    glEnd()

    # Blue: Top face
    glBegin(GL_QUADS)
    glColor3f(0.0, 0.0, 1.0)  # Blue
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, -1.0)
    glEnd()

    # Yellow: Bottom face
    glBegin(GL_QUADS)
    glColor3f(1.0, 1.0, 0.0)  # Yellow
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glEnd()

    # Magenta: Right face
    glBegin(GL_QUADS)
    glColor3f(1.0, 0.0, 1.0)  # Magenta
    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glEnd()

    # Cyan: Left face
    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 1.0)  # Cyan
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Simple OpenGL Cube")
    
    # Set up simple orthographic projection (no perspective)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Use orthographic projection for simpler visualization
    glOrtho(-2, 2, -2, 2, -10, 10)
    
    # Switch to model-view matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Enable depth testing
    glEnable(GL_DEPTH_TEST)
    
    # Set initial rotation
    rotation_x = 30
    rotation_y = 30
    
    # Mouse control variables
    mouse_down = False
    last_pos = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True
                    last_pos = event.pos
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if mouse_down:
                    dx = event.pos[0] - last_pos[0]
                    dy = event.pos[1] - last_pos[1]
                    rotation_y += dx * 0.5
                    rotation_x += dy * 0.5
                    last_pos = event.pos

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Reset the modelview matrix
        glLoadIdentity()
        
        # Apply rotations
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)
        
        # Draw our cube
        draw_cube()
        
        # Update display
        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main() 