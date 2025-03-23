import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def draw_triangle():
    # Draw a simple triangle with red, green, and blue vertices
    glBegin(GL_TRIANGLES)
    glColor3f(1.0, 0.0, 0.0)  # Red
    glVertex3f(0.0, 1.0, 0.0)  # Top vertex
    
    glColor3f(0.0, 1.0, 0.0)  # Green
    glVertex3f(-1.0, -1.0, 0.0)  # Bottom left vertex
    
    glColor3f(0.0, 0.0, 1.0)  # Blue
    glVertex3f(1.0, -1.0, 0.0)  # Bottom right vertex
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Simple Triangle')
    
    # Set white background
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    # Set up the perspective
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    
    # Move the triangle back a bit so it's visible
    glTranslatef(0.0, 0.0, -3.0)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
                
        # Clear the screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw the triangle
        draw_triangle()
        
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main() 