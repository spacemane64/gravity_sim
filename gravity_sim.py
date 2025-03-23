import pygame
import math
import numpy as np

# Initialize pygame
pygame.init()

# Set up the display
width, height = 1000, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Kepler's Orbital Motion Simulation with Moon")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 100, 255)
GRAY = (180, 180, 180)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
# Physical constants (scaled for visualization)
G = 1000  # Gravitational constant (scaled up for better visualization)

# Bodies
class Body:
    def __init__(self, mass, position, velocity, color, radius):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        self.radius = radius
        self.trail = []
        self.max_trail_length = 100
        self.being_dragged = False
        
    def update_position(self, dt):
        if not self.being_dragged:
            self.position += self.velocity * dt
            
        # Update trail
        self.trail.append((int(self.position[0]), int(self.position[1])))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
    
    def draw(self, surface):
        # Draw trail
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                # Fade the trail from the body color to black
                alpha = i / len(self.trail)
                trail_color = (
                    int(self.color[0] * alpha),
                    int(self.color[1] * alpha),
                    int(self.color[2] * alpha)
                )
                pygame.draw.line(surface, trail_color, self.trail[i-1], self.trail[i], 2)
        
        # Draw the body
        pygame.draw.circle(surface, self.color, 
                           (int(self.position[0]), int(self.position[1])), 
                           self.radius)
        
    def apply_force(self, force, dt):
        if not self.being_dragged:
            # F = ma -> a = F/m
            acceleration = force / self.mass
            self.velocity += acceleration * dt

def calculate_gravity(body1, body2):
    # Vector from body1 to body2
    r_vector = body2.position - body1.position
    
    # Distance between bodies
    distance = np.linalg.norm(r_vector)
    
    # Avoid division by zero or very small distances
    if distance < 1:
        distance = 1
    
    # Calculate gravitational force magnitude: F = G * m1 * m2 / r^2
    force_magnitude = G * body1.mass * body2.mass / (distance ** 2)
    
    # Direction of the force (unit vector from body1 to body2)
    direction = r_vector / distance
    
    # Calculate the force vector
    force_vector = force_magnitude * direction
    
    return force_vector

def reset_simulation():
    # Planet-star distance - increased to reduce star's influence on moon
    planet_star_distance = 350  # was 200
    planet2_star_distance = 100  # was 200

    # Moon-planet distance - made smaller to strengthen planet's gravity on moon
    moon_planet_distance = 30   # was 40
    
    # Reset star
    star.position = np.array([width/2, height/2], dtype=float)
    star.velocity = np.array([0, 0], dtype=float)
    star.trail = []
    
    # Reset planet - increased distance from star
    planet.position = np.array([width/2 + planet_star_distance, height/2], dtype=float)
    
    # Calculate planet's velocity for stable orbit given the new distance
    planet_velocity_magnitude = math.sqrt(G * star.mass / planet_star_distance)
    planet.velocity = np.array([0, planet_velocity_magnitude], dtype=float)
    planet.trail = []

        # Calculate planet's velocity for stable orbit given the new distance
    planet2_velocity_magnitude = math.sqrt(G * star.mass / planet2_star_distance)
    planet2.velocity = np.array([0, planet2_velocity_magnitude], dtype=float)
    planet2.trail = []
    
    # Reset moon - positioned closer to planet
    # Position the moon relative to the planet
    moon.position = np.array([width/2 + planet_star_distance + moon_planet_distance, height/2], dtype=float)
    
    # Calculate orbital velocity for the moon around the planet
    # Using a factor less than 1.0 to create a more stable orbit
    moon_velocity_magnitude = math.sqrt(G * planet.mass / moon_planet_distance) * 0.9
    
    # Set the moon's velocity perpendicular to the line connecting it to the planet
    # Adding planet's velocity to ensure it moves with the planet
    moon.velocity = np.array([0, moon_velocity_magnitude], dtype=float) + planet.velocity
    moon.trail = []

# Create bodies
# Large central body (like a star)
star = Body(
    mass=1000,
    position=[width/2, height/2],
    velocity=[0, 0],
    color=YELLOW,
    radius=20
)

# Planet-star distance - increased to reduce star's influence on moon
planet_star_distance = 350  # was 200
planet2_star_distance = 75  # was 200
# Moon-planet distance - made smaller to strengthen planet's gravity on moon
moon_planet_distance = 30   # was 40

# Medium orbiting body (like a planet)
planet = Body(
    mass=80,  # Increased mass to have stronger influence on moon
    position=[width/2 + planet_star_distance, height/2],
    # Initial velocity for a stable orbit
    velocity=[0, math.sqrt(G * star.mass / planet_star_distance)],
    color=BLUE,
    radius=12
)

planet2 = Body(
    mass=80,  # Increased mass to have stronger influence on moon
    position=[width/2 + planet_star_distance/3, height/2],
    # Initial velocity for a stable orbit
    velocity=[0, math.sqrt(G * star.mass / planet2_star_distance)],
    color=GREEN,
    radius=8
)

# Small orbiting body (like a moon)
moon = Body(
    mass=1,
    position=[width/2 + planet_star_distance + moon_planet_distance, height/2],
    # Initial velocity will be set below
    velocity=[0, 0],
    color=GRAY,
    radius=5
)

bodies = [star, planet, planet2, moon]

# Set initial moon velocity
# Calculate orbital velocity for the moon around the planet
moon_velocity_magnitude = math.sqrt(G * planet.mass / moon_planet_distance) * 0.9  # Factor reduced to create more stable orbit

# Set the moon's velocity perpendicular to the line connecting it to the planet
moon.velocity = np.array([0, moon_velocity_magnitude], dtype=float) + planet.velocity

# Time step for simulation - reduced for more stable integration
dt = 0.005  # was 0.01

# Font for displaying information
font = pygame.font.SysFont('Arial', 16)

# Main game loop
clock = pygame.time.Clock()
paused = False
dragging_body = None
running = True

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Toggle pause
                paused = not paused
                
            elif event.key == pygame.K_r:
                # Reset simulation
                reset_simulation()
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if a body was clicked
            mouse_pos = pygame.mouse.get_pos()
            for body in bodies:
                # Calculate distance to the body
                distance = math.sqrt((body.position[0] - mouse_pos[0])**2 + 
                                   (body.position[1] - mouse_pos[1])**2)
                if distance <= body.radius:
                    dragging_body = body
                    body.being_dragged = True
                    break
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_body:
                dragging_body.being_dragged = False
                dragging_body = None
        
        elif event.type == pygame.MOUSEMOTION:
            if dragging_body:
                # Update body position to mouse position
                mouse_pos = pygame.mouse.get_pos()
                dragging_body.position = np.array([mouse_pos[0], mouse_pos[1]], dtype=float)
                
                # When dragging, reset the body's trail
                dragging_body.trail = []
    
    # Update physics (unless paused)
    if not paused:
        # For more accurate simulation, we calculate forces first then apply them
        forces = {}
        for i, body1 in enumerate(bodies):
            net_force = np.zeros(2, dtype=float)
            for j, body2 in enumerate(bodies):
                if i != j:  # Don't calculate gravity from a body to itself
                    force = calculate_gravity(body1, body2)
                    net_force += force
            forces[body1] = net_force
        
        # Apply forces and update positions
        for body in bodies:
            if body in forces:
                body.apply_force(forces[body], dt)
            body.update_position(dt)
    
    # Clear screen
    screen.fill(BLACK)
    
    # Draw bodies
    for body in bodies:
        body.draw(screen)
    
    # Display info
    info_text = [
        f"Bodies: {len(bodies)}",
        f"Star Mass: {star.mass}",
        f"Planet Mass: {planet.mass}",
        f"Planet2 Mass: {planet2.mass}",
        f"Moon Mass: {moon.mass}",
        f"Planet-Star Distance: {int(np.linalg.norm(planet.position - star.position))}",
        f"Planet2-Star Distance: {int(np.linalg.norm(planet2.position - star.position))}",
        f"Moon-Planet Distance: {int(np.linalg.norm(moon.position - planet.position))}",
        f"Planet Velocity: {int(np.linalg.norm(planet.velocity))}",
        f"Planet2 Velocity: {int(np.linalg.norm(planet2.velocity))}",              
        f"Moon Velocity: {int(np.linalg.norm(moon.velocity))}",
        "Space: Pause/Resume",
        "R: Reset",
        "Click & Drag: Move bodies"
    ]
    
    for i, text in enumerate(info_text):
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (10, 10 + i * 20))
    
    # Display pause indicator
    if paused:
        pause_text = font.render("PAUSED", True, RED)
        screen.blit(pause_text, (width - 100, 20))
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)

pygame.quit() 