import pygame
import math
import numpy as np
import random

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
        self.velocity_edit_mode = False
        self.texture = None
        
        # Generate appropriate texture based on color
        if color == BLUE:
            self.generate_blue_texture()
        elif color == YELLOW:
            self.generate_star_texture()
        elif color == GREEN:
            self.generate_sandy_texture()
        elif color == GRAY:
            self.generate_moon_texture()
        
    def generate_blue_texture(self):
        # Create a surface for the texture
        texture_size = self.radius * 2
        self.texture = pygame.Surface((texture_size, texture_size), pygame.SRCALPHA)
        
        # Generate noise pattern with blue and green colors
        for x in range(texture_size):
            for y in range(texture_size):
                # Calculate distance from center
                dx = x - self.radius
                dy = y - self.radius
                distance = math.sqrt(dx**2 + dy**2)
                
                # Only draw within the radius
                if distance <= self.radius:
                    # Generate random noise value between 0 and 1
                    noise = random.random()
                    
                    # Mix blue and green based on noise
                    if noise < 0.6:  # More blue than green
                        blue_amount = 100 + int(noise * 155)  # Range from 100-255
                        green_amount = int(noise * 200)      # Range from 0-200
                        color = (0, green_amount, blue_amount)
                    else:
                        blue_amount = int(noise * 100)       # Range from 0-100
                        green_amount = 100 + int(noise * 155) # Range from 100-255
                        color = (0, green_amount, blue_amount)
                    
                    # Add some randomized intensity for more texture
                    intensity = 0.7 + random.random() * 0.3  # 70-100% intensity
                    r = min(255, int(color[0] * intensity))
                    g = min(255, int(color[1] * intensity))
                    b = min(255, int(color[2] * intensity))
                    
                    self.texture.set_at((x, y), (r, g, b))
    
    def generate_star_texture(self):
        # Create a surface for the texture
        texture_size = self.radius * 2
        self.texture = pygame.Surface((texture_size, texture_size), pygame.SRCALPHA)
        
        # Generate noise pattern with yellow and orange colors
        for x in range(texture_size):
            for y in range(texture_size):
                # Calculate distance from center
                dx = x - self.radius
                dy = y - self.radius
                distance = math.sqrt(dx**2 + dy**2)
                
                # Only draw within the radius
                if distance <= self.radius:
                    # Generate random noise value between 0 and 1
                    noise = random.random()
                    
                    # Mix yellow and orange based on noise and distance from center
                    # More orange at the center, more yellow at the edges
                    center_factor = 1 - (distance / self.radius)
                    
                    red = 220 + int(noise * 35)  # 220-255
                    green = 120 + int(noise * 135 * (1 - center_factor * 0.7))  # More yellow at edges
                    blue = int(noise * 50)  # Small amount of blue
                    
                    # Add some randomized flare effects
                    if random.random() < 0.05:  # 5% chance of a bright spot
                        red = 255
                        green = 230 + int(random.random() * 25)
                        blue = 100 + int(random.random() * 50)
                        
                    self.texture.set_at((x, y), (red, green, blue))
    
    def generate_sandy_texture(self):
        # Create a surface for the texture
        texture_size = self.radius * 2
        self.texture = pygame.Surface((texture_size, texture_size), pygame.SRCALPHA)
        
        # Generate noise pattern with sandy and dark yellow colors
        for x in range(texture_size):
            for y in range(texture_size):
                # Calculate distance from center
                dx = x - self.radius
                dy = y - self.radius
                distance = math.sqrt(dx**2 + dy**2)
                
                # Only draw within the radius
                if distance <= self.radius:
                    # Generate random noise value between 0 and 1
                    noise = random.random()
                    
                    # Base sandy color
                    red = 180 + int(noise * 75)     # 180-255
                    green = 150 + int(noise * 60)   # 150-210
                    blue = 60 + int(noise * 40)     # 60-100
                    
                    # Create darker patches
                    if noise < 0.3:  # 30% chance of darker areas
                        red = int(red * 0.7)
                        green = int(green * 0.7)
                        blue = int(blue * 0.6)
                    
                    self.texture.set_at((x, y), (red, green, blue))
    
    def generate_moon_texture(self):
        # Create a surface for the texture
        texture_size = self.radius * 2
        self.texture = pygame.Surface((texture_size, texture_size), pygame.SRCALPHA)
        
        # Generate noise pattern with gray colors (moon-like)
        for x in range(texture_size):
            for y in range(texture_size):
                # Calculate distance from center
                dx = x - self.radius
                dy = y - self.radius
                distance = math.sqrt(dx**2 + dy**2)
                
                # Only draw within the radius
                if distance <= self.radius:
                    # Generate random noise value between 0 and 1
                    noise = random.random()
                    
                    # Create gray color with variation
                    gray_value = 120 + int(noise * 80)  # Range from 120-200
                    
                    # Create darker craters
                    if noise < 0.2:  # 20% chance of craters
                        gray_value = 80 + int(noise * 60)  # Darker areas
                    
                    # Create some slightly brighter spots
                    if noise > 0.8:  # 20% chance of highlights
                        gray_value = 170 + int(noise * 60)  # Brighter areas
                    
                    self.texture.set_at((x, y), (gray_value, gray_value, gray_value))
        
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
        position = (int(self.position[0]), int(self.position[1]))
        
        if self.texture is not None:
            # Draw the textured planet
            surface.blit(self.texture, 
                        (position[0] - self.radius, position[1] - self.radius))
        else:
            # Draw regular planet
            pygame.draw.circle(surface, self.color, position, self.radius)
        
        # Draw velocity arrow when in velocity edit mode
        if self.velocity_edit_mode:
            # Scale factor to make arrow visible
            scale = 2.0
            arrow_end = self.position + self.velocity * scale
            
            # Draw the line
            pygame.draw.line(surface, WHITE, 
                            (int(self.position[0]), int(self.position[1])),
                            (int(arrow_end[0]), int(arrow_end[1])), 2)
            
            # Draw arrowhead
            if np.linalg.norm(self.velocity) > 0:
                # Calculate direction
                direction = self.velocity / np.linalg.norm(self.velocity)
                # Arrow head size
                head_size = 10
                # Calculate perpendicular vectors
                perpendicular = np.array([-direction[1], direction[0]])
                
                # Calculate arrowhead points
                point1 = arrow_end - direction * head_size + perpendicular * head_size/2
                point2 = arrow_end - direction * head_size - perpendicular * head_size/2
                
                # Draw arrowhead
                pygame.draw.polygon(surface, WHITE, [
                    (int(arrow_end[0]), int(arrow_end[1])),
                    (int(point1[0]), int(point1[1])),
                    (int(point2[0]), int(point2[1]))
                ])
            
            # Draw velocity magnitude text
            vel_magnitude = np.linalg.norm(self.velocity)
            vel_text = font.render(f"v={int(vel_magnitude)}", True, WHITE)
            surface.blit(vel_text, (int(arrow_end[0]) + 5, int(arrow_end[1]) + 5))
        
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
    planet2_star_distance = 150  # Distance for the green planet

    # Moon-planet distance - made smaller to strengthen planet's gravity on moon
    moon_planet_distance = 30   # was 40
    
    # Reset star
    star.position = np.array([width/2, height/2], dtype=float)
    star.velocity = np.array([0, 0], dtype=float)
    star.trail = []
    star.generate_star_texture()
    
    # Reset planet - increased distance from star
    planet.position = np.array([width/2 + planet_star_distance, height/2], dtype=float)
    
    # Calculate planet's velocity for stable orbit given the new distance
    planet_velocity_magnitude = math.sqrt(G * star.mass / planet_star_distance)
    planet.velocity = np.array([0, planet_velocity_magnitude], dtype=float)
    planet.trail = []
    
    # Regenerate texture for the blue planet
    planet.generate_blue_texture()

    # Reset planet2 (green planet) - closer to the star
    planet2.position = np.array([width/2 + planet2_star_distance, height/2], dtype=float)
    
    # Calculate planet2's velocity for stable orbit
    planet2_velocity_magnitude = math.sqrt(G * star.mass / planet2_star_distance)
    planet2.velocity = np.array([0, planet2_velocity_magnitude], dtype=float)
    planet2.trail = []
    planet2.generate_sandy_texture()
    
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
    moon.generate_moon_texture()

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
planet2_star_distance = 150  # Distance for the green planet
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

# Green planet - closer to the star
planet2 = Body(
    mass=40,  # Smaller mass than blue planet
    position=[width/2 + planet2_star_distance, height/2],
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
selected_body = None
velocity_edit_mode = False
velocity_edit_start = None
running = True

# Camera offset (will be updated to center on star)
camera_offset = np.array([0, 0], dtype=float)

while running:
    # Update camera to center on star
    target_center = np.array([width/2, height/2], dtype=float)
    camera_offset = star.position - target_center
    
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
                
            elif event.key == pygame.K_a:
                # Toggle velocity edit mode for the selected body
                if selected_body:
                    selected_body.velocity_edit_mode = not selected_body.velocity_edit_mode
                    velocity_edit_mode = selected_body.velocity_edit_mode
                    velocity_edit_start = None
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if in velocity edit mode
            if velocity_edit_mode and selected_body:
                # Start velocity edit from clicked position
                mouse_pos = pygame.mouse.get_pos()
                # Adjust mouse position by camera offset
                adjusted_mouse_pos = np.array([mouse_pos[0], mouse_pos[1]], dtype=float) + camera_offset
                velocity_edit_start = adjusted_mouse_pos
            else:
                # Check if a body was clicked
                mouse_pos = pygame.mouse.get_pos()
                # Adjust mouse position by camera offset
                adjusted_mouse_pos = np.array([mouse_pos[0], mouse_pos[1]], dtype=float) + camera_offset
                
                for body in bodies:
                    # Calculate distance to the body using camera-adjusted coordinates
                    distance = math.sqrt((body.position[0] - adjusted_mouse_pos[0])**2 + 
                                      (body.position[1] - adjusted_mouse_pos[1])**2)
                    if distance <= body.radius:
                        dragging_body = body
                        selected_body = body
                        body.being_dragged = True
                        break
        
        elif event.type == pygame.MOUSEBUTTONUP:
            # If in velocity edit mode, set the new velocity
            if velocity_edit_mode and selected_body and velocity_edit_start is not None:
                mouse_pos = pygame.mouse.get_pos()
                # Adjust mouse position by camera offset
                adjusted_mouse_pos = np.array([mouse_pos[0], mouse_pos[1]], dtype=float) + camera_offset
                
                # Calculate new velocity from the drag vector
                new_velocity = (adjusted_mouse_pos - selected_body.position) / 2.0
                selected_body.velocity = new_velocity
                
                velocity_edit_start = None
            
            # End dragging
            if dragging_body:
                dragging_body.being_dragged = False
                dragging_body = None
        
        elif event.type == pygame.MOUSEMOTION:
            # Preview velocity in edit mode
            if velocity_edit_mode and selected_body and velocity_edit_start is not None:
                mouse_pos = pygame.mouse.get_pos()
                # Adjust mouse position by camera offset
                adjusted_mouse_pos = np.array([mouse_pos[0], mouse_pos[1]], dtype=float) + camera_offset
                
                # Preview new velocity
                new_velocity = (adjusted_mouse_pos - selected_body.position) / 2.0
                selected_body.velocity = new_velocity
            
            # Regular dragging
            elif dragging_body:
                # Update body position to mouse position
                mouse_pos = pygame.mouse.get_pos()
                # Adjust mouse position by camera offset
                adjusted_mouse_pos = np.array([mouse_pos[0], mouse_pos[1]], dtype=float) + camera_offset
                dragging_body.position = adjusted_mouse_pos
                
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
    
    # Draw bodies with camera offset
    for body in bodies:
        # Calculate position adjusted for camera
        adjusted_position = body.position - camera_offset
        
        # Draw trail with camera offset
        if len(body.trail) > 1:
            for i in range(1, len(body.trail)):
                # Fade the trail from the body color to black
                alpha = i / len(body.trail)
                trail_color = (
                    int(body.color[0] * alpha),
                    int(body.color[1] * alpha),
                    int(body.color[2] * alpha)
                )
                # Get trail points adjusted for camera
                point1 = (int(body.trail[i-1][0] - camera_offset[0]), 
                          int(body.trail[i-1][1] - camera_offset[1]))
                point2 = (int(body.trail[i][0] - camera_offset[0]), 
                          int(body.trail[i][1] - camera_offset[1]))
                pygame.draw.line(screen, trail_color, point1, point2, 2)
        
        # Draw the body
        position = (int(adjusted_position[0]), int(adjusted_position[1]))
        
        if body.texture is not None:
            # Draw the textured planet
            screen.blit(body.texture, 
                      (position[0] - body.radius, position[1] - body.radius))
        else:
            # Draw regular planet
            pygame.draw.circle(screen, body.color, position, body.radius)
        
        # Draw velocity arrow when in velocity edit mode
        if body.velocity_edit_mode:
            # Scale factor to make arrow visible
            scale = 2.0
            # Calculate arrow end point adjusted for camera
            arrow_end = body.position + body.velocity * scale
            arrow_end_adjusted = arrow_end - camera_offset
            
            # Draw the line
            pygame.draw.line(screen, WHITE, 
                           position,
                           (int(arrow_end_adjusted[0]), int(arrow_end_adjusted[1])), 2)
            
            # Draw arrowhead
            if np.linalg.norm(body.velocity) > 0:
                # Calculate direction
                direction = body.velocity / np.linalg.norm(body.velocity)
                # Arrow head size
                head_size = 10
                # Calculate perpendicular vectors
                perpendicular = np.array([-direction[1], direction[0]])
                
                # Calculate arrowhead points
                point1 = arrow_end - direction * head_size + perpendicular * head_size/2
                point2 = arrow_end - direction * head_size - perpendicular * head_size/2
                
                # Adjust for camera
                point1_adjusted = point1 - camera_offset
                point2_adjusted = point2 - camera_offset
                
                # Draw arrowhead
                pygame.draw.polygon(screen, WHITE, [
                    (int(arrow_end_adjusted[0]), int(arrow_end_adjusted[1])),
                    (int(point1_adjusted[0]), int(point1_adjusted[1])),
                    (int(point2_adjusted[0]), int(point2_adjusted[1]))
                ])
            
            # Draw velocity magnitude text
            vel_magnitude = np.linalg.norm(body.velocity)
            vel_text = font.render(f"v={int(vel_magnitude)}", True, WHITE)
            screen.blit(vel_text, (int(arrow_end_adjusted[0]) + 5, int(arrow_end_adjusted[1]) + 5))
    
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
        "Click & Drag: Move bodies",
        "A: Toggle velocity edit mode (on selected body)",
        "In velocity mode: Click & drag to set velocity",
        "Camera: Centered on star"
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