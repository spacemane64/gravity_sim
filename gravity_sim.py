import pygame
import math
import numpy as np
import random

# Helper functions for coordinate conversion
def world_to_screen(world_pos, camera_offset, zoom):
    """Convert world coordinates to screen coordinates based on camera offset and zoom"""
    if isinstance(world_pos, tuple) or isinstance(world_pos, list):
        world_pos = np.array(world_pos, dtype=float)
    
    # Adjust for camera offset, then scale by zoom
    screen_x = int((world_pos[0] - camera_offset[0]) * zoom)
    screen_y = int((world_pos[1] - camera_offset[1]) * zoom)
    
    return (screen_x, screen_y)

def screen_to_world(screen_pos, camera_offset, zoom):
    """Convert screen coordinates to world coordinates based on camera offset and zoom"""
    if isinstance(screen_pos, tuple) or isinstance(screen_pos, list):
        screen_pos = np.array(screen_pos, dtype=float)
    
    # First scale by inverse zoom, then adjust for camera offset
    world_x = screen_pos[0] / zoom + camera_offset[0]
    world_y = screen_pos[1] / zoom + camera_offset[1]
    
    return np.array([world_x, world_y], dtype=float)

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
# Prediction path colors (lighter versions of body colors)
YELLOW_PRED = (255, 255, 128)
BLUE_PRED = (128, 170, 255)
GREEN_PRED = (128, 255, 128)
GRAY_PRED = (220, 220, 220)
# UI colors
SLIDER_BG = (80, 80, 80)
SLIDER_FG = (150, 150, 150)
SLIDER_HANDLE = (200, 200, 200)
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
        self.prediction_path = []  # Store future positions
        
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
    # Planet-star distance - increased for better realism
    planet_star_distance = 500  # Increased from 350
    planet2_star_distance = 250  # Increased from 150

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

# Slider class for UI control
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        
        # Calculate initial handle position
        self.handle_pos = self.value_to_pos(initial_val)
        
    def value_to_pos(self, val):
        # Convert a value to a position on the slider
        ratio = (val - self.min_val) / (self.max_val - self.min_val)
        return int(self.x + ratio * self.width)
    
    def pos_to_value(self, pos):
        # Convert a position to a value
        ratio = max(0, min(1, (pos - self.x) / self.width))
        return int(self.min_val + ratio * (self.max_val - self.min_val))
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            # Check if click is on slider handle
            if (abs(mouse_pos[0] - self.handle_pos) <= 10 and 
                self.y <= mouse_pos[1] <= self.y + self.height):
                self.dragging = True
                return True
            # Check if click is on slider bar
            elif (self.x <= mouse_pos[0] <= self.x + self.width and 
                  self.y <= mouse_pos[1] <= self.y + self.height):
                self.handle_pos = mouse_pos[0]
                self.value = self.pos_to_value(mouse_pos[0])
                return True
                
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
            
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_pos = pygame.mouse.get_pos()
            self.handle_pos = max(self.x, min(self.x + self.width, mouse_pos[0]))
            self.value = self.pos_to_value(self.handle_pos)
            return True
            
        return False
    
    def draw(self, surface):
        # Draw slider background
        pygame.draw.rect(surface, SLIDER_BG, (self.x, self.y, self.width, self.height))
        
        # Draw slider handle
        pygame.draw.circle(surface, SLIDER_HANDLE, (self.handle_pos, self.y + self.height // 2), 10)
        
        # Draw label and value
        label_text = font.render(f"{self.label}: {self.value}", True, WHITE)
        surface.blit(label_text, (self.x, self.y - 20))

def calculate_prediction_paths(bodies, steps=1000, interval=50):
    """Calculate future positions of bodies for prediction path"""
    # Create copies of bodies to avoid affecting the actual simulation
    temp_bodies = []
    for body in bodies:
        temp_body = Body(
            body.mass,
            body.position.copy(),
            body.velocity.copy(),
            body.color,
            body.radius
        )
        temp_bodies.append(temp_body)
        # Clear existing predictions
        body.prediction_path = []
    
    # No need to limit steps - use the provided value directly
    actual_steps = steps
    
    # For very high step counts, increase the interval automatically
    # to maintain responsiveness
    adaptive_interval = interval
    if actual_steps > 2000:
        # Scale the interval based on step count to limit total points
        adaptive_interval = max(interval, actual_steps // 40)
    
    # Simulate future steps
    for step in range(1, actual_steps + 1):
        # Calculate forces
        forces = {}
        for i, body1 in enumerate(temp_bodies):
            net_force = np.zeros(2, dtype=float)
            for j, body2 in enumerate(temp_bodies):
                if i != j:  # Don't calculate gravity from a body to itself
                    force = calculate_gravity(body1, body2)
                    net_force += force
            forces[body1] = net_force
        
        # Apply forces and update positions
        for i, temp_body in enumerate(temp_bodies):
            if temp_body in forces:
                # Apply force
                acceleration = forces[temp_body] / temp_body.mass
                temp_body.velocity += acceleration * dt
            # Update position
            temp_body.position += temp_body.velocity * dt
            
            # Record position at specified intervals
            if step % adaptive_interval == 0:
                bodies[i].prediction_path.append(temp_body.position.copy())

# Create bodies
# Large central body (like a star)
star = Body(
    mass=1000,
    position=[width/2, height/2],
    velocity=[0, 0],
    color=YELLOW,
    radius=35  # Increased from 20 (75% larger)
)

# Planet-star distance - increased for better realism
planet_star_distance = 500  # Increased from 350
planet2_star_distance = 250  # Increased from 150
# Moon-planet distance - made smaller to strengthen planet's gravity on moon
moon_planet_distance = 30   # was 40

# Medium orbiting body (like a planet)
planet = Body(
    mass=80,  # Increased mass to have stronger influence on moon
    position=[width/2 + planet_star_distance, height/2],
    # Initial velocity for a stable orbit
    velocity=[0, math.sqrt(G * star.mass / planet_star_distance)],
    color=BLUE,
    radius=7  # Updated from 9 to 7
)

# Green planet - closer to the star
planet2 = Body(
    mass=40,  # Smaller mass than blue planet
    position=[width/2 + planet2_star_distance, height/2],
    # Initial velocity for a stable orbit
    velocity=[0, math.sqrt(G * star.mass / planet2_star_distance)],
    color=GREEN,
    radius=4  # Updated from 6 to 4
)

# Small orbiting body (like a moon)
moon = Body(
    mass=1,
    position=[width/2 + planet_star_distance + moon_planet_distance, height/2],
    # Initial velocity will be set below
    velocity=[0, 0],
    color=GRAY,
    radius=2  # Updated from 3 to 2
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
show_predictions = True  # Toggle for showing prediction paths
prediction_update_counter = 0  # Counter to update predictions periodically

# Create slider for prediction steps
prediction_steps_slider = Slider(
    x=width - 220, 
    y=height - 40, 
    width=200, 
    height=20, 
    min_val=100, 
    max_val=5000, 
    initial_val=500,  # Default value
    label="Prediction Steps"
)

# Create slider for step interval
prediction_interval_slider = Slider(
    x=width - 220, 
    y=height - 80, 
    width=200, 
    height=20, 
    min_val=10, 
    max_val=100, 
    initial_val=50,  # Default value 
    label="Steps per Dot"
)

# Camera settings
camera_offset = np.array([0, 0], dtype=float)
camera_zoom = 1.0
camera_dragging = False
camera_drag_start = None
camera_tracking = None  # Which body the camera is tracking

# Time control
time_scale = 1.0  # Normal speed
slow_time = False

while running:
    # Handle camera tracking if enabled
    if camera_tracking is not None and camera_tracking in bodies:
        # Get the screen center
        center = np.array([width/2, height/2], dtype=float)
        # Calculate needed offset to center the tracking body
        camera_offset = camera_tracking.position - center / camera_zoom
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        # Check if sliders handled the event
        slider_handled = False
        if prediction_steps_slider.handle_event(event):
            slider_handled = True
        if prediction_interval_slider.handle_event(event):
            slider_handled = True
            
        # Only process other events if not handled by slider
        if not slider_handled:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Toggle pause
                    paused = not paused
                    
                elif event.key == pygame.K_r:
                    # Reset simulation
                    reset_simulation()
                    # Reset camera settings when simulation resets
                    camera_offset = np.array([0, 0], dtype=float)
                    camera_zoom = 1.0
                    # Reset time scale
                    time_scale = 1.0
                    slow_time = False
                    # Reset camera tracking
                    camera_tracking = None
                    
                elif event.key == pygame.K_a:
                    # Toggle velocity edit mode for the selected body
                    if selected_body:
                        selected_body.velocity_edit_mode = not selected_body.velocity_edit_mode
                        velocity_edit_mode = selected_body.velocity_edit_mode
                        velocity_edit_start = None
                        
                elif event.key == pygame.K_l:
                    # Toggle slow time
                    slow_time = not slow_time
                    if slow_time:
                        time_scale = 0.25  # 25% of normal speed (75% reduction)
                    else:
                        time_scale = 1.0  # Normal speed
                        
                elif event.key == pygame.K_p:
                    # Toggle prediction paths
                    show_predictions = not show_predictions
                    
                elif event.key == pygame.K_1:
                    # Track blue planet
                    camera_tracking = planet
                    
                elif event.key == pygame.K_2:
                    # Track sandy planet
                    camera_tracking = planet2
                    
                elif event.key == pygame.K_3:
                    # Track moon
                    camera_tracking = moon
                    
                elif event.key == pygame.K_4:
                    # Track sun
                    camera_tracking = star
                    
                elif event.key == pygame.K_5:
                    # Free camera (no tracking)
                    camera_tracking = None
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Right mouse button for camera panning
                if event.button == 3:  # Right click
                    camera_dragging = True
                    camera_drag_start = pygame.mouse.get_pos()
                    # Disable camera tracking when manually panning
                    camera_tracking = None
                # Mouse wheel for zooming
                elif event.button == 4:  # Scroll up
                    camera_zoom *= 1.1  # Zoom in
                elif event.button == 5:  # Scroll down
                    camera_zoom *= 0.9  # Zoom out
                    if camera_zoom < 0.1:  # Limit minimum zoom
                        camera_zoom = 0.1
                # Left click for body selection and interaction
                elif event.button == 1:  # Left click
                    # Check if in velocity edit mode
                    if velocity_edit_mode and selected_body:
                        # Start velocity edit from clicked position
                        mouse_pos = pygame.mouse.get_pos()
                        # Convert screen position to world position
                        world_pos = screen_to_world(mouse_pos, camera_offset, camera_zoom)
                        velocity_edit_start = world_pos
                    else:
                        # Check if a body was clicked
                        mouse_pos = pygame.mouse.get_pos()
                        # Convert screen position to world position
                        world_pos = screen_to_world(mouse_pos, camera_offset, camera_zoom)
                        
                        for body in bodies:
                            # Calculate distance to the body
                            distance = math.sqrt((body.position[0] - world_pos[0])**2 + 
                                             (body.position[1] - world_pos[1])**2)
                            if distance <= body.radius:
                                dragging_body = body
                                selected_body = body
                                body.being_dragged = True
                                break
        
        # Ensure MOUSEBUTTONUP and MOUSEMOTION are processed correctly
        # even if slider_handled was true, as these might be finishing a camera drag
        if event.type == pygame.MOUSEBUTTONUP:
            # End camera dragging
            if event.button == 3:  # Right click
                camera_dragging = False
                camera_drag_start = None
            # End body interaction
            elif event.button == 1:  # Left click
                # If in velocity edit mode, set the new velocity
                if velocity_edit_mode and selected_body and velocity_edit_start is not None:
                    mouse_pos = pygame.mouse.get_pos()
                    # Convert screen position to world position
                    world_pos = screen_to_world(mouse_pos, camera_offset, camera_zoom)
                    
                    # Calculate new velocity from the drag vector
                    new_velocity = (world_pos - selected_body.position) / 2.0
                    selected_body.velocity = new_velocity
                    
                    velocity_edit_start = None
                
                # End dragging
                if dragging_body:
                    dragging_body.being_dragged = False
                    dragging_body = None
        
        # Process camera dragging regardless of slider_handled
        elif event.type == pygame.MOUSEMOTION:
            # Camera panning (process this always)
            if camera_dragging and camera_drag_start:
                current_pos = pygame.mouse.get_pos()
                # Calculate the difference and scale by zoom (since we need to move more when zoomed in)
                dx = (camera_drag_start[0] - current_pos[0]) / camera_zoom
                dy = (camera_drag_start[1] - current_pos[1]) / camera_zoom
                camera_offset += np.array([dx, dy], dtype=float)
                camera_drag_start = current_pos
            # Only process these if slider not being interacted with
            elif not slider_handled:
                # Preview velocity in edit mode
                if velocity_edit_mode and selected_body and velocity_edit_start is not None:
                    mouse_pos = pygame.mouse.get_pos()
                    # Convert screen position to world position
                    world_pos = screen_to_world(mouse_pos, camera_offset, camera_zoom)
                    
                    # Preview new velocity
                    new_velocity = (world_pos - selected_body.position) / 2.0
                    selected_body.velocity = new_velocity
                
                # Regular dragging of bodies
                elif dragging_body:
                    # Update body position to mouse position
                    mouse_pos = pygame.mouse.get_pos()
                    # Convert screen position to world position
                    world_pos = screen_to_world(mouse_pos, camera_offset, camera_zoom)
                    dragging_body.position = world_pos
                    
                    # When dragging, reset the body's trail
                    dragging_body.trail = []
    
    # Update physics (unless paused)
    if not paused:
        # Apply time scale to dt
        effective_dt = dt * time_scale
        
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
                body.apply_force(forces[body], effective_dt)
            body.update_position(effective_dt)
        
        # Calculate prediction paths periodically, not every frame
        prediction_update_counter += 1
        update_interval = 15  # Update every 15 frames (approx. 0.25 seconds at 60 FPS)
        if prediction_steps_slider.value > 2000:
            update_interval = 30  # Less frequent updates for high step counts
        
        # Update predictions when not dragging and counter reaches interval
        if (show_predictions and 
            dragging_body is None and 
            not velocity_edit_mode and 
            prediction_update_counter >= update_interval):
            
            # Use the full slider value for steps
            calculate_prediction_paths(bodies, steps=prediction_steps_slider.value, interval=prediction_interval_slider.value)
            prediction_update_counter = 0  # Reset counter
    
    # Clear screen
    screen.fill(BLACK)
    
    # Draw bodies with camera offset and zoom
    for body in bodies:
        # Draw trail with camera offset and zoom
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
                point1 = world_to_screen((body.trail[i-1][0], body.trail[i-1][1]), camera_offset, camera_zoom)
                point2 = world_to_screen((body.trail[i][0], body.trail[i][1]), camera_offset, camera_zoom)
                pygame.draw.line(screen, trail_color, point1, point2, 2)
        
        # Draw prediction path if enabled
        if show_predictions and len(body.prediction_path) > 0:
            # Determine the prediction path color based on body color
            if body.color == YELLOW:
                pred_color = YELLOW_PRED
            elif body.color == BLUE:
                pred_color = BLUE_PRED
            elif body.color == GREEN:
                pred_color = GREEN_PRED
            elif body.color == GRAY:
                pred_color = GRAY_PRED
            else:
                pred_color = WHITE
            
            # For better performance with high step counts, draw the path as a surface
            if len(body.prediction_path) > 1:
                # Convert all points to screen coordinates
                screen_points = [world_to_screen(pos, camera_offset, camera_zoom) for pos in body.prediction_path]
                
                # Find boundaries for our temporary surface
                min_x = max(0, min(point[0] for point in screen_points) - 5)
                min_y = max(0, min(point[1] for point in screen_points) - 5)
                max_x = min(width, max(point[0] for point in screen_points) + 5)
                max_y = min(height, max(point[1] for point in screen_points) + 5)
                
                # Skip if outside screen or too small
                if max_x - min_x <= 0 or max_y - min_y <= 0:
                    continue
                    
                # Create a small surface just for the path
                path_surface = pygame.Surface((max_x - min_x, max_y - min_y), pygame.SRCALPHA)
                path_surface.fill((0, 0, 0, 0))  # Transparent background
                
                # Draw the path on the small surface
                # Offset the points to be relative to the surface
                for i in range(1, len(screen_points)):
                    p1 = (screen_points[i-1][0] - min_x, screen_points[i-1][1] - min_y)
                    p2 = (screen_points[i][0] - min_x, screen_points[i][1] - min_y)
                    
                    # Draw a thin line
                    pygame.draw.line(path_surface, pred_color, p1, p2, 1)
                
                # Blit the path surface onto the main screen
                screen.blit(path_surface, (min_x, min_y))
        
        # Calculate position adjusted for camera and zoom
        screen_pos = world_to_screen(body.position, camera_offset, camera_zoom)
        
        # Scale the radius by zoom
        screen_radius = int(body.radius * camera_zoom)
        
        # Draw the body
        if body.texture is not None:
            # Scale texture based on zoom
            scaled_size = int(body.radius * 2 * camera_zoom)
            if scaled_size < 1:  # Ensure minimum size
                scaled_size = 1
            
            scaled_texture = pygame.transform.scale(body.texture, (scaled_size, scaled_size))
            screen.blit(scaled_texture, 
                      (screen_pos[0] - scaled_size//2, screen_pos[1] - scaled_size//2))
        else:
            # Draw regular planet
            pygame.draw.circle(screen, body.color, screen_pos, screen_radius)
        
        # Draw velocity arrow when in velocity edit mode
        if body.velocity_edit_mode:
            # Scale factor to make arrow visible
            scale = 2.0
            # Calculate arrow end point in world space
            arrow_end = body.position + body.velocity * scale
            # Convert to screen space
            arrow_end_screen = world_to_screen(arrow_end, camera_offset, camera_zoom)
            
            # Draw the line
            pygame.draw.line(screen, WHITE, screen_pos, arrow_end_screen, 2)
            
            # Draw arrowhead
            if np.linalg.norm(body.velocity) > 0:
                # Calculate direction
                direction = body.velocity / np.linalg.norm(body.velocity)
                # Arrow head size scaled with zoom
                head_size = 10 * camera_zoom
                # Calculate perpendicular vectors
                perpendicular = np.array([-direction[1], direction[0]])
                
                # Calculate arrowhead points in world space
                point1 = arrow_end - direction * head_size/camera_zoom + perpendicular * head_size/(2*camera_zoom)
                point2 = arrow_end - direction * head_size/camera_zoom - perpendicular * head_size/(2*camera_zoom)
                
                # Convert to screen space
                point1_screen = world_to_screen(point1, camera_offset, camera_zoom)
                point2_screen = world_to_screen(point2, camera_offset, camera_zoom)
                
                # Draw arrowhead
                pygame.draw.polygon(screen, WHITE, [
                    arrow_end_screen,
                    point1_screen,
                    point2_screen
                ])
            
            # Draw velocity magnitude text
            vel_magnitude = np.linalg.norm(body.velocity)
            vel_text = font.render(f"v={int(vel_magnitude)}", True, WHITE)
            screen.blit(vel_text, (arrow_end_screen[0] + 5, arrow_end_screen[1] + 5))
    
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
        f"Camera Zoom: {camera_zoom:.2f}x",
        f"Time Scale: {time_scale:.2f}x",
        "Space: Pause/Resume",
        "R: Reset",
        "L: Toggle slow motion",
        "P: Toggle prediction paths",
        "1-4: Focus camera on bodies",
        "5: Free camera",
        "Click & Drag: Move bodies",
        "Right-click & Drag: Pan camera",
        "Scroll wheel: Zoom in/out",
        "A: Toggle velocity edit mode (on selected body)",
        "In velocity mode: Click & drag to set velocity"
    ]
    
    for i, text in enumerate(info_text):
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (10, 10 + i * 20))
    
    # Display pause indicator
    if paused:
        pause_text = font.render("PAUSED", True, RED)
        screen.blit(pause_text, (width - 100, 20))
    
    # Display tracking info
    if camera_tracking is not None:
        if camera_tracking == star:
            tracking_text = font.render("Tracking: Star", True, YELLOW)
        elif camera_tracking == planet:
            tracking_text = font.render("Tracking: Blue Planet", True, BLUE)
        elif camera_tracking == planet2:
            tracking_text = font.render("Tracking: Sandy Planet", True, GREEN)
        elif camera_tracking == moon:
            tracking_text = font.render("Tracking: Moon", True, GRAY)
        screen.blit(tracking_text, (width - 200, 20))
    
    # Draw sliders
    prediction_steps_slider.draw(screen)
    prediction_interval_slider.draw(screen)
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)

pygame.quit() 