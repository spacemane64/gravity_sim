import pygame
import numpy as np
import math
import random
import time
import os
from typing import Tuple, List

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
MAP_WIDTH = 1024
MAP_HEIGHT = 512
FPS = 60

# Colors
COLOR_BACKGROUND = (10, 10, 30)
COLOR_UI_BG = (30, 30, 50, 180)
COLOR_UI_TEXT = (220, 220, 220)
COLOR_UI_BORDER = (100, 100, 150)


class PerlinNoise:
    """Simple Perlin noise implementation for terrain generation"""
    
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else random.randint(0, 10000)
        random.seed(self.seed)
        
        # Generate permutation table
        self.perm = list(range(256))
        random.shuffle(self.perm)
        self.perm += self.perm  # Double it to avoid overflow later
    
    def noise(self, x, y, z):
        """3D Perlin noise function"""
        # Find unit cube that contains the point
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        Z = int(math.floor(z)) & 255
        
        # Find relative x, y, z of point in cube
        x -= math.floor(x)
        y -= math.floor(y)
        z -= math.floor(z)
        
        # Compute fade curves for each of x, y, z
        u = self._fade(x)
        v = self._fade(y)
        w = self._fade(z)
        
        # Hash coordinates of the 8 cube corners
        A = self.perm[X] + Y
        AA = self.perm[A] + Z
        AB = self.perm[A + 1] + Z
        B = self.perm[X + 1] + Y
        BA = self.perm[B] + Z
        BB = self.perm[B + 1] + Z
        
        # Add blended results from 8 corners of cube
        res = self._lerp(w, 
                  self._lerp(v, 
                      self._lerp(u, 
                          self._grad(self.perm[AA], x, y, z),
                          self._grad(self.perm[BA], x - 1, y, z)
                      ),
                      self._lerp(u, 
                          self._grad(self.perm[AB], x, y - 1, z),
                          self._grad(self.perm[BB], x - 1, y - 1, z)
                      )
                  ),
                  self._lerp(v, 
                      self._lerp(u, 
                          self._grad(self.perm[AA + 1], x, y, z - 1),
                          self._grad(self.perm[BA + 1], x - 1, y, z - 1)
                      ),
                      self._lerp(u, 
                          self._grad(self.perm[AB + 1], x, y - 1, z - 1),
                          self._grad(self.perm[BB + 1], x - 1, y - 1, z - 1)
                      )
                  )
              )
        
        # Scale to [-1, 1]
        return res
    
    def _fade(self, t):
        """Fade function as defined by Ken Perlin"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, t, a, b):
        """Linear interpolation between a and b by t where t is [0, 1]"""
        return a + t * (b - a)
    
    def _grad(self, hash, x, y, z):
        """Calculate the dot product of a randomly selected gradient vector"""
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


class Biome:
    """Represents a biome with specific color and conditions"""
    
    def __init__(self, name, color, min_height, max_height, min_moisture=0.0, max_moisture=1.0, min_temp=0.0, max_temp=1.0):
        self.name = name
        self.color = color
        self.min_height = min_height
        self.max_height = max_height
        self.min_moisture = min_moisture
        self.max_moisture = max_moisture
        self.min_temp = min_temp
        self.max_temp = max_temp
    
    def matches(self, height, moisture, temperature):
        """Check if the given parameters match this biome"""
        return (self.min_height <= height <= self.max_height and
                self.min_moisture <= moisture <= self.max_moisture and
                self.min_temp <= temperature <= self.max_temp)


class UIElement:
    """Base class for UI elements"""
    
    def __init__(self, rect, text="", font_size=20, bg_color=COLOR_UI_BG, text_color=COLOR_UI_TEXT, border_color=COLOR_UI_BORDER):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font_size = font_size
        self.bg_color = bg_color
        self.text_color = text_color
        self.border_color = border_color
        self.font = pygame.font.SysFont("Arial", font_size)
        self.active = False
    
    def draw(self, screen):
        """Draw the UI element"""
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, 2)
        
        if self.text:
            text_surface = self.font.render(self.text, True, self.text_color)
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)
    
    def is_clicked(self, pos):
        """Check if the element is clicked"""
        return self.rect.collidepoint(pos)
    
    def set_font_size(self, size):
        """Update the font size"""
        self.font_size = size
        self.font = pygame.font.SysFont("Arial", size)


class DraggableUIElement(UIElement):
    """UI element that can be dragged and resized"""
    
    def __init__(self, rect, text="", **kwargs):
        super().__init__(rect, text, **kwargs)
        self.dragging = False
        self.drag_offset = (0, 0)
        self.resizing = False
        self.min_width = 100
        self.min_height = 40
        self.resize_handle_size = 20
        
    def get_resize_handle_rect(self):
        """Get the rectangle for the resize handle (bottom-right corner)"""
        return pygame.Rect(
            self.rect.right - self.resize_handle_size,
            self.rect.bottom - self.resize_handle_size,
            self.resize_handle_size,
            self.resize_handle_size
        )
    
    def draw(self, screen):
        """Draw the UI element with resize handle"""
        # Draw the main element
        super().draw(screen)
        
        # Draw a resize handle in the bottom-right corner
        resize_handle = self.get_resize_handle_rect()
        pygame.draw.rect(screen, self.border_color, resize_handle)
        pygame.draw.line(screen, self.text_color, 
                         (resize_handle.left + 5, resize_handle.bottom - 5),
                         (resize_handle.right - 5, resize_handle.bottom - 5), 2)
        pygame.draw.line(screen, self.text_color, 
                         (resize_handle.right - 5, resize_handle.top + 5),
                         (resize_handle.right - 5, resize_handle.bottom - 5), 2)
    
    def handle_event(self, event):
        """Handle mouse events for dragging and resizing"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            
            # Check if resize handle is clicked
            if self.get_resize_handle_rect().collidepoint(mouse_pos):
                self.resizing = True
                return True
            
            # Check if the element is clicked for dragging
            elif self.rect.collidepoint(mouse_pos):
                self.dragging = True
                self.drag_offset = (mouse_pos[0] - self.rect.x, mouse_pos[1] - self.rect.y)
                return True
                
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
            self.resizing = False
            
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                # Move the element
                self.rect.x = event.pos[0] - self.drag_offset[0]
                self.rect.y = event.pos[1] - self.drag_offset[1]
                return True
                
            elif self.resizing:
                # Resize the element
                new_width = max(self.min_width, event.pos[0] - self.rect.x)
                new_height = max(self.min_height, event.pos[1] - self.rect.y)
                self.rect.width = new_width
                self.rect.height = new_height
                
                # Scale font size based on height ONLY, not width
                # This ensures horizontal resizing doesn't affect text size
                new_font_size = max(6, min(18, int(self.rect.height / 14)))
                self.set_font_size(new_font_size)
                return True
                
        return False


class BiomeLegend(DraggableUIElement):
    """Legend showing biome colors and names"""
    
    def __init__(self, rect, biomes, **kwargs):
        super().__init__(rect, **kwargs)
        self.biomes = biomes
        self.visible = True
        self.scroll_offset = 0
        self.original_font_size = self.font_size
    
    def draw(self, screen):
        """Draw the biome legend"""
        if not self.visible:
            return
            
        # Draw background and border
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, 2)
        
        # Draw resize handle
        resize_handle = self.get_resize_handle_rect()
        pygame.draw.rect(screen, self.border_color, resize_handle)
        pygame.draw.line(screen, self.text_color, 
                         (resize_handle.left + 5, resize_handle.bottom - 5),
                         (resize_handle.right - 5, resize_handle.bottom - 5), 2)
        pygame.draw.line(screen, self.text_color, 
                         (resize_handle.right - 5, resize_handle.top + 5),
                         (resize_handle.right - 5, resize_handle.bottom - 5), 2)
        
        # Recalculate font size based on panel width
        entry_height = max(14, self.font_size + 4)  # Smaller minimum height
        
        # Draw title with slightly larger font
        title_font = pygame.font.SysFont("Arial", self.font_size + 1)  # Smaller title font
        title_surface = title_font.render("Biome Legend", True, self.text_color)
        title_rect = title_surface.get_rect(topleft=(self.rect.x + 8, self.rect.y + 8))
        screen.blit(title_surface, title_rect)
        
        # Calculate max biomes based on current size
        available_height = self.rect.height - 40  # Smaller space for title and margins
        max_displayed = max(1, min(len(self.biomes), int(available_height / entry_height)))
        
        # Adjust scroll offset if needed to prevent empty space
        if self.scroll_offset > len(self.biomes) - max_displayed:
            self.scroll_offset = max(0, len(self.biomes) - max_displayed)
        
        # Draw biome entries
        y_offset = 30  # Reduced from 40
        
        # Calculate which biomes to display based on scroll offset
        display_biomes = self.biomes[self.scroll_offset:self.scroll_offset + max_displayed]
        
        # Calculate proper width for text based on panel width
        available_text_width = self.rect.width - 50  # Smaller space for color square and margins
        
        for biome in display_biomes:
            # Draw color square
            color_square_size = min(14, max(8, int(self.rect.width * 0.06)))  # Smaller color squares
            color_rect = pygame.Rect(self.rect.x + 8, self.rect.y + y_offset, color_square_size, color_square_size)
            pygame.draw.rect(screen, biome.color, color_rect)
            pygame.draw.rect(screen, (100, 100, 100), color_rect, 1)
            
            # Draw biome name, potentially truncating with ellipsis if too long
            name_surface = self.font.render(biome.name, True, self.text_color)
            if name_surface.get_width() > available_text_width:
                # Try to fit with smaller font
                smaller_font = pygame.font.SysFont("Arial", max(6, self.font_size - 1))
                name_surface = smaller_font.render(biome.name, True, self.text_color)
                if name_surface.get_width() > available_text_width:
                    # Still too long, truncate with ellipsis
                    for length in range(len(biome.name) - 1, 0, -1):
                        truncated = biome.name[:length] + "..."
                        name_surface = smaller_font.render(truncated, True, self.text_color)
                        if name_surface.get_width() <= available_text_width:
                            break
            
            screen.blit(name_surface, (self.rect.x + color_square_size + 14, self.rect.y + y_offset + (color_square_size - name_surface.get_height()) // 2))
            
            y_offset += entry_height
        
        # Draw scroll indicators if needed
        if len(self.biomes) > max_displayed:
            if self.scroll_offset > 0:
                pygame.draw.polygon(screen, self.text_color, [
                    (self.rect.centerx - 8, self.rect.y + 20),
                    (self.rect.centerx + 8, self.rect.y + 20),
                    (self.rect.centerx, self.rect.y + 12)
                ])
            
            if self.scroll_offset + max_displayed < len(self.biomes):
                bottom_y = self.rect.y + y_offset + 4
                pygame.draw.polygon(screen, self.text_color, [
                    (self.rect.centerx - 8, bottom_y),
                    (self.rect.centerx + 8, bottom_y),
                    (self.rect.centerx, bottom_y + 8)
                ])
    
    def handle_event(self, event):
        """Handle scroll events for the legend and dragging/resizing"""
        if not self.visible:
            return False
            
        # First check for dragging/resizing (from parent class)
        parent_result = super().handle_event(event)
        if parent_result:
            return True
            
        # Then check for scrolling if mouse is over the element
        if self.rect.collidepoint(pygame.mouse.get_pos()):
            if event.type == pygame.MOUSEWHEEL:
                # Calculate max biomes based on current size
                entry_height = max(14, self.font_size + 4)
                available_height = self.rect.height - 40
                max_displayed = max(1, min(len(self.biomes), int(available_height / entry_height)))
                
                # Scroll up/down within valid range
                if event.y > 0 and self.scroll_offset > 0:
                    self.scroll_offset -= 1
                elif event.y < 0 and self.scroll_offset < len(self.biomes) - max_displayed:
                    self.scroll_offset += 1
                return True
        
        return False
    
    def toggle_visibility(self):
        """Toggle legend visibility"""
        self.visible = not self.visible
        return self.visible


class PlanetMapGenerator:
    """Generates a 2D map of a procedurally generated planet"""
    
    def __init__(self, 
                seed=None, 
                ocean_level=0.45,
                mountain_level=0.75,
                width=MAP_WIDTH,
                height=MAP_HEIGHT,
                continent_count=5):  # Added parameter for controlling number of continents
        """
        Initialize the map generator
        
        Args:
            seed: Random seed for noise generation
            ocean_level: Value below which is considered ocean (0-1)
            mountain_level: Value above which is considered mountains (0-1)
            width: Width of the map in pixels
            height: Height of the map in pixels
            continent_count: Number of continental regions to generate
        """
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.ocean_level = ocean_level
        self.mountain_level = mountain_level
        self.continent_count = continent_count
        
        # Set random seed
        random.seed(self.seed)
        
        # Initialize noise generators
        self.noise_gen = PerlinNoise(seed=self.seed)
        self.moisture_gen = PerlinNoise(seed=self.seed + 1)
        self.temp_gen = PerlinNoise(seed=self.seed + 2)
        self.continent_shape_gen = PerlinNoise(seed=self.seed + 3)
        
        # Generate continent placement data
        self.continents = self._generate_continent_shapes()
        
        # Set different scales for various terrain features
        self.continent_scale = 0.6
        self.mountain_scale = 2.5
        self.detail_scale = 6.0
        self.moisture_scale = 2.8
        self.temp_scale = 1.5
        
        # Define biomes
        self.biomes = self._define_biomes()
        
        # Generate the map
        self.map_surface = None
        self.height_map = None
        self.moisture_map = None
        self.temp_map = None
        self.biome_map = None
        
        # Create different map view surfaces
        self.biome_surface = None
        self.height_surface = None
        self.moisture_surface = None
        self.temp_surface = None
        self.continent_surface = None  # Added for debugging continent shapes
        
        # Current display mode
        self.current_view = "biome"
        
        # Generate map data
        self.generate_map()
        
        # Generate all map views
        self._generate_map_views()
    
    def _generate_continent_shapes(self):
        """Generate continent placement and shape data"""
        continents = []
        
        # Generate a full coverage map by first creating a grid of influence zones
        grid_size = max(2, min(4, self.continent_count // 2))  # Determine grid based on continent count
        
        # Calculate number of land vs ocean continents
        land_continent_count = max(2, int(self.continent_count * 0.7))  # At least 2 land continents
        ocean_continent_count = self.continent_count - land_continent_count
        
        # Create a grid of possible continent centers to ensure better coverage
        grid_positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Convert grid to spherical coordinates (avoid exact uniform grid for more natural look)
                theta_jitter = random.uniform(-0.1, 0.1)
                phi_jitter = random.uniform(-0.1, 0.1)
                
                theta = (j / grid_size + theta_jitter) * 2 * math.pi  # Longitude with jitter
                phi = (i / grid_size + phi_jitter) * math.pi          # Latitude with jitter
                
                # Only add if not too close to poles (avoid extreme distortion)
                if 0.1 * math.pi < phi < 0.9 * math.pi:
                    grid_positions.append((theta, phi))
        
        # If we need more positions than the grid provides, add random ones
        while len(grid_positions) < self.continent_count:
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0.2 * math.pi, 0.8 * math.pi)
            grid_positions.append((theta, phi))
        
        # Shuffle positions for randomness
        random.shuffle(grid_positions)
        
        # 1. Create land continents
        for i in range(land_continent_count):
            if len(grid_positions) > 0:
                theta, phi = grid_positions.pop()
                
                # Convert to 3D point on unit sphere
                x, y, z = self._spherical_to_cartesian(theta, phi)
                
                # Generate random shape characteristics for this land continent
                # Land continents are larger with more elevation
                base_size = random.uniform(0.18, 0.3)  # Larger size for land continents
                
                # Generate irregular shapes with multiple "growth points"
                growth_points = random.randint(1, 3)  # Number of sub-centers
                secondary_points = []
                
                # Main center is the strongest
                main_strength = random.uniform(0.6, 0.9)
                
                for _ in range(growth_points):
                    # Create secondary growth points around the main one
                    angle_offset = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(0.05, 0.15)  # Close but not too close
                    
                    # Calculate offset position (approximate on sphere)
                    sec_theta = theta + dist * math.cos(angle_offset)
                    sec_phi = phi + dist * math.sin(angle_offset)
                    
                    # Ensure proper wrapping around sphere
                    while sec_theta > 2 * math.pi:
                        sec_theta -= 2 * math.pi
                    while sec_theta < 0:
                        sec_theta += 2 * math.pi
                        
                    # Clamp latitude to valid range
                    sec_phi = max(0.05 * math.pi, min(0.95 * math.pi, sec_phi))
                    
                    # Convert secondary point to cartesian
                    sx, sy, sz = self._spherical_to_cartesian(sec_theta, sec_phi)
                    
                    # Strength of this growth point (relative to main center)
                    strength = random.uniform(0.3, 0.8)
                    
                    secondary_points.append({
                        'pos': (sx, sy, sz),
                        'strength': strength,
                        'size': base_size * random.uniform(0.5, 0.9)
                    })
                
                # Generate unique features for this continent
                shape_scale = random.uniform(0.9, 1.7)  # Shape detail scale
                mountain_chance = random.uniform(0.4, 0.9)  # Higher probability for mountains
                fractal_factor = random.uniform(0.5, 1.0)  # How fractal/irregular the coastline is
                elevation_offset = random.uniform(0.0, 0.3)  # Positive offset to ensure land
                
                # Override noise settings for this specific continent
                continent_specific_seed = random.randint(0, 10000)
                
                continents.append({
                    'type': 'land',
                    'center': (x, y, z),  # 3D coordinates on unit sphere
                    'secondary_points': secondary_points,
                    'main_strength': main_strength,
                    'size': base_size,
                    'shape_scale': shape_scale,
                    'elevation_offset': elevation_offset,
                    'mountain_chance': mountain_chance,
                    'fractal_factor': fractal_factor,
                    'seed': continent_specific_seed
                })
        
        # 2. Create ocean continents (areas with mostly ocean and small islands)
        for i in range(ocean_continent_count):
            if len(grid_positions) > 0:
                theta, phi = grid_positions.pop()
                
                # Convert to 3D point on unit sphere
                x, y, z = self._spherical_to_cartesian(theta, phi)
                
                # Generate random shape characteristics for ocean continent
                # Ocean continents are larger but with negative elevation offset
                base_size = random.uniform(0.2, 0.35)  # Larger coverage area
                
                # Generate islands within this ocean area
                island_count = random.randint(2, 8)
                islands = []
                
                for _ in range(island_count):
                    # Create island positions distributed around the center
                    angle_offset = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(0.05, base_size * 0.8)
                    
                    # Calculate offset position (approximate on sphere)
                    isl_theta = theta + dist * math.cos(angle_offset)
                    isl_phi = phi + dist * math.sin(angle_offset)
                    
                    # Ensure proper wrapping
                    while isl_theta > 2 * math.pi:
                        isl_theta -= 2 * math.pi
                    while isl_theta < 0:
                        isl_theta += 2 * math.pi
                        
                    # Clamp latitude to valid range
                    isl_phi = max(0.05 * math.pi, min(0.95 * math.pi, isl_phi))
                    
                    # Convert island point to cartesian
                    ix, iy, iz = self._spherical_to_cartesian(isl_theta, isl_phi)
                    
                    # Island properties
                    island_size = random.uniform(0.02, 0.08)  # Small islands
                    island_height = random.uniform(0.05, 0.2)  # Height boost
                    
                    islands.append({
                        'pos': (ix, iy, iz),
                        'size': island_size,
                        'height': island_height
                    })
                
                # Ocean basin properties
                shape_scale = random.uniform(0.8, 1.5)
                elevation_offset = random.uniform(-0.25, -0.1)  # Negative offset to ensure ocean
                fractal_factor = random.uniform(0.6, 1.2)  # How varied the ocean floor is
                
                # Override noise settings for this specific ocean region
                ocean_specific_seed = random.randint(0, 10000)
                
                continents.append({
                    'type': 'ocean',
                    'center': (x, y, z),
                    'size': base_size,
                    'islands': islands,
                    'shape_scale': shape_scale,
                    'elevation_offset': elevation_offset,
                    'fractal_factor': fractal_factor,
                    'seed': ocean_specific_seed
                })
        
        return continents
    
    def _noise(self, x, y, z):
        """Generate terrain height using multiple octaves of Perlin noise and continent shapes"""
        # Calculate the influence of each continent on this point
        continent_influence = 0
        base_elevation = 0
        max_weight = 0.01  # Small initial value to avoid division by zero
        
        for continent in self.continents:
            # Common properties
            continent_type = continent.get('type', 'land')  # Default to land for backward compatibility
            cx, cy, cz = continent['center']
            size = continent['size']
            shape_scale = continent['shape_scale']
            elevation_offset = continent['elevation_offset']
            fractal_factor = continent.get('fractal_factor', 0.8)  # Default if not specified
            continent_seed = continent['seed']
            
            # Calculate base distance to continent center
            dot_product = x*cx + y*cy + z*cz
            angle = math.acos(max(-1, min(1, dot_product)))  # Clamp to avoid numerical errors
            
            # Base influence is a sigmoid centered at the continent boundary
            # Clamp the exponent to avoid overflow
            exponent = min(20, max(-20, (angle / size - 1.0) * 10))
            distance_factor = 1.0 / (1.0 + math.exp(exponent))
            
            # For land continents, consider secondary growth points
            if continent_type == 'land' and 'secondary_points' in continent:
                secondary_points = continent['secondary_points']
                main_strength = continent.get('main_strength', 0.7)
                
                # Calculate influence from each secondary point
                secondary_influence = 0
                for point in secondary_points:
                    sx, sy, sz = point['pos']
                    sec_size = point['size']
                    strength = point['strength']
                    
                    # Calculate distance to this secondary point
                    sec_dot = x*sx + y*sy + z*sz
                    sec_angle = math.acos(max(-1, min(1, sec_dot)))
                    
                    # Calculate influence from this secondary point with clamped exponent
                    sec_exponent = min(20, max(-20, (sec_angle / sec_size - 1.0) * 10))
                    sec_factor = 1.0 / (1.0 + math.exp(sec_exponent))
                    secondary_influence += sec_factor * strength
                
                # Combine main center influence with secondary points
                # This creates more irregular, realistic continent shapes
                combined_factor = (distance_factor * main_strength + 
                                   secondary_influence * (1 - main_strength))
                
                distance_factor = combined_factor
            
            # For ocean continents, add islands if any
            if continent_type == 'ocean' and 'islands' in continent:
                island_influence = 0
                islands = continent['islands']
                
                for island in islands:
                    ix, iy, iz = island['pos']
                    island_size = max(0.01, island['size'])  # Ensure non-zero size
                    island_height = island['height']
                    
                    # Calculate distance to island center
                    isl_dot = x*ix + y*iy + z*iz
                    isl_angle = math.acos(max(-1, min(1, isl_dot)))
                    
                    # Islands have a sharper falloff (steeper islands) with clamped exponent
                    isl_exponent = min(20, max(-20, (isl_angle / island_size - 1.0) * 15))
                    isl_factor = 1.0 / (1.0 + math.exp(isl_exponent))
                    
                    # Add this island's influence, scaled by its height
                    island_influence = max(island_influence, isl_factor * island_height)
                
                # In ocean regions, islands override the base ocean depression
                if island_influence > 0.1:
                    distance_factor = island_influence
                    # Override elevation offset for islands
                    elevation_offset = 0.1
            
            if distance_factor > 0.01:  # Only process if this continent has meaningful influence
                # Use continent-specific noise for this region
                continent_noise_gen = PerlinNoise(seed=continent_seed)
                
                # Generate continent-specific noise
                local_scale = shape_scale * self.continent_scale
                
                # Primary continent shape (larger scale)
                shape_noise = (continent_noise_gen.noise(
                    x * local_scale * 0.5, 
                    y * local_scale * 0.5, 
                    z * local_scale * 0.5
                ) + 1) / 2
                
                # Medium detail (medium scale)
                medium_noise = (continent_noise_gen.noise(
                    x * local_scale * 1.5, 
                    y * local_scale * 1.5, 
                    z * local_scale * 1.5
                ) + 1) / 2
                
                # Fine detail (smaller scale)
                detail_noise = (continent_noise_gen.noise(
                    x * local_scale * 3, 
                    y * local_scale * 3, 
                    z * local_scale * 3
                ) + 1) / 2
                
                # Coastal detail (highest frequency)
                coast_noise = (continent_noise_gen.noise(
                    x * local_scale * 6, 
                    y * local_scale * 6, 
                    z * local_scale * 6
                ) + 1) / 2
                
                # Generate fractal noise with multiple octaves
                # Adjust weights based on fractal_factor - higher values make more irregular shapes
                continent_height = (
                    shape_noise * (0.5 - fractal_factor * 0.2) +
                    medium_noise * (0.25 + fractal_factor * 0.1) +
                    detail_noise * (0.15 + fractal_factor * 0.05) +
                    coast_noise * (0.1 + fractal_factor * 0.05)
                )
                
                # Add elevation offset based on continent type
                continent_height += elevation_offset
                
                # For land continents, add mountains if applicable
                if continent_type == 'land':
                    mountain_chance = continent.get('mountain_chance', 0.5)
                    
                    # Mountain noise at medium frequency
                    mountain_noise = (continent_noise_gen.noise(
                        x * local_scale * 2 + 7.7,  # Offset to make different pattern 
                        y * local_scale * 2 + 7.7,
                        z * local_scale * 2 + 7.7
                    ) + 1) / 2
                    
                    # Ridge noise for sharper mountain ranges
                    ridge_noise = (continent_noise_gen.noise(
                        x * local_scale * 4 + 13.5,  # Different offset
                        y * local_scale * 4 + 13.5,
                        z * local_scale * 4 + 13.5
                    ) + 1) / 2
                    
                    # Enhance ridgelines by taking absolute value and squaring
                    ridge_noise = 1 - abs(ridge_noise * 2 - 1) 
                    ridge_noise = ridge_noise * ridge_noise  # Square to enhance peaks
                    
                    # Combine different mountain noises
                    has_mountains = random.random() < mountain_chance
                    if has_mountains:
                        mountain_factor = mountain_noise * 0.3 + ridge_noise * 0.2
                        
                        # Apply mountains only to land (where base height is higher)
                        if continent_height > self.ocean_level - 0.05:
                            continent_height += mountain_factor
                
                # Apply the height using the distance factor as weight
                base_elevation += continent_height * distance_factor
                max_weight += distance_factor
        
        # Normalize the height by the total influence
        if max_weight > 0.01:  # If we had any meaningful continent influence
            base_elevation /= max_weight
            
            # Apply some global noise to add variety
            detail = (self.noise_gen.noise(
                x * self.detail_scale, 
                y * self.detail_scale, 
                z * self.detail_scale
            ) + 1) / 2 * 0.1  # Reduced from 0.15 to make continent shapes more dominant
            
            # Combine the continent elevation with some global detail
            value = base_elevation * 0.9 + detail
            
            # Enhance continent vs ocean contrast
            # Apply sigmoid function to create more defined land/water boundaries
            sigmoid_input = min(20, max(-20, (value - 0.5) * 12))  # Clamp to avoid overflow
            sigmoid = 1.0 / (1.0 + math.exp(-sigmoid_input))
            value = value * 0.8 + sigmoid * 0.2
        else:
            # Deep ocean areas with minimal continent influence
            ocean_noise = (self.noise_gen.noise(
                x * self.detail_scale * 0.5, 
                y * self.detail_scale * 0.5, 
                z * self.detail_scale * 0.5
            ) + 1) / 2
            
            value = ocean_noise * 0.25  # Slightly increased to make ocean floor more varied
        
        # Ensure value is in [0,1] range
        return max(0, min(1, value))
    
    def _define_biomes(self):
        """Define the different biomes based on height, moisture, and temperature"""
        # Colors are in RGB format
        biomes = [
            # Ocean biomes
            Biome("Deep Ocean", (0, 0, 80), 0.0, 0.35, 0.0, 1.0, 0.0, 1.0),
            Biome("Ocean", (0, 0, 150), 0.35, self.ocean_level - 0.02, 0.0, 1.0, 0.0, 1.0),
            Biome("Shallow Ocean", (0, 50, 180), self.ocean_level - 0.02, self.ocean_level, 0.0, 1.0, 0.0, 1.0),
            
            # Cold biomes
            Biome("Snow", (250, 250, 250), self.ocean_level, self.mountain_level, 0.4, 1.0, 0.0, 0.2),
            Biome("Tundra", (180, 180, 220), self.ocean_level, self.mountain_level, 0.0, 0.4, 0.0, 0.2),
            
            # Temperate biomes
            Biome("Grassland", (100, 220, 100), self.ocean_level, 0.55, 0.3, 0.6, 0.3, 0.7),
            Biome("Forest", (34, 140, 34), self.ocean_level, 0.65, 0.6, 1.0, 0.3, 0.7),
            Biome("Seasonal Forest", (73, 100, 35), self.ocean_level, 0.6, 0.4, 0.7, 0.2, 0.4),
            
            # Dry biomes
            Biome("Plains", (180, 200, 80), self.ocean_level, 0.55, 0.2, 0.5, 0.4, 0.7),
            Biome("Desert", (210, 180, 60), self.ocean_level, 0.6, 0.0, 0.3, 0.7, 1.0),
            Biome("Savanna", (180, 150, 50), self.ocean_level, 0.6, 0.3, 0.5, 0.7, 0.9),
            
            # Wet biomes
            Biome("Swamp", (90, 110, 90), self.ocean_level, 0.5, 0.7, 1.0, 0.4, 0.7),
            Biome("Rainforest", (40, 100, 40), self.ocean_level, 0.6, 0.8, 1.0, 0.7, 1.0),
            
            # Mountain biomes
            Biome("Mountains", (110, 110, 110), self.mountain_level, 0.9, 0.0, 1.0, 0.0, 1.0),
            Biome("Snowy Mountains", (200, 200, 230), self.mountain_level, 0.9, 0.5, 1.0, 0.0, 0.3),
            Biome("Rocky Mountains", (140, 120, 100), self.mountain_level, 0.9, 0.0, 0.4, 0.3, 1.0),
            
            # Highest peaks
            Biome("Mountain Peak", (255, 255, 255), 0.9, 1.0, 0.0, 1.0, 0.0, 1.0),
        ]
        return biomes
    
    def _get_biome(self, height, moisture, temperature):
        """Return the biome that matches the given parameters"""
        for biome in self.biomes:
            if biome.matches(height, moisture, temperature):
                return biome
        
        # Default biome if no match found
        if height < 0.35:  # Deep ocean threshold
            return self.biomes[0]  # Deep Ocean
        elif height < self.ocean_level - 0.02:  # Regular ocean threshold
            return self.biomes[1]  # Ocean
        elif height < self.ocean_level:  # Shallow ocean threshold
            return self.biomes[2]  # Shallow Ocean
        elif height < self.mountain_level:
            return self.biomes[5]  # Grassland
        else:
            return self.biomes[13]  # Mountains
    
    def _spherical_to_cartesian(self, theta, phi):
        """Convert spherical coordinates to cartesian
        
        Args:
            theta: Longitude angle (0 to 2π)
            phi: Latitude angle (0 to π, 0 is north pole, π is south pole)
        
        Returns:
            (x, y, z) coordinates on unit sphere
        """
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        return (x, y, z)
    
    def _generate_map_views(self):
        """Generate different map view surfaces"""
        # Biome map (already generated in generate_map)
        self.biome_surface = self.map_surface
        
        # Height map
        self.height_surface = pygame.Surface((self.width, self.height))
        height_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(self.height):
            for x in range(self.width):
                # Get height value
                h = self.height_map[y, x]
                
                # Create grayscale color based on height
                if h < self.ocean_level:
                    # Blue gradient for oceans
                    depth = h / self.ocean_level
                    color = (0, 0, int(100 + 155 * depth))
                else:
                    # Normalized height for land (0-1 range)
                    land_h = (h - self.ocean_level) / (1 - self.ocean_level)
                    value = int(land_h * 255)
                    color = (value, value, value)
                
                height_array[y, x] = color
        
        # Apply height map to surface
        temp_height_array = np.transpose(height_array, (1, 0, 2))
        pygame_surface_array = pygame.surfarray.pixels3d(self.height_surface)
        pygame_surface_array[:] = temp_height_array
        del pygame_surface_array
        
        # Moisture map
        self.moisture_surface = pygame.Surface((self.width, self.height))
        moisture_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(self.height):
            for x in range(self.width):
                # Get moisture value and height (for ocean mask)
                m = self.moisture_map[y, x]
                h = self.height_map[y, x]
                
                if h < self.ocean_level:
                    # Use blue for oceans
                    color = (0, 0, 150)
                else:
                    # Use blue gradient for moisture (dry = brown, wet = blue)
                    blue = int(m * 255)
                    green = int((1 - m) * 100)
                    red = int((1 - m) * 150)
                    color = (red, green, blue)
                
                moisture_array[y, x] = color
        
        # Apply moisture map to surface
        temp_moisture_array = np.transpose(moisture_array, (1, 0, 2))
        pygame_surface_array = pygame.surfarray.pixels3d(self.moisture_surface)
        pygame_surface_array[:] = temp_moisture_array
        del pygame_surface_array
        
        # Temperature map
        self.temp_surface = pygame.Surface((self.width, self.height))
        temp_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(self.height):
            for x in range(self.width):
                # Get temperature value and height (for ocean mask)
                t = self.temp_map[y, x]
                h = self.height_map[y, x]
                
                if h < self.ocean_level:
                    # Use blue for oceans
                    color = (0, 0, 150)
                else:
                    # Use red gradient for temperature (cold = blue, hot = red)
                    red = int(t * 255)
                    blue = int((1 - t) * 255)
                    green = min(red, blue) // 2
                    color = (red, green, blue)
                
                temp_array[y, x] = color
        
        # Apply temperature map to surface
        temp_temp_array = np.transpose(temp_array, (1, 0, 2))
        pygame_surface_array = pygame.surfarray.pixels3d(self.temp_surface)
        pygame_surface_array[:] = temp_temp_array
        del pygame_surface_array
        
        # Generate a continent visualization surface
        self.continent_surface = pygame.Surface((self.width, self.height))
        continent_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Generate a random color for each continent for visualization
        continent_colors = [(random.randint(50, 250), random.randint(50, 250), random.randint(50, 250)) 
                            for _ in range(len(self.continents))]
        
        for y in range(self.height):
            for x in range(self.width):
                # Convert pixel coordinates to spherical coordinates
                theta = (x / self.width) * 2 * math.pi  # Longitude (0 to 2π)
                phi = (y / self.height) * math.pi       # Latitude (0 to π)
                
                # Convert to cartesian coordinates on unit sphere
                px, py, pz = self._spherical_to_cartesian(theta, phi)
                
                # Get height value for ocean masking
                h = self.height_map[y, x]
                
                if h < self.ocean_level:
                    # Use blue gradient for oceans
                    depth = h / self.ocean_level
                    color = (0, 0, int(100 + 155 * depth))
                else:
                    # Find the continent with strongest influence on this point
                    max_influence = 0
                    continent_idx = -1
                    
                    for i, continent in enumerate(self.continents):
                        cx, cy, cz = continent['center']
                        size = continent['size']
                        
                        # Calculate distance (angle) to continent center
                        dot_product = px*cx + py*cy + pz*cz
                        angle = math.acos(max(-1, min(1, dot_product)))
                        
                        # Calculate influence factor
                        influence = 1.0 / (1.0 + math.exp((angle / size - 1.0) * 10))
                        
                        if influence > max_influence:
                            max_influence = influence
                            continent_idx = i
                    
                    if continent_idx >= 0 and max_influence > 0.2:
                        # Use the continent's color
                        color = continent_colors[continent_idx]
                    else:
                        # Low influence areas (could be smaller islands or coastal areas)
                        # Use a gray color
                        gray = int(150 + 105 * h)
                        color = (gray, gray, gray)
                
                continent_array[y, x] = color
        
        # Apply continent visualization to surface
        temp_continent_array = np.transpose(continent_array, (1, 0, 2))
        pygame_surface_array = pygame.surfarray.pixels3d(self.continent_surface)
        pygame_surface_array[:] = temp_continent_array
        del pygame_surface_array
    
    def set_view(self, view_type):
        """Set the current map view"""
        if view_type in ["biome", "height", "moisture", "temperature", "continent"]:
            self.current_view = view_type
            return True
        return False
    
    def draw_map(self, screen, pos=(0, 0), scale=1.0):
        """Draw the map on the given surface at the specified position and scale"""
        # Select the appropriate surface based on the current view
        if self.current_view == "biome":
            surface = self.biome_surface
        elif self.current_view == "height":
            surface = self.height_surface
        elif self.current_view == "moisture":
            surface = self.moisture_surface
        elif self.current_view == "temperature":
            surface = self.temp_surface
        elif self.current_view == "continent":
            surface = self.continent_surface  # New continent view
        else:
            surface = self.biome_surface
        
        # Scale if needed
        if scale != 1.0:
            scaled_width = int(self.width * scale)
            scaled_height = int(self.height * scale)
            scaled_map = pygame.transform.scale(surface, (scaled_width, scaled_height))
            screen.blit(scaled_map, pos)
        else:
            screen.blit(surface, pos)
    
    def generate_map(self):
        """Generate a map of the planet"""
        # Create arrays to store the height, moisture, and temperature values
        self.height_map = np.zeros((self.height, self.width))
        self.moisture_map = np.zeros((self.height, self.width))
        self.temp_map = np.zeros((self.height, self.width))
        self.biome_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Sample height, moisture, and temperature at each point on the map
        for y in range(self.height):
            for x in range(self.width):
                # Convert pixel coordinates to spherical coordinates
                # Equirectangular projection
                theta = (x / self.width) * 2 * math.pi  # Longitude (0 to 2π)
                phi = (y / self.height) * math.pi       # Latitude (0 to π)
                
                # Convert to cartesian coordinates on unit sphere
                px, py, pz = self._spherical_to_cartesian(theta, phi)
                
                # Sample height using noise
                height = self._noise(px, py, pz)
                self.height_map[y, x] = height
                
                # Sample moisture (higher near equator and oceans)
                moisture_base = self.moisture_gen.noise(px * self.moisture_scale, 
                                                      py * self.moisture_scale, 
                                                      pz * self.moisture_scale)
                # Normalize moisture to [0, 1]
                moisture = (moisture_base + 1) / 2
                
                # Boost moisture near equator and reduce at poles
                equator_factor = 1 - abs(phi / math.pi - 0.5) * 2
                moisture = moisture * 0.6 + equator_factor * 0.4
                
                # Boost moisture near oceans
                if height < self.ocean_level + 0.05:
                    ocean_distance = min(1, (self.ocean_level + 0.05 - height) * 10)
                    moisture = moisture * (1 - ocean_distance * 0.3) + ocean_distance * 0.3
                
                self.moisture_map[y, x] = moisture
                
                # Sample temperature (higher near equator, lower at poles and mountains)
                temp_base = self.temp_gen.noise(px * self.temp_scale, 
                                              py * self.temp_scale, 
                                              pz * self.temp_scale) * 0.3
                
                # Base temperature on latitude (equator = hot, poles = cold)
                temp_lat = 1 - abs(phi / math.pi - 0.5) * 2
                
                # Reduce temperature with elevation
                temp_elev = max(0, 1 - max(0, (height - self.ocean_level) * 2))
                
                # Combine factors
                temperature = (temp_base + 1) / 2 * 0.2 + temp_lat * 0.6 + temp_elev * 0.2
                self.temp_map[y, x] = temperature
                
                # Get biome based on height, moisture, and temperature
                biome = self._get_biome(height, moisture, temperature)
                self.biome_map[y, x] = biome.color
        
        # Create the biome surface
        self.map_surface = pygame.Surface((self.width, self.height))
        
        # Apply the biome map to the surface
        temp_array = np.transpose(self.biome_map, (1, 0, 2))
        pygame_surface_array = pygame.surfarray.pixels3d(self.map_surface)
        pygame_surface_array[:] = temp_array
        del pygame_surface_array
    
    def save_map_image(self, filename=None):
        """Save the map as an image file"""
        if filename is None:
            # Generate a filename based on seed, continent count, and timestamp
            timestamp = int(time.time())
            filename = f"planet_map_seed{self.seed}_continents{self.continent_count}_{timestamp}.png"
        
        # Ensure the filename has a proper extension
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            filename += '.png'
        
        # Save the current view surface
        if self.current_view == "biome":
            surface = self.biome_surface
        elif self.current_view == "height":
            surface = self.height_surface
        elif self.current_view == "moisture":
            surface = self.moisture_surface
        elif self.current_view == "temperature":
            surface = self.temp_surface
        elif self.current_view == "continent":
            surface = self.continent_surface
        else:
            surface = self.biome_surface
            
        pygame.image.save(surface, filename)
        print(f"Map saved as: {filename}")
        return filename
    
    def generate_new_map(self, seed=None):
        """Generate a new map with a new seed"""
        self.seed = seed if seed is not None else random.randint(0, 10000)
        
        # Reinitialize noise generators with new seeds
        self.noise_gen = PerlinNoise(seed=self.seed)
        self.moisture_gen = PerlinNoise(seed=self.seed + 1)
        self.temp_gen = PerlinNoise(seed=self.seed + 2)
        self.continent_shape_gen = PerlinNoise(seed=self.seed + 3)
        
        # Generate new map
        self.generate_map()
        
        # Update all map views
        self._generate_map_views()
        
        return self.seed


class Button(UIElement):
    """Button UI element"""
    
    def __init__(self, rect, text, action=None, **kwargs):
        super().__init__(rect, text, **kwargs)
        self.action = action
    
    def handle_event(self, event):
        """Handle events for the button"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_clicked(event.pos) and self.action:
                return self.action()
        return None


class InfoPanel(DraggableUIElement):
    """Panel for displaying information"""
    
    def __init__(self, rect, **kwargs):
        super().__init__(rect, **kwargs)
        self.lines = []
        self.original_font_size = kwargs.get('font_size', 16)
    
    def set_text(self, lines):
        """Set the text lines to display"""
        self.lines = lines
    
    def draw(self, screen):
        """Draw the panel with multiple lines of text"""
        # Draw background and border
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, 2)
        
        # Draw resize handle
        resize_handle = self.get_resize_handle_rect()
        pygame.draw.rect(screen, self.border_color, resize_handle)
        pygame.draw.line(screen, self.text_color, 
                         (resize_handle.left + 5, resize_handle.bottom - 5),
                         (resize_handle.right - 5, resize_handle.bottom - 5), 2)
        pygame.draw.line(screen, self.text_color, 
                         (resize_handle.right - 5, resize_handle.top + 5),
                         (resize_handle.right - 5, resize_handle.bottom - 5), 2)
        
        if not self.lines:
            return
            
        # Calculate available width for text
        available_width = self.rect.width - 16  # Smaller padding on each side
        
        y_offset = 8  # Smaller top padding
        for i, line in enumerate(self.lines):
            # Skip empty lines but add space
            if not line:
                y_offset += self.font_size // 3  # Smaller empty line space
                continue
                
            # Check if we need to wrap or truncate text
            text_surface = self.font.render(line, True, self.text_color)
            if text_surface.get_width() > available_width:
                # Try to fit with smaller font if text is too long
                smaller_font = pygame.font.SysFont("Arial", max(6, self.font_size - 1))
                text_surface = smaller_font.render(line, True, self.text_color)
                
                if text_surface.get_width() > available_width:
                    # Still too long, truncate with ellipsis
                    for length in range(len(line) - 1, 0, -1):
                        truncated = line[:length] + "..."
                        text_surface = smaller_font.render(truncated, True, self.text_color)
                        if text_surface.get_width() <= available_width:
                            break
            
            # Check if we're out of vertical space
            if y_offset + text_surface.get_height() > self.rect.height - 8:
                # Draw ellipsis to indicate more content
                ellipsis = self.font.render("...", True, self.text_color)
                ellipsis_rect = ellipsis.get_rect(topleft=(self.rect.x + 8, y_offset))
                screen.blit(ellipsis, ellipsis_rect)
                break
                
            text_rect = text_surface.get_rect(topleft=(self.rect.x + 8, self.rect.y + y_offset))
            screen.blit(text_surface, text_rect)
            
            # Adjust spacing based on importance (headers get more space)
            if i == 0 or line.startswith("-"):
                y_offset += text_surface.get_height() + 3  # Smaller header spacing
            else:
                y_offset += text_surface.get_height() + 1  # Smaller line spacing


class TextInput(UIElement):
    """Text input field UI element"""
    
    def __init__(self, rect, placeholder="", max_length=20, **kwargs):
        super().__init__(rect, "", **kwargs)
        self.placeholder = placeholder
        self.max_length = max_length
        self.text = ""
        self.active = False
        self.cursor_pos = 0
        self.cursor_visible = True
        self.cursor_timer = 0
        self.cursor_blink_time = 500  # milliseconds
    
    def draw(self, screen):
        """Draw the text input field"""
        # Draw background
        bg_color = self.bg_color
        if self.active:
            # Brighten background when active
            bg_color = tuple(min(c + 30, 255) for c in self.bg_color[:3]) + (self.bg_color[3],) if len(self.bg_color) > 3 else tuple(min(c + 30, 255) for c in self.bg_color)
        
        pygame.draw.rect(screen, bg_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, 2)
        
        # Draw text or placeholder
        text_to_render = self.text if self.text else self.placeholder
        color = self.text_color if self.text else (150, 150, 150)  # Gray for placeholder
        
        if text_to_render:
            # Check if text fits within the input field
            text_width = self.font.size(text_to_render)[0]
            available_width = self.rect.width - 20  # 10px padding on each side
            
            if text_width > available_width:
                # Calculate visible portion of text based on cursor position
                if self.cursor_pos == len(self.text):  # Cursor at end
                    # Show as much text from the end as fits
                    visible_text = text_to_render
                    while len(visible_text) > 0 and self.font.size(visible_text)[0] > available_width:
                        visible_text = visible_text[1:]
                else:
                    # Center around cursor position
                    left_pos = self.cursor_pos
                    right_pos = self.cursor_pos
                    
                    # Try to show an equal amount on each side of cursor
                    while left_pos > 0 or right_pos < len(text_to_render):
                        test_text = text_to_render[max(0, left_pos):min(len(text_to_render), right_pos)]
                        if self.font.size(test_text)[0] > available_width:
                            break
                        
                        if left_pos > 0:
                            left_pos -= 1
                        if right_pos < len(text_to_render):
                            right_pos += 1
                    
                    visible_text = text_to_render[max(0, left_pos+1):min(len(text_to_render), right_pos)]
                
                # Indicate text truncation if needed
                if visible_text != text_to_render:
                    if visible_text != text_to_render[-len(visible_text):]:
                        visible_text = "..." + visible_text[3:] if len(visible_text) > 3 else visible_text
                    if self.cursor_pos < len(self.text) - 3:
                        visible_text = visible_text[:-3] + "..." if len(visible_text) > 3 else visible_text
            else:
                visible_text = text_to_render
            
            text_surface = self.font.render(visible_text, True, color)
            text_rect = text_surface.get_rect(midleft=(self.rect.x + 10, self.rect.centery))
            screen.blit(text_surface, text_rect)
        
        # Draw cursor if active
        if self.active and self.cursor_visible:
            # Calculate cursor position
            cursor_x = self.rect.x + 10
            
            if self.text:
                # Adjust for visible text portion
                visible_text = self.text[:self.cursor_pos]
                text_width = self.font.size(visible_text)[0]
                available_width = self.rect.width - 20
                
                if text_width <= available_width:
                    cursor_x += text_width
                else:
                    # Position cursor at end of visible text
                    cursor_x = self.rect.x + 10 + min(available_width, self.font.size(visible_text[-10:])[0])
            
            cursor_y_top = self.rect.centery - self.font_size // 2 + 1  # Adjusted for smaller font
            cursor_y_bottom = self.rect.centery + self.font_size // 2 - 1  # Adjusted for smaller font
            
            pygame.draw.line(screen, self.text_color, 
                            (cursor_x, cursor_y_top), 
                            (cursor_x, cursor_y_bottom), 1)  # Thinner cursor
    
    def handle_event(self, event):
        """Handle events for the text input"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle active state when clicked
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            
            if self.active != was_active:
                # Reset cursor blink timer when activation changes
                self.cursor_timer = pygame.time.get_ticks()
                self.cursor_visible = True
            
            return self.active
        
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                # Submit input
                self.active = False
                return "submit"
            
            elif event.key == pygame.K_BACKSPACE:
                # Handle backspace
                if self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
            
            elif event.key == pygame.K_DELETE:
                # Handle delete
                if self.cursor_pos < len(self.text):
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
            
            elif event.key == pygame.K_LEFT:
                # Move cursor left
                self.cursor_pos = max(0, self.cursor_pos - 1)
            
            elif event.key == pygame.K_RIGHT:
                # Move cursor right
                self.cursor_pos = min(len(self.text), self.cursor_pos + 1)
            
            elif event.key == pygame.K_HOME:
                # Move cursor to start
                self.cursor_pos = 0
            
            elif event.key == pygame.K_END:
                # Move cursor to end
                self.cursor_pos = len(self.text)
            
            elif event.unicode and event.unicode.isprintable():
                # Add character at cursor position if max length not reached
                if len(self.text) < self.max_length:
                    self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                    self.cursor_pos += 1
            
            # Reset cursor blink timer on any keypress
            self.cursor_timer = pygame.time.get_ticks()
            self.cursor_visible = True
            
            return True
        
        return False
    
    def update(self, current_time):
        """Update the cursor blinking"""
        if self.active and current_time - self.cursor_timer > self.cursor_blink_time:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = current_time
    
    def get_text(self):
        """Get the current text value"""
        return self.text
    
    def set_text(self, text):
        """Set the text and cursor position"""
        self.text = text
        self.cursor_pos = len(text)


def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Procedural Planet Map Generator")
    clock = pygame.time.Clock()
    
    # Create font for text - reduced by 40%
    font = pygame.font.SysFont("Arial", 12)  # Reduced from 20
    
    # Create UI elements
    # Bottom controls
    btn_generate = Button(
        rect=(50, SCREEN_HEIGHT - 70, 200, 50),
        text="Generate New Map",
        font_size=12  # Reduced from 20
    )
    
    btn_save = Button(
        rect=(270, SCREEN_HEIGHT - 70, 200, 50),
        text="Save Map Image",
        font_size=12  # Reduced from 20
    )
    
    btn_legend = Button(
        rect=(490, SCREEN_HEIGHT - 70, 200, 50),
        text="Toggle Legend",
        font_size=12  # Reduced from 20
    )
    
    # Add map view buttons - moved to the right side of the screen
    view_button_width = 160
    view_button_height = 40
    view_button_x = SCREEN_WIDTH - view_button_width - 30  # Position on the right side
    
    btn_biome = Button(
        rect=(view_button_x, 100, view_button_width, view_button_height),
        text="Biome View",
        font_size=11  # Reduced from 18
    )
    
    btn_height = Button(
        rect=(view_button_x, 100 + view_button_height + 15, view_button_width, view_button_height),
        text="Height View",
        font_size=11  # Reduced from 18
    )
    
    btn_moisture = Button(
        rect=(view_button_x, 100 + (view_button_height + 15) * 2, view_button_width, view_button_height),
        text="Moisture View",
        font_size=11  # Reduced from 18
    )
    
    btn_temp = Button(
        rect=(view_button_x, 100 + (view_button_height + 15) * 3, view_button_width, view_button_height),
        text="Temperature View",
        font_size=11  # Reduced from 18
    )
    
    # Add continent view button
    btn_continent = Button(
        rect=(view_button_x, 100 + (view_button_height + 15) * 4, view_button_width, view_button_height),
        text="Continent View",
        font_size=11
    )
    
    # Add seed input and button
    seed_label = UIElement(
        rect=(650, SCREEN_HEIGHT - 100, 100, 30),
        text="Seed:",
        font_size=11,  # Reduced from 18
        bg_color=COLOR_BACKGROUND,
        border_color=COLOR_BACKGROUND
    )
    
    seed_input = TextInput(
        rect=(650, SCREEN_HEIGHT - 70, 180, 50),
        placeholder="Enter seed...",
        font_size=11,  # Reduced from 18
        max_length=10
    )
    
    # Add continent count input
    continent_label = UIElement(
        rect=(view_button_x, 100 + (view_button_height + 15) * 5 + 20, 100, 30),
        text="Continents:",
        font_size=11,
        bg_color=COLOR_BACKGROUND,
        border_color=COLOR_BACKGROUND
    )
    
    continent_input = TextInput(
        rect=(view_button_x, 100 + (view_button_height + 15) * 5 + 50, 160, 40),
        placeholder="3-8",
        font_size=11,
        max_length=2
    )
    
    btn_set_seed = Button(
        rect=(840, SCREEN_HEIGHT - 70, 140, 50),
        text="Set Seed",
        font_size=11  # Reduced from 18
    )
    
    # Create draggable info panel
    info_panel = InfoPanel(
        rect=(SCREEN_WIDTH - 300, 20, 280, 220),
        font_size=11  # Reduced from 18
    )
    
    # Create map generator with default 5 continents
    map_generator = PlanetMapGenerator(continent_count=5)
    
    # Update seed input with current seed
    seed_input.set_text(str(map_generator.seed))
    
    # Update continent count input with current value
    continent_input.set_text(str(map_generator.continent_count))
    
    # Create the biome legend in the bottom left - now draggable
    legend = BiomeLegend(
        rect=(20, SCREEN_HEIGHT - 360, 250, 280),
        biomes=map_generator.biomes,
        font_size=10  # Reduced from 16
    )
    
    # Set button actions
    def generate_new_map():
        # Get continent count from input
        continent_count = 5  # Default
        try:
            input_value = int(continent_input.get_text())
            if 2 <= input_value <= 15:  # Reasonable range
                continent_count = input_value
            else:
                print(f"Using default continent count (5) instead of {input_value}")
        except (ValueError, TypeError):
            pass  # Use default
            
        # Create a new generator with the specified continent count
        new_generator = PlanetMapGenerator(continent_count=continent_count)
        
        # Update the legend with new biomes
        legend.biomes = new_generator.biomes
        
        # Return the seed for display
        seed_input.set_text(str(new_generator.seed))
        print(f"Generated new map with seed: {new_generator.seed} and {continent_count} continents")
        
        return new_generator, new_generator.seed
    
    def save_map():
        filename = map_generator.save_map_image()
        print(f"Saved map as: {filename}")
        return filename
    
    def toggle_legend():
        is_visible = legend.toggle_visibility()
        return "shown" if is_visible else "hidden"
    
    def set_biome_view():
        map_generator.set_view("biome")
        return "biome"
    
    def set_height_view():
        map_generator.set_view("height")
        return "height"
    
    def set_moisture_view():
        map_generator.set_view("moisture")
        return "moisture"
    
    def set_temp_view():
        map_generator.set_view("temperature")
        return "temperature"
    
    def set_continent_view():
        map_generator.set_view("continent")
        return "continent"
    
    def set_seed():
        seed_text = seed_input.get_text()
        try:
            # Try to convert the seed to an integer
            seed = int(seed_text)
            
            # Get continent count from input
            continent_count = 5  # Default
            try:
                input_value = int(continent_input.get_text())
                if 2 <= input_value <= 15:  # Reasonable range
                    continent_count = input_value
                else:
                    print(f"Using default continent count (5) instead of {input_value}")
            except (ValueError, TypeError):
                pass  # Use default
                
            # Create a new generator with the specified seed and continent count
            new_generator = PlanetMapGenerator(seed=seed, continent_count=continent_count)
            
            # Update the legend with new biomes
            legend.biomes = new_generator.biomes
            
            print(f"Set seed to: {new_generator.seed} with {continent_count} continents")
            return new_generator, new_generator.seed
        except ValueError:
            print(f"Invalid seed: {seed_text}. Using random seed instead.")
            
            # Still respect continent count
            continent_count = 5  # Default
            try:
                input_value = int(continent_input.get_text())
                if 2 <= input_value <= 15:
                    continent_count = input_value
            except (ValueError, TypeError):
                pass
                
            new_generator = PlanetMapGenerator(continent_count=continent_count)
            
            # Update the legend with new biomes
            legend.biomes = new_generator.biomes
            
            seed_input.set_text(str(new_generator.seed))
            return new_generator, new_generator.seed
    
    btn_generate.action = generate_new_map
    btn_save.action = save_map
    btn_legend.action = toggle_legend
    btn_biome.action = set_biome_view
    btn_height.action = set_height_view
    btn_moisture.action = set_moisture_view
    btn_temp.action = set_temp_view
    btn_continent.action = set_continent_view
    btn_set_seed.action = set_seed
    
    # Main loop
    running = True
    
    # Set initial info panel
    info_panel.set_text([
        f"Seed: {map_generator.seed}",
        f"Continents: {map_generator.continent_count}",
        "Controls:",
        "- Generate New Map: Create a new map",
        "- Save Map Image: Save map as PNG",
        "- Toggle Legend: Show/hide biome legend",
        "- View buttons: Change map display",
        "- Set Seed: Use a specific seed",
        "",
        "Drag panels to reposition them",
        "Drag corner to resize panels",
        "",
        f"Current view: Biome"
    ])
    
    # Print instructions
    print("Procedural Planet Map Generator")
    print(f"Initial seed: {map_generator.seed}")
    print(f"Initial continents: {map_generator.continent_count}")
    print("Controls:")
    print("  - Click 'Generate New Map' for a new random planet")
    print("  - Click 'Save Map Image' to save the current map as PNG")
    print("  - Enter a seed number and click 'Set Seed' to use a specific seed")
    print("  - Set the number of continents (2-15)")
    print("  - Use the view buttons to switch between different map displays")
    print("  - You can drag and resize panels by their corners")
    
    # Add padding around the map to prevent UI overlap
    map_padding = 20
    
    # Calculate map position to center it with space for UI
    map_scale = min(
        (SCREEN_WIDTH - 400) / map_generator.width,  # Reduced width to make room for UI on sides
        (SCREEN_HEIGHT - 160) / map_generator.height  # Reduced height to make room for UI on top/bottom
    )
    map_width = int(map_generator.width * map_scale)
    map_height = int(map_generator.height * map_scale)
    map_x = (SCREEN_WIDTH - map_width - 180) // 2  # Offset to the left to make room for right panel
    map_y = (SCREEN_HEIGHT - 100 - map_height) // 2  # Keep vertical centering with bottom controls
    
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle draggable UI elements first
            if info_panel.handle_event(event):
                continue  # Skip other event processing if the panel was interacted with
                
            if legend.handle_event(event):
                continue  # Skip other event processing if the legend was interacted with
            
            # Handle text input events
            seed_input.handle_event(event)
            continent_input.handle_event(event)
            
            # Handle button events
            if event.type == pygame.MOUSEBUTTONDOWN:
                result = btn_generate.handle_event(event)
                if result:
                    # Unpack the return value
                    map_generator, seed = result
                    
                    info_panel.set_text([
                        f"Seed: {seed}",
                        f"Continents: {map_generator.continent_count}",
                        "Controls:",
                        "- Generate New Map: Create a new map",
                        "- Save Map Image: Save map as PNG",
                        "- Toggle Legend: Show/hide biome legend",
                        "- View buttons: Change map display",
                        "- Set Seed: Use a specific seed",
                        "",
                        "Drag panels to reposition them",
                        "Drag corner to resize panels",
                        "",
                        f"Current view: {map_generator.current_view.capitalize()}"
                    ])
                
                filename = btn_save.handle_event(event)
                if filename:
                    info_panel.set_text([
                        f"Seed: {map_generator.seed}",
                        f"Continents: {map_generator.continent_count}",
                        f"Saved map as: {filename}",
                        "",
                        "Controls:",
                        "- Generate New Map: Create a new map",
                        "- Save Map Image: Save map as PNG",
                        "- Toggle Legend: Show/hide biome legend",
                        "- View buttons: Change map display",
                        "- Set Seed: Use a specific seed",
                        "",
                        "Drag panels to reposition them",
                        "Drag corner to resize panels"
                    ])
                
                legend_state = btn_legend.handle_event(event)
                if legend_state:
                    info_panel.set_text([
                        f"Seed: {map_generator.seed}",
                        f"Continents: {map_generator.continent_count}",
                        f"Legend {legend_state}",
                        "",
                        "Controls:",
                        "- Generate New Map: Create a new map",
                        "- Save Map Image: Save map as PNG",
                        "- Toggle Legend: Show/hide biome legend",
                        "- View buttons: Change map display",
                        "- Set Seed: Use a specific seed",
                        "",
                        "Drag panels to reposition them",
                        "Drag corner to resize panels"
                    ])
                
                # Handle seed setting
                result = btn_set_seed.handle_event(event)
                if result:
                    # Unpack the return value
                    map_generator, seed = result
                    
                    info_panel.set_text([
                        f"Seed: {seed}",
                        f"Continents: {map_generator.continent_count}",
                        "Set custom seed",
                        "",
                        "Controls:",
                        "- Generate New Map: Create a new map",
                        "- Save Map Image: Save map as PNG",
                        "- Toggle Legend: Show/hide biome legend",
                        "- View buttons: Change map display",
                        "- Set Seed: Use a specific seed",
                        "",
                        "Drag panels to reposition them",
                        "Drag corner to resize panels"
                    ])
                
                # Handle view switching buttons
                view = None
                if btn_biome.handle_event(event):
                    view = "Biome"
                elif btn_height.handle_event(event):
                    view = "Height"
                elif btn_moisture.handle_event(event):
                    view = "Moisture"
                elif btn_temp.handle_event(event):
                    view = "Temperature"
                elif btn_continent.handle_event(event):
                    view = "Continent"
                
                if view:
                    info_panel.set_text([
                        f"Seed: {map_generator.seed}",
                        f"Continents: {map_generator.continent_count}",
                        f"Switched to {view} view",
                        "",
                        "Controls:",
                        "- Generate New Map: Create a new map",
                        "- Save Map Image: Save map as PNG",
                        "- Toggle Legend: Show/hide biome legend",
                        "- View buttons: Change map display",
                        "- Set Seed: Use a specific seed",
                        "",
                        "Drag panels to reposition them",
                        "Drag corner to resize panels"
                    ])
        
        # Update text input cursor
        seed_input.update(current_time)
        continent_input.update(current_time)
        
        # Clear screen
        screen.fill(COLOR_BACKGROUND)
        
        # Draw map
        map_generator.draw_map(screen, (map_x, map_y), map_scale)
        
        # Draw UI elements
        btn_generate.draw(screen)
        btn_save.draw(screen)
        btn_legend.draw(screen)
        btn_biome.draw(screen)
        btn_height.draw(screen)
        btn_moisture.draw(screen)
        btn_temp.draw(screen)
        btn_continent.draw(screen)
        seed_label.draw(screen)
        seed_input.draw(screen)
        continent_label.draw(screen)
        continent_input.draw(screen)
        btn_set_seed.draw(screen)
        
        # Draw title
        title_text = font.render("Procedural Planet Map Generator", True, COLOR_UI_TEXT)
        screen.blit(title_text, (20, 20))
        
        # Draw map border
        pygame.draw.rect(screen, COLOR_UI_BORDER, 
                         (map_x - 2, map_y - 2, map_width + 4, map_height + 4), 2)
        
        # Draw draggable elements last so they appear on top
        info_panel.draw(screen)
        legend.draw(screen)
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":
    main() 