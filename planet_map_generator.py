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
                ocean_level=0.45,  # Increased from 0.35 to create much larger oceans
                mountain_level=0.75,
                width=MAP_WIDTH,
                height=MAP_HEIGHT):
        """
        Initialize the map generator
        
        Args:
            seed: Random seed for noise generation
            ocean_level: Value below which is considered ocean (0-1)
            mountain_level: Value above which is considered mountains (0-1)
            width: Width of the map in pixels
            height: Height of the map in pixels
        """
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.ocean_level = ocean_level
        self.mountain_level = mountain_level
        
        # Initialize noise generators
        self.noise_gen = PerlinNoise(seed=self.seed)
        self.moisture_gen = PerlinNoise(seed=self.seed + 1)
        self.temp_gen = PerlinNoise(seed=self.seed + 2)
        
        # Set different scales for various terrain features
        self.continent_scale = 0.6  # Decreased from 1.0 to create larger continent features
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
        
        # Current display mode
        self.current_view = "biome"
        
        # Generate map data
        self.generate_map()
        
        # Generate all map views
        self._generate_map_views()
    
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
    
    def _noise(self, x, y, z):
        """Generate terrain height using multiple octaves of Perlin noise"""
        # Multiple octaves at different frequencies for more natural terrain
        continent = self.noise_gen.noise(x * self.continent_scale, 
                                        y * self.continent_scale, 
                                        z * self.continent_scale)
        
        # Enhance continent formation by applying a threshold
        # This creates more defined continent edges rather than gradual slopes
        continent_threshold = 0.1  # Controls how "binary" the continent vs ocean division is
        continent = (continent + 1) / 2  # Normalize to [0, 1]
        continent = max(0, min(1, (continent - 0.5 + continent_threshold) / (continent_threshold * 2)))
        
        mountains = self.noise_gen.noise(x * self.mountain_scale, 
                                        y * self.mountain_scale, 
                                        z * self.mountain_scale) * 0.5
        
        detail = self.noise_gen.noise(x * self.detail_scale, 
                                     y * self.detail_scale, 
                                     z * self.detail_scale) * 0.25
        
        # Combine octaves
        value = continent * 0.7 + mountains * 0.2 + detail * 0.1
        
        # Normalize to [0, 1]
        return value
    
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
    
    def set_view(self, view_type):
        """Set the current map view"""
        if view_type in ["biome", "height", "moisture", "temperature"]:
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
            # Generate a filename based on seed and timestamp
            timestamp = int(time.time())
            filename = f"planet_map_seed{self.seed}_{timestamp}.png"
        
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
    
    # Create map generator
    map_generator = PlanetMapGenerator()
    
    # Update seed input with current seed
    seed_input.set_text(str(map_generator.seed))
    
    # Create the biome legend in the bottom left - now draggable
    legend = BiomeLegend(
        rect=(20, SCREEN_HEIGHT - 360, 250, 280),
        biomes=map_generator.biomes,
        font_size=10  # Reduced from 16
    )
    
    # Set button actions
    def generate_new_map():
        new_seed = map_generator.generate_new_map()
        seed_input.set_text(str(new_seed))
        print(f"Generated new map with seed: {new_seed}")
        return new_seed
    
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
    
    def set_seed():
        seed_text = seed_input.get_text()
        try:
            # Try to convert the seed to an integer
            seed = int(seed_text)
            new_seed = map_generator.generate_new_map(seed)
            print(f"Set seed to: {new_seed}")
            return new_seed
        except ValueError:
            print(f"Invalid seed: {seed_text}. Using random seed instead.")
            new_seed = map_generator.generate_new_map()
            seed_input.set_text(str(new_seed))
            return new_seed
    
    btn_generate.action = generate_new_map
    btn_save.action = save_map
    btn_legend.action = toggle_legend
    btn_biome.action = set_biome_view
    btn_height.action = set_height_view
    btn_moisture.action = set_moisture_view
    btn_temp.action = set_temp_view
    btn_set_seed.action = set_seed
    
    # Main loop
    running = True
    
    # Set initial info panel
    info_panel.set_text([
        f"Seed: {map_generator.seed}",
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
    print("Controls:")
    print("  - Click 'Generate New Map' for a new random planet")
    print("  - Click 'Save Map Image' to save the current map as PNG")
    print("  - Enter a seed number and click 'Set Seed' to use a specific seed")
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
            
            # Handle button events
            if event.type == pygame.MOUSEBUTTONDOWN:
                result = btn_generate.handle_event(event)
                if result:
                    info_panel.set_text([
                        f"Seed: {result}",
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
                    info_panel.set_text([
                        f"Seed: {result}",
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
                
                if view:
                    info_panel.set_text([
                        f"Seed: {map_generator.seed}",
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
        seed_label.draw(screen)
        seed_input.draw(screen)
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