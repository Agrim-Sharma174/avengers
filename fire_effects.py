import cv2
import numpy as np
import random
import math

class FireEffect:
    def __init__(self, width, height):
        """Initialize fire effect generator."""
        self.width = width
        self.height = height
        
        # Generate base fire image
        self.base_fire_img = self.generate_fire_image(100, 100)
        self.small_fire_img = cv2.resize(self.base_fire_img, (50, 50))
        
    def generate_fire_image(self, width, height):
        """Generate a realistic fire image with alpha channel."""
        # Create a transparent background
        img = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Create a more realistic fire shape
        # Base (orange)
        cv2.ellipse(img, (width//2, int(height*0.65)), (int(width*0.3), int(height*0.4)), 
                   0, 0, 360, (0, 69, 255, 255), -1)
        
        # Middle (red)
        cv2.ellipse(img, (width//2, int(height*0.5)), (int(width*0.2), int(height*0.35)), 
                   0, 0, 360, (0, 0, 255, 255), -1)
        
        # Top (yellow)
        cv2.ellipse(img, (width//2, int(height*0.4)), (int(width*0.1), int(height*0.25)), 
                   0, 0, 360, (0, 165, 255, 255), -1)
        
        # Add some noise and variation
        for i in range(20):
            x = random.randint(width//3, width*2//3)
            y = random.randint(height//3, height*2//3)
            radius = random.randint(5, 15)
            color = (
                0,  # B
                random.randint(150, 200),  # G
                255,  # R
                random.randint(100, 200)  # A
            )
            cv2.circle(img, (x, y), radius, color, -1)
        
        return img
        
    def overlay_image(self, background, foreground, x, y, scale=1.0, rotation=0):
        """Overlay foreground image with alpha channel on background."""
        # Scale image if needed
        if scale != 1.0:
            foreground = cv2.resize(foreground, (0, 0), fx=scale, fy=scale)
        
        # Rotate image if needed
        if rotation != 0:
            # Get the image center
            h, w = foreground.shape[:2]
            center = (w//2, h//2)
            
            # Calculate the rotation matrix
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            
            # Perform the rotation
            foreground = cv2.warpAffine(foreground, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        
        # Get dimensions
        fg_h, fg_w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Calculate position to center the image at (x,y)
        x = int(x - fg_w/2)
        y = int(y - fg_h/2)
        
        # Ensure coordinates are within frame
        if x >= bg_w or y >= bg_h or x + fg_w <= 0 or y + fg_h <= 0:
            return background
        
        # Clip the overlay to be within the background bounds
        x_offset = max(0, -x)
        y_offset = max(0, -y)
        x = max(0, x)
        y = max(0, y)
        
        # Determine dimensions of overlay
        w = min(fg_w - x_offset, bg_w - x)
        h = min(fg_h - y_offset, bg_h - y)
        
        if h <= 0 or w <= 0:
            return background
        
        # Extract alpha channel and create masks
        alpha = foreground[y_offset:y_offset+h, x_offset:x_offset+w, 3] / 255.0
        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        
        # Extract RGB channels
        foreground_rgb = foreground[y_offset:y_offset+h, x_offset:x_offset+w, 0:3]
        
        # Blend images
        cropped_bg = background[y:y+h, x:x+w]
        blended = cv2.multiply(1.0 - alpha, cropped_bg.astype(float)) + cv2.multiply(alpha, foreground_rgb.astype(float))
        background[y:y+h, x:x+w] = blended.astype(np.uint8)
        
        return background
    
    def create_small_fire(self, frame_count=0):
        """Create a small fire effect with animation."""
        # Apply subtle animation based on frame count
        scale = 1.0 + 0.1 * math.sin(frame_count * 0.1)
        rotation = 5 * math.sin(frame_count * 0.05)
        
        # Get base image
        fire = self.small_fire_img.copy()
        
        return fire, scale, rotation
    
    def create_thrown_fire(self, start_pos, target_pos, progress, intensity=1.0):
        """Create a fire effect that animates from start to target position."""
        # Calculate current position
        current_x = int(start_pos[0] + (target_pos[0] - start_pos[0]) * progress)
        current_y = int(start_pos[1] + (target_pos[1] - start_pos[1]) * progress)
        
        # Scale based on distance traveled and intensity
        scale = 0.5 + progress * (1.0 + intensity)
        
        # Rotation for dynamic effect
        rotation = progress * 360  # Full rotation during travel
        
        return (current_x, current_y, scale, rotation)
    
    def create_fire_particle_system(self, center_x, center_y, num_particles=20, spread=100, intensity=1.0):
        """Create a system of fire particles radiating from a center point."""
        particles = []
        
        for _ in range(num_particles):
            # Random direction
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(50, spread)
            
            # Target position
            target_x = center_x + distance * math.cos(angle)
            target_y = center_y + distance * math.sin(angle)
            
            # Random speed variation
            speed = random.uniform(0.8, 1.2)
            
            # Random size variation
            size = random.uniform(0.8, 1.2) * intensity
            
            particles.append({
                'start': (center_x, center_y),
                'target': (target_x, target_y),
                'speed': speed,
                'size': size
            })
            
        return particles 