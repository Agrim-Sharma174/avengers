import cv2
import numpy as np
import time
import math
import random
import os

class SimpleFireThrower:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera. Make sure your webcam is connected and not being used by another application.")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create assets directory if it doesn't exist
        os.makedirs("assets", exist_ok=True)
        
        # Create fire image
        self.fire_img = self.create_fire_image(100, 100)
        self.small_fire_img = cv2.resize(self.fire_img, (50, 50))
        
        # Store fire locations
        self.fire_locations = []
        
        # Frame counter for animation
        self.frame_count = 0
        
        # Mouse position for creating fires
        self.mouse_x = 0
        self.mouse_y = 0
        
        # State variables for collecting and throwing
        self.collecting_fire = False
        self.throwing_fire = False
        self.throw_start_time = 0
        self.throw_duration = 1.5  # seconds
        
        # Animation parameters
        self.collected_fires = []
        self.thrown_fires = []
        
        print("Simple Fire Thrower initialized. Using mouse controls instead of hand gestures.")
        print("Controls:")
        print("  - Left Click: Create fire")
        print("  - Right Click: Collect fires")
        print("  - Middle Click: Throw collected fires")
        print("  - Press 'q' to quit")
    
    def create_fire_image(self, width, height):
        """Generate a fire image with alpha channel."""
        # Create a transparent background
        img = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Create a fire shape
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
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_x = x
        self.mouse_y = y
        
        # Left click: create fire
        if event == cv2.EVENT_LBUTTONDOWN and not self.collecting_fire and not self.throwing_fire:
            self.fire_locations.append((x, y, 1.0, 0.0, self.frame_count))
            print(f"Created fire at ({x}, {y})")
        
        # Right click: collect fires
        elif event == cv2.EVENT_RBUTTONDOWN and not self.collecting_fire and not self.throwing_fire and len(self.fire_locations) > 0:
            self.collecting_fire = True
            self.collected_fires = self.fire_locations.copy()
            self.fire_locations = []
            print("Collecting fires!")
        
        # Middle click: throw collected fires
        elif event == cv2.EVENT_MBUTTONDOWN and self.collecting_fire and not self.throwing_fire:
            self.collecting_fire = False
            self.throwing_fire = True
            self.throw_start_time = time.time()
            
            # Create thrown fires with start and target positions
            self.thrown_fires = self.create_fire_particle_system(
                x, y, num_particles=30, spread=300, intensity=1.5)
            
            print("Throwing fire!")
    
    def run(self):
        # Set up mouse callback
        cv2.namedWindow("Simple Fire Thrower")
        cv2.setMouseCallback("Simple Fire Thrower", self.mouse_callback)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Increment frame counter
            self.frame_count += 1
            
            # Draw fires that have already been placed
            for i, loc in enumerate(self.fire_locations):
                x, y, scale, rotation, _ = loc
                
                # Apply subtle animation based on frame count
                current_scale = scale * (1.0 + 0.1 * math.sin(self.frame_count * 0.1 + i * 0.5))
                current_rotation = rotation + 5 * math.sin(self.frame_count * 0.05 + i * 0.5)
                
                # Draw the fire
                frame = self.overlay_image(frame, self.small_fire_img, x, y, current_scale, current_rotation)
            
            # If we're in throwing animation mode
            if self.throwing_fire:
                elapsed = time.time() - self.throw_start_time
                progress = min(elapsed / self.throw_duration, 1.0)
                
                if progress >= 1.0:
                    self.throwing_fire = False
                    self.thrown_fires = []
                else:
                    # Animate thrown fires
                    for fire in self.thrown_fires:
                        start_pos = fire['start']
                        target_pos = fire['target']
                        speed = fire['speed']
                        size = fire['size']
                        
                        # Calculate current fire position and attributes
                        fire_progress = min(elapsed * speed / self.throw_duration, 1.0)
                        current_x = int(start_pos[0] + (target_pos[0] - start_pos[0]) * fire_progress)
                        current_y = int(start_pos[1] + (target_pos[1] - start_pos[1]) * fire_progress)
                        
                        # Scale based on distance traveled and intensity
                        scale = 0.5 + fire_progress * (1.0 + size)
                        
                        # Rotation for dynamic effect
                        rotation = fire_progress * 360  # Full rotation during travel
                        
                        # Draw the fire
                        frame = self.overlay_image(frame, self.small_fire_img, current_x, current_y, scale, rotation)
            
            # Display instructions
            cv2.putText(frame, "Left Click: Create fire", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Right Click: Collect fires", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Middle Click: Throw fire", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show fire counter
            cv2.putText(frame, f"Fires: {len(self.fire_locations)}", (self.width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Simple Fire Thrower", frame)
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        fire_thrower = SimpleFireThrower()
        fire_thrower.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Try installing OpenCV with: pip install opencv-python")
        input("Press Enter to exit...") 