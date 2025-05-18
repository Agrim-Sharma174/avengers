import cv2
import numpy as np
import random
import math
import os
import urllib.request
from pathlib import Path

class HeroEffect:
    def __init__(self, width, height):
        """Initialize hero effect generator."""
        self.width = width
        self.height = height
        
        # Create assets directory if it doesn't exist
        self.assets_dir = Path("assets")
        self.assets_dir.mkdir(exist_ok=True)
        
        # Create hero-specific directories
        self.heroes_dir = self.assets_dir / "heroes"
        self.heroes_dir.mkdir(exist_ok=True)
        
        # Initialize hero assets with unique power behaviors
        self.hero_assets = {
            "iron_man": {
                "name": "Iron Man",
                "color": (0, 0, 255),  # Red
                "effect_name": "Repulsor Beam",
                "mask_path": "assets/heroes/iron_man/mask.png",
                "effect_path": "assets/heroes/iron_man/effect.png",
                "effect_image": None,
                "mask_image": None,
                "power_type": "beam",  # Continuous beam
                "power_speed": 2.0,    # Speed increased for easier gameplay
                "power_damage": 3,     # Damage increased
                "power_color": (0, 200, 255)  # Blue-white
            },
            "spider_man": {
                "name": "Spider-Man", 
                "color": (0, 0, 255),  # Red
                "effect_name": "Web Shooter",
                "mask_path": "assets/heroes/spider_man/mask.png",
                "effect_path": "assets/heroes/spider_man/effect.png",
                "effect_image": None,
                "mask_image": None,
                "power_type": "web",  # Web that sticks to targets
                "power_speed": 1.8,   # Speed increased
                "power_damage": 2,    # Damage increased
                "power_color": (255, 255, 255)  # White
            },
            "thor": {
                "name": "Thor",
                "color": (0, 215, 255),  # Yellow
                "effect_name": "Lightning",
                "mask_path": "assets/heroes/thor/mask.png",
                "effect_path": "assets/heroes/thor/effect.png",
                "effect_image": None,
                "mask_image": None,
                "power_type": "lightning",  # Lightning that chains between targets
                "power_speed": 2.2,        # Speed increased
                "power_damage": 4,         # Damage increased
                "power_color": (255, 255, 100)  # Yellowish
            },
            "hulk": {
                "name": "Hulk",
                "color": (0, 255, 0),  # Green
                "effect_name": "Smash",
                "mask_path": "assets/heroes/hulk/mask.png",
                "effect_path": "assets/heroes/hulk/effect.png",
                "effect_image": None,
                "mask_image": None,
                "power_type": "smash",  # Area of effect damage
                "power_speed": 1.5,     # Speed increased
                "power_damage": 5,      # Damage increased
                "power_color": (0, 255, 0)  # Green
            },
            "captain_america": {
                "name": "Captain America",
                "color": (255, 0, 0),  # Blue
                "effect_name": "Shield Throw",
                "mask_path": "assets/heroes/captain_america/mask.png",
                "effect_path": "assets/heroes/captain_america/effect.png",
                "effect_image": None,
                "mask_image": None,
                "power_type": "shield",  # Shield that bounces between targets
                "power_speed": 1.8,      # Speed increased
                "power_damage": 3,       # Damage increased
                "power_color": (255, 0, 0)  # Blue
            }
        }
        
        # Load hero masks
        self.load_hero_masks()
        
        # For compatibility with the old HeroEffects class in avengers_game.py
        self.hero_assets_compat = {
            "Iron Man": {
                "power_color": (30, 90, 255),  # Red-orange
                "power_size": 25,
                "attack_speed": 20,  # Increased from 15
                "power_effect": "repulsor",
            },
            "Spider-Man": {
                "power_color": (0, 0, 255),  # Red
                "power_size": 15,
                "attack_speed": 25,  # Increased from 20
                "power_effect": "web",
            },
            "Thor": {
                "power_color": (255, 215, 0),  # Lightning
                "power_size": 35,
                "attack_speed": 16,  # Increased from 12
                "power_effect": "lightning",
            },
            "Hulk": {
                "power_color": (0, 200, 0),  # Green
                "power_size": 40,
                "attack_speed": 14,  # Increased from 10
                "power_effect": "smash",
            },
            "Captain America": {
                "power_color": (255, 0, 0),  # Blue
                "power_size": 30,
                "attack_speed": 22,  # Increased from 17
                "power_effect": "shield",
            }
        }
        
        # Load hero images and generate effects
        self.load_hero_assets()
        
        # Generate base effect images for each hero
        self.generate_hero_effects()

    def download_image(self, url, save_path):
        """Download an image from URL if it doesn't exist."""
        if not os.path.exists(save_path):
            try:
                print(f"Downloading {url} to {save_path}...")
                urllib.request.urlretrieve(url, save_path)
                return True
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                return False
        return True

    def load_hero_assets(self):
        """Load hero images and masks."""
        for hero_id, hero_data in self.hero_assets.items():
            # Load mask image
            mask_path = hero_data["mask_path"]
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_img is not None:
                    # If image is jpg (no alpha), add alpha channel
                    if len(mask_img.shape) == 3 and mask_img.shape[2] == 3:
                        alpha = np.ones((mask_img.shape[0], mask_img.shape[1], 1), dtype=mask_img.dtype) * 255
                        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2BGRA)
                    hero_data["mask_image"] = mask_img
                else:
                    print(f"Warning: Could not load mask image for {hero_id}")
            else:
                print(f"Warning: Mask image not found for {hero_id} at {mask_path}")
            
            # Load effect image
            effect_path = hero_data["effect_path"]
            if os.path.exists(effect_path):
                effect_img = cv2.imread(effect_path, cv2.IMREAD_UNCHANGED)
                if effect_img is not None:
                    # If image is jpg (no alpha), add alpha channel
                    if len(effect_img.shape) == 3 and effect_img.shape[2] == 3:
                        alpha = np.ones((effect_img.shape[0], effect_img.shape[1], 1), dtype=effect_img.dtype) * 255
                        effect_img = cv2.cvtColor(effect_img, cv2.COLOR_BGR2BGRA)
                    hero_data["effect_image"] = effect_img
                else:
                    print(f"Warning: Could not load effect image for {hero_id}")
            else:
                print(f"Warning: Effect image not found for {hero_id} at {effect_path}")
                
            # If either image failed to load, generate a placeholder
            if hero_data["mask_image"] is None:
                self.generate_hero_mask_image(hero_id, mask_path)
                hero_data["mask_image"] = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            if hero_data["effect_image"] is None:
                self.generate_hero_effect_image(hero_id, effect_path)
                hero_data["effect_image"] = cv2.imread(effect_path, cv2.IMREAD_UNCHANGED)

    def generate_hero_mask_image(self, hero_id, save_path):
        """Generate a placeholder mask image for a hero."""
        hero_data = self.hero_assets[hero_id]
        mask = np.zeros((200, 200, 4), dtype=np.uint8)
        
        # Draw simple face shape
        color = hero_data["color"][::-1] + (255,)  # Convert BGR to RGBA
        
        # Face outline
        cv2.ellipse(mask, (100, 100), (80, 100), 0, 0, 360, color, -1)
        
        # Add text with hero name
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(hero_data["name"], font, 0.6, 2)[0]
        text_x = (mask.shape[1] - text_size[0]) // 2
        cv2.putText(mask, hero_data["name"], (text_x, 160), font, 0.6, (255, 255, 255, 255), 2)
        
        cv2.imwrite(save_path, mask)

    def generate_hero_effect_image(self, hero_id, save_path):
        """Generate a placeholder effect image for a hero."""
        hero_data = self.hero_assets[hero_id]
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        
        color = hero_data["color"][::-1] + (255,)  # Convert BGR to RGBA
        power_color = hero_data["power_color"][::-1] + (255,)  # Convert BGR to RGBA
        
        if hero_id == "iron_man":
            # Iron Man repulsor beam
            cv2.circle(img, (50, 50), 30, (255, 255, 255, 200), -1)
            cv2.circle(img, (50, 50), 20, power_color, -1)
            # Add beam effect
            cv2.ellipse(img, (50, 50), (40, 15), 0, 0, 360, (255, 255, 255, 120), -1)
        
        elif hero_id == "spider_man":
            # Spider-Man web
            # Web center
            cv2.circle(img, (50, 50), 15, power_color, -1)
            # Web strands
            for i in range(8):
                angle = i * math.pi / 4
                end_x = int(50 + 45 * math.cos(angle))
                end_y = int(50 + 45 * math.sin(angle))
                cv2.line(img, (50, 50), (end_x, end_y), power_color, 2)
                
                # Add small connecting strands
                if i % 2 == 0:
                    mid_x = int(50 + 25 * math.cos(angle))
                    mid_y = int(50 + 25 * math.sin(angle))
                    perp_angle = angle + math.pi / 2
                    side1_x = int(mid_x + 8 * math.cos(perp_angle))
                    side1_y = int(mid_y + 8 * math.sin(perp_angle))
                    side2_x = int(mid_x - 8 * math.cos(perp_angle))
                    side2_y = int(mid_y - 8 * math.sin(perp_angle))
                    cv2.line(img, (mid_x, mid_y), (side1_x, side1_y), power_color, 1)
                    cv2.line(img, (mid_x, mid_y), (side2_x, side2_y), power_color, 1)
        
        elif hero_id == "thor":
            # Thor lightning
            # Lightning bolt
            start_x, start_y = 50, 10
            points = [(start_x, start_y)]
            x, y = start_x, start_y
            
            # Generate lightning path
            while y < 90:
                x += random.randint(-10, 10)
                y += random.randint(10, 15)
                x = max(10, min(90, x))  # Keep within bounds
                points.append((x, y))
            
            # Draw the main lightning bolt
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i+1], power_color, 3)
                
            # Add some branches
            for i in range(1, len(points) - 1, 2):
                if random.random() > 0.5:
                    branch_x = points[i][0] + random.randint(-20, 20)
                    branch_y = points[i][1] + random.randint(-5, 15)
                    branch_x = max(0, min(99, branch_x))
                    branch_y = max(0, min(99, branch_y))
                    cv2.line(img, points[i], (branch_x, branch_y), power_color, 2)
            
            # Add glow
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i+1], (255, 255, 255, 100), 6)
        
        elif hero_id == "hulk":
            # Hulk smash
            # Fist shape
            cv2.rectangle(img, (30, 30), (70, 70), power_color, -1)
            
            # Impact lines
            for i in range(12):
                angle = i * math.pi / 6
                length = random.randint(30, 45)
                end_x = int(50 + length * math.cos(angle))
                end_y = int(50 + length * math.sin(angle))
                cv2.line(img, (50, 50), (end_x, end_y), (255, 255, 255, 180), 2)
            
            # Impact center
            cv2.circle(img, (50, 50), 20, (255, 255, 255, 150), -1)
            cv2.circle(img, (50, 50), 10, power_color, -1)
        
        elif hero_id == "captain_america":
            # Captain America shield
            # Outer ring - blue
            cv2.circle(img, (50, 50), 40, (255, 0, 0, 220), -1)
            # Middle ring - red and white
            cv2.circle(img, (50, 50), 30, (0, 0, 255, 220), -1)
            cv2.circle(img, (50, 50), 20, (255, 255, 255, 220), -1)
            # Inner ring - red
            cv2.circle(img, (50, 50), 10, (0, 0, 255, 220), -1)
            # Center - star shape (simplified as a circle)
            cv2.circle(img, (50, 50), 5, (255, 255, 255, 255), -1)
            
            # Add motion blur
            angle = 30  # Degrees
            M = cv2.getRotationMatrix2D((50, 50), angle, 1)
            motion_blur = cv2.warpAffine(img.copy(), M, (100, 100))
            img = cv2.addWeighted(img, 0.7, motion_blur, 0.3, 0)
        
        cv2.imwrite(save_path, img)

    def generate_hero_effects(self):
        """Generate effect images for each hero."""
        # The effects are already loaded in load_hero_assets
        pass
        
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
    
    def create_hero_effect(self, hero_id, frame_count=0):
        """Create a hero-specific effect with animation."""
        # Apply subtle animation based on frame count
        scale = 1.0 + 0.1 * math.sin(frame_count * 0.1)
        rotation = 5 * math.sin(frame_count * 0.05)
        
        # Get base image for the hero
        effect = self.hero_assets[hero_id]["effect_image"].copy()
        
        return effect, scale, rotation
    
    def create_targeted_effect(self, hero_id, start_pos, target_pos, progress, intensity=1.0, frame_count=0):
        """Create a hero-specific effect that animates from start to target position."""
        # Get hero data
        if hero_id in ["Iron Man", "iron_man"]:
            hero_id = "iron_man"
        elif hero_id in ["Spider-Man", "spider_man"]:
            hero_id = "spider_man"
        elif hero_id in ["Thor", "thor"]:
            hero_id = "thor"
        elif hero_id in ["Hulk", "hulk"]:
            hero_id = "hulk"
        elif hero_id in ["Captain America", "captain_america"]:
            hero_id = "captain_america"
        
        hero_data = self.hero_assets.get(hero_id, self.hero_assets["iron_man"])
        effect_image = hero_data["effect_image"]
        
        if effect_image is None:
            return None
        
        # Create a temporary frame for drawing the effect
        effect_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calculate current position
        start_x, start_y = int(start_pos[0]), int(start_pos[1])
        target_x, target_y = int(target_pos[0]), int(target_pos[1])
        
        # Calculate the position along the path based on progress
        current_x = int(start_x + (target_x - start_x) * progress)
        current_y = int(start_y + (target_y - start_y) * progress)
        
        # Calculate angle for rotation
        angle = math.atan2(target_y - start_y, target_x - start_x) * 180 / math.pi
        
        # Scale effect based on intensity
        scale = intensity * (1.0 + 0.2 * math.sin(frame_count * 0.1))  # Add pulsing effect
        
        # Resize and rotate effect image
        effect_size = max(50, int(min(self.width, self.height) * 0.1 * scale))
        resized_effect = cv2.resize(effect_image, (effect_size, effect_size))
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D((effect_size//2, effect_size//2), angle, 1.0)
        rotated_effect = cv2.warpAffine(resized_effect, M, (effect_size, effect_size))
        
        # Calculate position to place effect
        effect_x = current_x - effect_size//2
        effect_y = current_y - effect_size//2
        
        # Create mask for the effect
        if rotated_effect.shape[2] == 4:  # If image has alpha channel
            alpha = rotated_effect[:, :, 3] / 255.0
            alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            effect_rgb = rotated_effect[:, :, :3]
        else:
            alpha = np.ones((rotated_effect.shape[0], rotated_effect.shape[1], 3))
            effect_rgb = rotated_effect
        
        # Calculate valid region for overlay
        x1 = max(0, effect_x)
        y1 = max(0, effect_y)
        x2 = min(self.width, effect_x + effect_size)
        y2 = min(self.height, effect_y + effect_size)
        
        # Calculate effect image region
        ex1 = max(0, -effect_x)
        ey1 = max(0, -effect_y)
        ex2 = min(effect_size, self.width - effect_x)
        ey2 = min(effect_size, self.height - effect_y)
        
        # Only proceed if we have valid regions
        if x2 > x1 and y2 > y1 and ex2 > ex1 and ey2 > ey1:
            effect_region = effect_rgb[ey1:ey2, ex1:ex2]
            alpha_region = alpha[ey1:ey2, ex1:ex2]
            
            # Create region in effect frame
            effect_frame[y1:y2, x1:x2] = (effect_region * alpha_region).astype(np.uint8)
        
        # Add trail effect based on hero type
        if hero_data["power_type"] in ["beam", "lightning"]:
            cv2.line(effect_frame, (start_x, start_y), (current_x, current_y),
                    hero_data["power_color"], 2, cv2.LINE_AA)
        elif hero_data["power_type"] == "web":
            # Draw web strand
            cv2.line(effect_frame, (start_x, start_y), (current_x, current_y),
                    (255, 255, 255), 1, cv2.LINE_AA)
        
        return effect_frame
    
    def create_effect_particle_system(self, hero_id, center_x, center_y, targets=None, num_particles=20, spread=100, intensity=1.0):
        """Create a system of particles radiating from a center point or targeting specific enemies."""
        particles = []
        hero_data = self.hero_assets[hero_id]
        
        # If targets provided, aim powers at them
        if targets and len(targets) > 0:
            # Sort targets by distance to find closest ones
            sorted_targets = sorted(targets, 
                                   key=lambda t: (t['x'] - center_x)**2 + (t['y'] - center_y)**2)
            
            # Limit to reasonable number of targets
            max_targets = min(len(sorted_targets), 5)
            
            for i in range(max_targets):
                target = sorted_targets[i]
                
                # Add some randomness to avoid all powers hitting the same point
                target_x = target['x'] + random.randint(-target['size']//2, target['size']//2)
                target_y = target['y'] + random.randint(-target['size']//2, target['size']//2)
                
                # Create particles aimed at this target
                count = max(1, num_particles // max_targets)
                for _ in range(count):
                    # Add slight variation to start position
                    start_x = center_x + random.randint(-10, 10)
                    start_y = center_y + random.randint(-10, 10)
                    
                    # Random speed variation
                    speed_factor = random.uniform(0.8, 1.2) * hero_data["power_speed"]
                    
                    # Random size variation
                    size = random.uniform(0.8, 1.2) * intensity
                    
                    particles.append({
                        'hero_id': hero_id,
                        'start': (start_x, start_y),
                        'target': (target_x, target_y),
                        'target_object': target,  # Store reference to actual target
                        'speed': speed_factor,
                        'size': size,
                        'power_type': hero_data["power_type"],
                        'damage': hero_data["power_damage"]
                    })
        else:
            # Fallback to original spread pattern if no targets
            for _ in range(num_particles):
                # Random direction
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(50, spread)
                
                # Target position
                target_x = center_x + distance * math.cos(angle)
                target_y = center_y + distance * math.sin(angle)
                
                # Random speed variation
                speed = random.uniform(0.8, 1.2) * hero_data["power_speed"]
                
                # Random size variation
                size = random.uniform(0.8, 1.2) * intensity
                
                particles.append({
                    'hero_id': hero_id,
                    'start': (center_x, center_y),
                    'target': (target_x, target_y),
                    'target_object': None,
                    'speed': speed,
                    'size': size,
                    'power_type': hero_data["power_type"],
                    'damage': hero_data["power_damage"]
                })
            
        return particles
        
    def apply_face_mask(self, frame, hero_id, faces):
        """Apply hero mask to detected faces."""
        mask_img = self.hero_assets[hero_id]["mask_image"]
        
        if mask_img is None or len(faces) == 0:
            return frame
            
        for (x, y, w, h) in faces:
            # Resize mask to fit face
            resized_mask = cv2.resize(mask_img, (w, h))
            
            # Apply mask
            frame = self.overlay_image(frame, resized_mask, x + w//2, y + h//2)
            
        return frame
    
    def get_hero_power(self, hero_name):
        """Compatibility method for the old HeroEffects class."""
        return self.hero_assets_compat.get(hero_name, self.hero_assets_compat["Iron Man"])

    def load_hero_masks(self):
        """Load hero mask images."""
        self.hero_masks = {}
        
        # Load each hero's mask
        mask_files = {
            "iron_man": "assets/heroes/iron_man.png",
            "spider_man": "assets/heroes/spider_man.jpg",
            "thor": "assets/heroes/thor.jpg",
            "hulk": "assets/heroes/hulk.jpg",
            "captain_america": "assets/heroes/captain_america.png"
        }
        
        for hero_id, mask_path in mask_files.items():
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    # If image is jpg (no alpha), add alpha channel
                    if len(mask.shape) == 3 and mask.shape[2] == 3:
                        alpha = np.ones((mask.shape[0], mask.shape[1], 1), dtype=mask.dtype) * 255
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
                    self.hero_masks[hero_id] = mask
            except Exception as e:
                print(f"Error loading mask for {hero_id}: {e}")

    def apply_hero_mask(self, frame, hero_id, face_x, face_y, face_w, face_h):
        """Apply hero mask to a face."""
        if hero_id not in self.hero_masks:
            return frame
            
        mask = self.hero_masks[hero_id]
        if mask is None:
            return frame
            
        try:
            # Add some padding around the face for better mask fit
            pad_w = int(face_w * 0.2)
            pad_h = int(face_h * 0.2)
            mask_w = face_w + 2 * pad_w
            mask_h = face_h + 2 * pad_h
            
            # Resize mask while maintaining aspect ratio
            mask_aspect = mask.shape[1] / mask.shape[0]
            face_aspect = mask_w / mask_h
            
            if mask_aspect > face_aspect:
                # Mask is wider than face
                new_w = mask_w
                new_h = int(new_w / mask_aspect)
            else:
                # Mask is taller than face
                new_h = mask_h
                new_w = int(new_h * mask_aspect)
            
            # Ensure minimum dimensions
            new_w = max(new_w, 1)
            new_h = max(new_h, 1)
            
            # Resize mask
            resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Calculate position to center mask on face
            x_offset = face_x - pad_w + (mask_w - new_w) // 2
            y_offset = face_y - pad_h + (mask_h - new_h) // 2
            
            # Create a region of interest (ROI) in the frame
            roi_y1 = max(0, y_offset)
            roi_y2 = min(frame.shape[0], y_offset + new_h)
            roi_x1 = max(0, x_offset)
            roi_x2 = min(frame.shape[1], x_offset + new_w)
            
            # Get the region of the mask that will be visible
            mask_y1 = max(0, -y_offset)
            mask_y2 = min(resized_mask.shape[0], mask_y1 + (roi_y2 - roi_y1))
            mask_x1 = max(0, -x_offset)
            mask_x2 = min(resized_mask.shape[1], mask_x1 + (roi_x2 - roi_x1))
            
            # Adjust ROI dimensions to match mask region
            roi_y2 = roi_y1 + (mask_y2 - mask_y1)
            roi_x2 = roi_x1 + (mask_x2 - mask_x1)
            
            if roi_y2 > roi_y1 and roi_x2 > roi_x1 and mask_y2 > mask_y1 and mask_x2 > mask_x1:
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                mask_roi = resized_mask[mask_y1:mask_y2, mask_x1:mask_x2]
                
                # Ensure ROI and mask_roi have the same dimensions
                if roi.shape[:2] == mask_roi.shape[:2]:
                    # Convert mask to BGRA if it's BGR
                    if mask_roi.shape[2] == 3:
                        mask_roi = cv2.cvtColor(mask_roi, cv2.COLOR_BGR2BGRA)
                    
                    # Ensure frame ROI has 3 channels
                    if len(roi.shape) == 2:
                        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    
                    # Create alpha channels
                    alpha_mask = mask_roi[:, :, 3:4].astype(float) / 255.0
                    alpha_frame = 1.0 - alpha_mask
                    
                    # Blend images
                    for c in range(3):  # Process each color channel separately
                        roi[:, :, c] = (alpha_frame[:, :, 0] * roi[:, :, c] + 
                                      alpha_mask[:, :, 0] * mask_roi[:, :, c]).astype(np.uint8)
                    
                    frame[roi_y1:roi_y2, roi_x1:roi_x2] = roi
            
        except Exception as e:
            print(f"Error applying mask: {e}")
            import traceback
            traceback.print_exc()
            
        return frame 