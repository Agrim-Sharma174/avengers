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
                "mask_url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
                "effect_url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
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
                "mask_url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
                "effect_url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
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
                "mask_url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
                "effect_url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
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
                "mask_url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
                "effect_url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
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
                "mask_url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
                "effect_url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
                "effect_image": None,
                "mask_image": None,
                "power_type": "shield",  # Shield that bounces between targets
                "power_speed": 1.8,      # Speed increased
                "power_damage": 3,       # Damage increased
                "power_color": (255, 0, 0)  # Blue
            }
        }
        
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
        # For now, generate placeholder images for each hero
        # In a real implementation, you'd download actual assets
        for hero_id, hero_data in self.hero_assets.items():
            hero_dir = self.heroes_dir / hero_id
            hero_dir.mkdir(exist_ok=True)
            
            # Create placeholder effect image if not downloaded
            effect_path = hero_dir / "effect.png"
            if not effect_path.exists():
                self.generate_hero_effect_image(hero_id, str(effect_path))
            hero_data["effect_image"] = cv2.imread(str(effect_path), cv2.IMREAD_UNCHANGED)
            
            # Create placeholder mask image if not downloaded
            mask_path = hero_dir / "mask.png"
            if not mask_path.exists():
                self.generate_hero_mask_image(hero_id, str(mask_path))
            hero_data["mask_image"] = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

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
        power_type = hero_data["power_type"]
        power_color = hero_data["power_color"]
        
        # Create a temporary frame for drawing the effect
        effect_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calculate current position
        start_x, start_y = int(start_pos[0]), int(start_pos[1])
        target_x, target_y = int(target_pos[0]), int(target_pos[1])
        
        # Add jitter for more dynamic effects
        jitter_x = int(random.uniform(-5, 5) * intensity)
        jitter_y = int(random.uniform(-5, 5) * intensity)
        
        if power_type == "beam":  # Iron Man - bright repulsor beam
            # Draw a line from start to target
            cv2.line(effect_frame, (start_x, start_y), (target_x, target_y), power_color, 
                    thickness=int(12 * intensity), lineType=cv2.LINE_AA)
            
            # Add a glow effect
            cv2.line(effect_frame, (start_x, start_y), (target_x, target_y), 
                    (255, 255, 255), thickness=int(6 * intensity), lineType=cv2.LINE_AA)
            
            # Add impact point at target
            cv2.circle(effect_frame, (target_x + jitter_x, target_y + jitter_y), 
                     int(20 * intensity), power_color, -1)
            cv2.circle(effect_frame, (target_x, target_y), 
                     int(10 * intensity), (255, 255, 255), -1)
            
            # Add expanding rings at impact
            for i in range(3):
                radius = int((15 + i * 10) * intensity * (0.5 + 0.5 * progress))
                cv2.circle(effect_frame, (target_x, target_y), radius, 
                         power_color, 1, cv2.LINE_AA)
            
        elif power_type == "web":  # Spider-Man - web line with sticky effect
            # Calculate positions along the path
            current_x = int(start_x + (target_x - start_x) * progress)
            current_y = int(start_y + (target_y - start_y) * progress)
            
            # Draw the main web line
            cv2.line(effect_frame, (start_x, start_y), (current_x, current_y), 
                    power_color, thickness=int(4 * intensity), lineType=cv2.LINE_AA)
            
            # Add small web branches along the line
            line_length = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
            num_branches = int(line_length / 15)
            
            for i in range(num_branches):
                t = i / num_branches
                branch_x = int(start_x + (target_x - start_x) * t)
                branch_y = int(start_y + (target_y - start_y) * t)
                
                # Only draw branches that are within the current progress
                if t <= progress:
                    angle = math.atan2(target_y - start_y, target_x - start_x) + math.pi/2
                    length = random.randint(5, 15) * intensity
                    
                    # Draw two branches perpendicular to the web line
                    end1_x = int(branch_x + length * math.cos(angle))
                    end1_y = int(branch_y + length * math.sin(angle))
                    end2_x = int(branch_x - length * math.cos(angle))
                    end2_y = int(branch_y - length * math.sin(angle))
                    
                    cv2.line(effect_frame, (branch_x, branch_y), (end1_x, end1_y), 
                            power_color, thickness=1, lineType=cv2.LINE_AA)
                    cv2.line(effect_frame, (branch_x, branch_y), (end2_x, end2_y), 
                            power_color, thickness=1, lineType=cv2.LINE_AA)
            
            # Add an impact effect at the end - enhanced web pattern
            if progress > 0.7:
                # Draw a web pattern at the target
                web_size = int(25 * intensity)
                # Outer circle
                cv2.circle(effect_frame, (target_x, target_y), web_size, 
                         power_color, thickness=1, lineType=cv2.LINE_AA)
                
                # Web strands
                for i in range(8):
                    angle = i * math.pi / 4
                    end_x = int(target_x + web_size * math.cos(angle))
                    end_y = int(target_y + web_size * math.sin(angle))
                    cv2.line(effect_frame, (target_x, target_y), (end_x, end_y), 
                            power_color, thickness=1, lineType=cv2.LINE_AA)
                    
                    # Add connecting strands
                    for j in range(2):
                        distance = web_size * (j+1) / 3
                        mid_x = int(target_x + distance * math.cos(angle))
                        mid_y = int(target_y + distance * math.sin(angle))
                        
                        next_angle = (i + 1) % 8 * math.pi / 4
                        next_x = int(target_x + distance * math.cos(next_angle))
                        next_y = int(target_y + distance * math.sin(next_angle))
                        
                        cv2.line(effect_frame, (mid_x, mid_y), (next_x, next_y), 
                               power_color, thickness=1, lineType=cv2.LINE_AA)
        
        elif power_type == "lightning":  # Thor - Enhanced lightning with more branches
            # Generate lightning path
            points = [(start_x, start_y)]
            
            # Calculate direct distance and angle
            total_dist = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
            angle = math.atan2(target_y - start_y, target_x - start_x)
            
            num_segments = max(6, int(total_dist / 25))
            
            for i in range(1, num_segments):
                # Calculate ideal position along the direct path
                ideal_x = start_x + (target_x - start_x) * (i / num_segments)
                ideal_y = start_y + (target_y - start_y) * (i / num_segments)
                
                # Add some randomness perpendicular to the path
                perp_angle = angle + math.pi/2
                offset = random.randint(-20, 20)
                
                point_x = int(ideal_x + offset * math.cos(perp_angle))
                point_y = int(ideal_y + offset * math.sin(perp_angle))
                
                points.append((point_x, point_y))
            
            # Final point is the target
            points.append((target_x, target_y))
            
            # Draw lightning segments based on progress
            num_visible = max(2, int(len(points) * progress))
            
            # Draw all segments up to the current progress
            for i in range(num_visible - 1):
                # Main lightning line - wider for more visibility
                cv2.line(effect_frame, points[i], points[i+1], power_color, 
                       thickness=int(5 * intensity), lineType=cv2.LINE_AA)
                
                # Add glow effect
                cv2.line(effect_frame, points[i], points[i+1], (255, 255, 255), 
                       thickness=int(3 * intensity), lineType=cv2.LINE_AA)
                
                # Add more branching lightning
                if random.random() < 0.4 and i > 0:  # Increased probability
                    branch_len = random.randint(15, 30)
                    branch_angle = angle + random.uniform(-math.pi/3, math.pi/3)
                    branch_x = int(points[i][0] + branch_len * math.cos(branch_angle))
                    branch_y = int(points[i][1] + branch_len * math.sin(branch_angle))
                    
                    cv2.line(effect_frame, points[i], (branch_x, branch_y), power_color, 
                           thickness=int(3 * intensity), lineType=cv2.LINE_AA)
                    
                    # Add sub-branches
                    if random.random() < 0.5:
                        sub_branch_len = random.randint(5, 15)
                        sub_branch_angle = branch_angle + random.uniform(-math.pi/4, math.pi/4)
                        sub_branch_x = int(branch_x + sub_branch_len * math.cos(sub_branch_angle))
                        sub_branch_y = int(branch_y + sub_branch_len * math.sin(sub_branch_angle))
                        
                        cv2.line(effect_frame, (branch_x, branch_y), 
                               (sub_branch_x, sub_branch_y), power_color, 
                               thickness=int(2 * intensity), lineType=cv2.LINE_AA)
            
            # Add impact at target if progress is near complete
            if progress > 0.8:
                # Expanding circles
                for i in range(3):
                    radius = int((10 + i * 10) * intensity * progress)
                    cv2.circle(effect_frame, (target_x, target_y), radius, 
                             power_color, thickness=1, lineType=cv2.LINE_AA)
                
                # Bright center
                cv2.circle(effect_frame, (target_x, target_y), int(15 * intensity), 
                         (255, 255, 255), -1)
                cv2.circle(effect_frame, (target_x, target_y), int(10 * intensity), 
                         power_color, -1)
        
        elif power_type == "smash":  # Hulk - Enhanced area effect smash
            # Calculate progress along path
            current_x = int(start_x + (target_x - start_x) * progress)
            current_y = int(start_y + (target_y - start_y) * progress)
            
            # Draw traveling fist effect
            if progress < 0.7:
                # Fist becoming larger as it approaches target
                size = int(25 + 25 * progress)
                cv2.rectangle(effect_frame, 
                           (current_x - size//2, current_y - size//2),
                           (current_x + size//2, current_y + size//2),
                           power_color, -1)
                
                # Add motion blur
                prev_x = int(start_x + (target_x - start_x) * max(0, progress - 0.1))
                prev_y = int(start_y + (target_y - start_y) * max(0, progress - 0.1))
                prev_size = int(25 + 25 * max(0, progress - 0.1))
                
                cv2.rectangle(effect_frame, 
                           (prev_x - prev_size//2, prev_y - prev_size//2),
                           (prev_x + prev_size//2, prev_y + prev_size//2),
                           power_color, 1)
            else:
                # Impact effect at target
                impact_size = int(60 * intensity)
                
                # Green shockwave
                for r in range(4):
                    radius = int(impact_size * (0.5 + r * 0.3) * (progress - 0.7) / 0.3)
                    thickness = int(6 * (1.0 - r * 0.2))
                    cv2.circle(effect_frame, (target_x, target_y), radius, 
                             power_color, thickness=thickness, lineType=cv2.LINE_AA)
                
                # Impact cracks - more and thicker
                for i in range(10):
                    angle = i * math.pi / 5 + frame_count * 0.01
                    length = impact_size * 1.5 * (progress - 0.7) / 0.3
                    end_x = int(target_x + length * math.cos(angle))
                    end_y = int(target_y + length * math.sin(angle))
                    
                    thickness = int(4 * intensity * (1.0 - (progress - 0.7) * 0.5))
                    cv2.line(effect_frame, (target_x, target_y), (end_x, end_y), 
                           power_color, thickness=thickness, lineType=cv2.LINE_AA)
                
                # Debris particles
                for _ in range(15):
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(radius/2, radius)
                    particle_x = int(target_x + distance * math.cos(angle))
                    particle_y = int(target_y + distance * math.sin(angle))
                    particle_size = random.randint(2, 5)
                    
                    cv2.circle(effect_frame, (particle_x, particle_y), particle_size, 
                             power_color, -1)
        
        elif power_type == "shield":  # Captain America - Enhanced shield with star
            # Calculate current position along curved path
            t = progress
            # Add a slight curve to the path
            x_offset = (target_x - start_x) * 0.3 * math.sin(math.pi * t)
            y_offset = 0
            
            current_x = int(start_x + (target_x - start_x) * t + x_offset)
            current_y = int(start_y + (target_y - start_y) * t + y_offset)
            
            # Shield size
            shield_size = int(30 * intensity)
            
            # Create shield
            # Outer circle (blue)
            cv2.circle(effect_frame, (current_x, current_y), shield_size, (255, 0, 0), -1)
            # Inner circles
            cv2.circle(effect_frame, (current_x, current_y), int(shield_size * 0.8), (0, 0, 255), -1)
            cv2.circle(effect_frame, (current_x, current_y), int(shield_size * 0.6), (255, 255, 255), -1)
            cv2.circle(effect_frame, (current_x, current_y), int(shield_size * 0.4), (0, 0, 255), -1)
            cv2.circle(effect_frame, (current_x, current_y), int(shield_size * 0.2), (255, 255, 255), -1)
            
            # Add star in center
            star_size = int(shield_size * 0.2)
            star_points = 5
            star_outer_radius = star_size
            star_inner_radius = int(star_size * 0.5)
            
            # Calculate star points
            star_vertices = []
            for i in range(star_points * 2):
                angle = math.pi / 2 + i * math.pi / star_points
                radius = star_outer_radius if i % 2 == 0 else star_inner_radius
                x = int(current_x + radius * math.cos(angle))
                y = int(current_y + radius * math.sin(angle))
                star_vertices.append((x, y))
            
            # Draw star
            if len(star_vertices) > 2:
                pts = np.array(star_vertices, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(effect_frame, [pts], (255, 255, 255))
            
            # Add motion blur based on rotation
            rotation = progress * 720  # Multiple rotations
            
            # Add trail effect
            alpha = 0.7  # Increased trail opacity
            trail_points = 4  # More trail points
            for i in range(1, trail_points + 1):
                trail_t = max(0, t - 0.05 * i)
                trail_x_offset = (target_x - start_x) * 0.3 * math.sin(math.pi * trail_t)
                trail_x = int(start_x + (target_x - start_x) * trail_t + trail_x_offset)
                trail_y = int(start_y + (target_y - start_y) * trail_t)
                
                trail_alpha = alpha * (1 - i / (trail_points + 1))
                trail_size = int(shield_size * (1 - i * 0.1))
                
                # Semi-transparent trail
                cv2.circle(effect_frame, (trail_x, trail_y), trail_size, 
                        (int(255 * trail_alpha), 0, 0), -1, lineType=cv2.LINE_AA)
            
            # Add impact effect at target if nearly complete
            if progress > 0.9:
                # Expanding circles
                for i in range(3):
                    radius = int((shield_size + 10 + i * 10) * (progress - 0.9) / 0.1)
                    cv2.circle(effect_frame, (target_x, target_y), radius, 
                             (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                
                # Ricocheting mini-shields
                if random.random() < 0.3:  # Only occasionally show mini-shields
                    for i in range(3):
                        angle = random.uniform(0, 2 * math.pi)
                        distance = radius * 0.8
                        mini_x = int(target_x + distance * math.cos(angle))
                        mini_y = int(target_y + distance * math.sin(angle))
                        mini_size = int(shield_size * 0.3)
                        
                        # Mini shield
                        cv2.circle(effect_frame, (mini_x, mini_y), mini_size, (255, 0, 0), -1)
                        cv2.circle(effect_frame, (mini_x, mini_y), int(mini_size * 0.6), 
                                 (255, 255, 255), -1)
        
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