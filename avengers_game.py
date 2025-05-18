import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import random
from hero_effects import HeroEffect

class AvengersGame:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,  # Reduced from 0.7 for easier detection
            min_tracking_confidence=0.6    # Reduced from 0.7 for easier tracking
        )
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.5)
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize hero effects
        self.hero_effect = HeroEffect(self.width, self.height)
        
        # Store effect locations (x, y, scale, rotation, frame_created)
        self.effect_locations = []
        
        # Tracking previous finger positions
        self.prev_thumb_tip = None
        self.prev_index_tip = None
        
        # State variables
        self.collecting_power = False
        self.using_power = False
        self.power_start_time = 0
        self.power_duration = 1.5  # seconds
        
        # Animation parameters
        self.collected_powers = 0
        self.active_powers = []
        self.powers = []  # Initialize powers list
        
        # Game state
        self.current_hero = "Iron Man"  # Default hero
        self.score = 0
        self.level = 1
        self.targets = []
        self.max_targets = 3  # Reduced from 5
        self.target_spawn_timer = time.time()
        self.target_spawn_interval = 3.0  # Increased from 2.0 to make it easier
        
        # Enemy types - now with drone designs
        self.enemy_types = [
            {"name": "Drone", "size": 40, "speed": 0.5, "health": 1, "color": (0, 0, 255), "points": 10},
            {"name": "Sentinel", "size": 50, "speed": 0.3, "health": 2, "color": (0, 50, 255), "points": 20},
            {"name": "Ultron Bot", "size": 60, "speed": 0.2, "health": 3, "color": (0, 100, 255), "points": 30}
        ]
        
        # Drone images for different enemy types
        self.drone_images = {
            "Drone": self.create_drone_image(40, (0, 0, 255)),
            "Sentinel": self.create_drone_image(50, (0, 50, 255)),
            "Ultron Bot": self.create_drone_image(60, (0, 100, 255))
        }
        
        # Special effects
        self.hit_effects = []
        self.kill_effects = []  # New array to store kill effects
        self.hit_effect_duration = 0.5  # seconds
        self.kill_effect_duration = 1.5  # seconds, longer for kill effects
        
        # Frame counter for effects
        self.frame_count = 0
        
        # Menu state
        self.in_menu = True
        self.hero_select_menu = False
        
        # For aiming
        self.aiming_mode = False
        self.aim_target = None
        self.aim_start_pos = None
        
        # Add these new variables
        self.current_target = None
        self.target_lock_distance = 150  # Increased from 100 for easier targeting
        self.wave_number = 1
        self.enemies_per_wave = 3  # Reduced from 5
        self.enemies_remaining = 0
        self.game_over = False
        self.player_health = 100
        self.danger_zone_radius = 150  # Area around face considered dangerous
        self.max_powers = 5  # Increased from 3 to give player more powers
        self.face_position = None
        self.last_power_time = time.time()  # Initialize last power time
        self.power_cooldown = 0.5  # Reduced from 1.0 seconds to make power usage easier
        
        # Player circle size (smaller than before)
        self.player_circle_size = 30  # Reduced size
        
        # Initialize the first wave
        self.spawn_wave()
    
    def create_drone_image(self, size, color):
        """Create a drone image instead of simple circles"""
        img = np.zeros((size*2, size*2, 4), dtype=np.uint8)
        
        # Base hexagonal body
        points = []
        for i in range(6):
            angle = i * math.pi / 3
            x = int(size + (size-8) * math.cos(angle))
            y = int(size + (size-8) * math.sin(angle))
            points.append([x, y])
        
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(img, [points], (*color[:3], 200))  # Semi-transparent body
        cv2.polylines(img, [points], True, (*color[:3], 255), 2)  # Solid outline
        
        # Core center
        cv2.circle(img, (size, size), size//3, (*color[:3], 255), -1)
        cv2.circle(img, (size, size), size//3 - 2, (255, 255, 255, 150), 1)
        
        # Draw propellers (4 sets)
        propeller_positions = [
            (size - size//2, size - size//2),  # Top-left
            (size + size//2, size - size//2),  # Top-right
            (size - size//2, size + size//2),  # Bottom-left
            (size + size//2, size + size//2)   # Bottom-right
        ]
        
        # Draw each propeller with multiple blades
        for pos in propeller_positions:
            # Propeller hub
            cv2.circle(img, pos, size//6, (*color[:3], 255), -1)
            cv2.circle(img, pos, size//6 - 1, (255, 255, 255, 150), 1)
            
            # Propeller blades (3 blades with motion blur effect)
            for i in range(3):
                angle = i * 2 * math.pi / 3
                # Main blade
                end_x = int(pos[0] + (size//3) * math.cos(angle))
                end_y = int(pos[1] + (size//3) * math.sin(angle))
                cv2.line(img, pos, (end_x, end_y), (*color[:3], 200), 2)
                
                # Motion blur effect
                blur_angle1 = angle + 0.2
                blur_angle2 = angle - 0.2
                blur_x1 = int(pos[0] + (size//3) * math.cos(blur_angle1))
                blur_y1 = int(pos[1] + (size//3) * math.sin(blur_angle1))
                blur_x2 = int(pos[0] + (size//3) * math.cos(blur_angle2))
                blur_y2 = int(pos[1] + (size//3) * math.sin(blur_angle2))
                cv2.line(img, pos, (blur_x1, blur_y1), (*color[:3], 100), 1)
                cv2.line(img, pos, (blur_x2, blur_y2), (*color[:3], 100), 1)
        
        # Add tech details
        # Energy lines
        for i in range(3):
            angle = i * 2 * math.pi / 3
            start_x = int(size + (size//3) * math.cos(angle))
            start_y = int(size + (size//3) * math.sin(angle))
            end_x = int(size + (size-10) * math.cos(angle))
            end_y = int(size + (size-10) * math.sin(angle))
            cv2.line(img, (start_x, start_y), (end_x, end_y), (255, 255, 255, 150), 1)
        
        # Add glow effect around core
        cv2.circle(img, (size, size), size//2, (*color[:3], 50), -1)
        
        return img
    
    def is_pinch_gesture(self, hand_landmarks):
        """Detect pinch gesture (thumb and index finger touching)."""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Convert normalized coordinates to pixel coordinates
        thumb_x, thumb_y = int(thumb_tip.x * self.width), int(thumb_tip.y * self.height)
        index_x, index_y = int(index_tip.x * self.width), int(index_tip.y * self.height)
        
        # Calculate distance between thumb and index finger tips
        distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
        
        # Store current positions
        self.prev_thumb_tip = (thumb_x, thumb_y)
        self.prev_index_tip = (index_x, index_y)
        
        # If distance is small enough, consider it a pinch
        return distance < 30, (thumb_x + index_x) // 2, (thumb_y + index_y) // 2
    
    def is_fist_gesture(self, hand_landmarks):
        """Detect if the hand is making a fist."""
        # Get fingertip and middle joints
        fingertips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        middle_joints = [
            self.mp_hands.HandLandmark.THUMB_IP,  # For thumb, use IP joint
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        # Check if all fingertips are below their middle joints (fingers curled)
        all_fingers_curled = True
        for tip, mid in zip(fingertips, middle_joints):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mid].y:
                all_fingers_curled = False
                break
                
        return all_fingers_curled
    
    def is_open_palm_gesture(self, hand_landmarks):
        """Detect if the hand is open palm facing the camera."""
        # Get fingertip and middle joints
        fingertips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        middle_joints = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        # Check if all fingertips are extended (above their middle joints)
        all_fingers_extended = True
        for tip, mid in zip(fingertips, middle_joints):
            if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mid].y:
                all_fingers_extended = False
                break
        
        # Also check if the palm is facing the camera
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        middle_finger_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # If the middle finger's base is closer to the camera than the wrist, palm is likely facing camera
        palm_facing_camera = middle_finger_mcp.z < wrist.z
        
        return all_fingers_extended and palm_facing_camera
    
    def detect_thumb_up(self, hand_landmarks):
        """Detect if the thumb is up (for menu selection)."""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Thumb is pointing up if its tip is higher than the other joints
        return (thumb_tip.y < thumb_ip.y < thumb_mcp.y < wrist.y)
    
    def detect_pointing_gesture(self, hand_landmarks):
        """Detect if the index finger is pointing (for targeting)."""
        # Index finger should be extended
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # Other fingers should be curled
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        
        # Index finger extended
        index_extended = index_tip.y < index_dip.y < index_pip.y
        
        # Other fingers curled
        other_curled = (
            middle_tip.y > middle_pip.y and
            ring_tip.y > ring_pip.y and
            pinky_tip.y > pinky_pip.y
        )
        
        is_pointing = index_extended and other_curled
        
        if is_pointing:
            # Get pointing direction
            direction_x = index_tip.x - index_mcp.x
            direction_y = index_tip.y - index_mcp.y
            
            # Convert to screen coordinates
            tip_x = int(index_tip.x * self.width)
            tip_y = int(index_tip.y * self.height)
            mcp_x = int(index_mcp.x * self.width)
            mcp_y = int(index_mcp.y * self.height)
            
            return True, (tip_x, tip_y), (mcp_x, mcp_y)
        
        return False, None, None
    
    def spawn_wave(self):
        self.enemies_per_wave = 2 + min(2, self.wave_number)  # Slower scaling, max 4 enemies
        self.wave_number += 1
        self.targets = []
        self.enemies_remaining = self.enemies_per_wave
        
        # Randomize enemy types per wave
        types = random.choices(
            self.enemy_types,
            weights=[0.6, 0.3, 0.1],  # More basic enemies
            k=self.enemies_per_wave
        )
        
        for enemy_type in types:
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            pos = self.get_edge_position(edge)
            self.create_enemy(pos, enemy_type)
            time.sleep(0.5)  # Reduced delay between spawns
    
    def get_edge_position(self, edge):
        padding = 50
        if edge == 'top':
            return (random.randint(padding, self.width-padding), padding)
        elif edge == 'bottom':
            return (random.randint(padding, self.width-padding), self.height-padding)
        elif edge == 'left':
            return (padding, random.randint(padding, self.height-padding))
        else:  # right
            return (self.width-padding, random.randint(padding, self.height-padding))
    
    def create_enemy(self, position, enemy_type):
        base_speed = 0.5  # Reduced from 1.0
        speed = base_speed + (self.wave_number * 0.02)  # Smaller speed increase per wave
        size = enemy_type['size']
        target = {
            'x': position[0],
            'y': position[1],
            'dx': 0,
            'dy': 0,
            'size': size,
            'speed': min(speed, 1.0),  # Maximum speed cap reduced
            'color': enemy_type['color'],
            'health': 100,
            'locked': False,
            'enemy_type': enemy_type['name']  # Store enemy type for effects
        }
        self.targets.append(target)
    
    def update_targets(self, faces):
        # Modified target movement to head towards face
        if len(faces) > 0 and self.face_position:
            face_center = self.face_position
            
            for target in self.targets:
                # Calculate direction towards face
                dx = face_center[0] - target['x']
                dy = face_center[1] - target['y']
                dist = math.sqrt(dx**2 + dy**2)
                
                # Normalize direction and apply speed
                if dist > 0:
                    target['dx'] = dx/dist * target['speed']
                    target['dy'] = dy/dist * target['speed']
                
                # Check if enemy reached danger zone
                if dist < self.danger_zone_radius:
                    self.player_health -= 0.2  # Reduced damage from 0.5
                    
                # Update position
                target['x'] += target['dx']
                target['y'] += target['dy']
    
    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        faces = []
        
        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                x, y, width, height = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
                faces.append((x, y, width, height))
                self.face_position = (x + width//2, y + height//2)
        
        return faces
    
    def handle_gestures(self, hand_landmarks):
        # Clear previous states
        self.aiming_mode = False
        self.collecting_power = False
        
        # Menu is now handled by key press, so we don't check for menu gesture
            
        if self.is_targeting_gesture(hand_landmarks):
            self.aiming_mode = True
            self.handle_targeting(hand_landmarks)
            
        if self.is_power_throw_gesture(hand_landmarks):
            self.handle_power_throw(hand_landmarks)
            
        if self.is_power_collect_gesture(hand_landmarks):
            self.collecting_power = True
            self.collect_powers()
    
    def detect_menu_gesture(self, hand_landmarks):
        # The menu is now opened with the 'C' key, not a gesture
        return False  # Always return False so the gesture doesn't trigger menu

    def is_targeting_gesture(self, hand_landmarks):
        # Index finger extended, others closed
        index_ext = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
                   hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y
        middle_closed = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > \
                       hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        return index_ext and middle_closed

    def is_power_throw_gesture(self, hand_landmarks):
        # Open palm with all fingers extended
        return self.is_open_palm_gesture(hand_landmarks)

    def is_power_collect_gesture(self, hand_landmarks):
        # Closed fist
        return self.is_fist_gesture(hand_landmarks)
    
    def get_hand_position(self, hand_landmarks):
        # Get hand position based on index finger
        x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.width)
        y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.height)
        return (x, y)
    
    def handle_targeting(self, hand_landmarks):
        hand_pos = self.get_hand_position(hand_landmarks)
        closest = None
        min_dist = float('inf')
        
        for enemy in self.targets:
            # Calculate distance to hand position
            dx = enemy['x'] - hand_pos[0]
            dy = enemy['y'] - hand_pos[1]
            dist = math.sqrt(dx**2 + dy**2)
            
            # Prioritize enemies moving towards face
            moving_towards = False
            if self.face_position:
                face_dx = self.face_position[0] - enemy['x']
                face_dy = self.face_position[1] - enemy['y']
                # Dot product to determine if moving towards face
                moving_towards = (face_dx * enemy['dx'] + face_dy * enemy['dy']) > 0
                
            # Update closest target considering both distance and direction
            if dist < self.target_lock_distance and dist < min_dist:
                if moving_towards or dist < self.target_lock_distance/2:
                    min_dist = dist
                    closest = enemy

        # Clear previous locks
        for enemy in self.targets:
            enemy['locked'] = False
        
        if closest:
            closest['locked'] = True
            self.current_target = closest
        else:
            self.current_target = None
    
    def handle_power_throw(self, hand_landmarks):
        current_time = time.time()
        if (self.collected_powers > 0 and 
            current_time - self.last_power_time > self.power_cooldown and
            self.current_target):
            
            hero_power = self.hero_effect.get_hero_power(self.current_hero)
            hand_pos = self.get_hand_position(hand_landmarks)
            
            # Calculate trajectory with predictive aiming
            tx, ty = self.current_target['x'], self.current_target['y']
            dx = tx - hand_pos[0]
            dy = ty - hand_pos[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            # Predict target position based on its speed
            time_to_hit = distance / hero_power["attack_speed"]
            future_x = tx + self.current_target['dx'] * time_to_hit
            future_y = ty + self.current_target['dy'] * time_to_hit
            
            # Recalculate direction to predicted position
            dx = future_x - hand_pos[0]
            dy = future_y - hand_pos[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                power = {
                    'x': hand_pos[0],
                    'y': hand_pos[1],
                    'dx': dx/distance * hero_power["attack_speed"],
                    'dy': dy/distance * hero_power["attack_speed"],
                    'size': hero_power["power_size"],
                    'color': hero_power["power_color"],
                    'effect': hero_power["power_effect"],
                    'target': self.current_target,
                    'hero': self.current_hero  # Store the hero for kill effects
                }
                self.powers.append(power)
                self.last_power_time = current_time
                self.collected_powers -= 1
    
    def collect_powers(self):
        if self.collected_powers < self.max_powers:
            current_time = time.time()
            if current_time - self.last_power_time > self.power_cooldown / 2:
                self.collected_powers += 1
                self.last_power_time = current_time
    
    def update_powers(self):
        for power in self.powers[:]:
            power['x'] += power['dx']
            power['y'] += power['dy']
            
            # Check if power is out of bounds
            if (power['x'] < 0 or power['x'] > self.width or 
                power['y'] < 0 or power['y'] > self.height):
                self.powers.remove(power)
                continue
            
            # Check for collision with target
            if power['target'] in self.targets:
                target = power['target']
                distance = math.hypot(power['x']-target['x'], power['y']-target['y'])
                
                if distance < target['size'] + power['size']:
                    # Hit effect
                    self.add_hit_effect(target['x'], target['y'], power['color'])
                    target['health'] -= 70  # Increased damage for easier kills
                    self.powers.remove(power)
                    
                    # Remove target if health depleted
                    if target['health'] <= 0:
                        # Add hero-specific kill effect
                        self.add_kill_effect(target['x'], target['y'], self.current_hero, target['size'])
                        self.targets.remove(target)
                        self.score += 15 * self.wave_number
                        self.enemies_remaining -= 1
    
    def add_hit_effect(self, x, y, color):
        # Add a hit effect at the specified location
        self.hit_effects.append({
            'x': x,
            'y': y,
            'color': color,
            'size': 10,
            'time': time.time(),
            'max_size': 30
        })
    
    def update_hit_effects(self, frame):
        current_time = time.time()
        for effect in self.hit_effects[:]:
            elapsed = current_time - effect['time']
            if elapsed > self.hit_effect_duration:
                self.hit_effects.remove(effect)
                continue
            
            # Grow the effect
            progress = elapsed / self.hit_effect_duration
            size = int(effect['size'] + progress * (effect['max_size'] - effect['size']))
            alpha = 1 - progress
            
            # Draw the effect
            cv2.circle(frame, (int(effect['x']), int(effect['y'])), 
                     size, effect['color'], 2)
            
        return frame
    
    def add_kill_effect(self, x, y, hero_name, target_size):
        """Add a hero-specific kill effect at the target location"""
        # Convert hero name to appropriate format for effects
        if hero_name == "Iron Man":
            hero_id = "iron_man"
        elif hero_name == "Spider-Man":
            hero_id = "spider_man"
        elif hero_name == "Thor":
            hero_id = "thor"
        elif hero_name == "Hulk":
            hero_id = "hulk"
        elif hero_name == "Captain America":
            hero_id = "captain_america"
        else:
            hero_id = "iron_man"  # Default
            
        # Add effect with increased intensity for dramatic kills
        self.kill_effects.append({
            'x': x,
            'y': y,
            'hero': hero_id,
            'size': target_size * 1.5,
            'time': time.time(),
            'frame_count': 0,
            'intensity': 1.5
        })

    def update_kill_effects(self, frame):
        """Render hero-specific kill effects on the frame"""
        current_time = time.time()
        for effect in self.kill_effects[:]:
            elapsed = current_time - effect['time']
            if elapsed > self.kill_effect_duration:
                self.kill_effects.remove(effect)
                continue
            
            # Animate the effect
            effect['frame_count'] += 1
            progress = min(1.0, elapsed / self.kill_effect_duration)
            intensity = effect['intensity'] * (1.0 - 0.5 * progress)  # Reduce intensity over time
            
            # Get hero effect from HeroEffect class
            effect_frame = self.hero_effect.create_targeted_effect(
                effect['hero'],
                (effect['x'], effect['y']), 
                (effect['x'], effect['y']),
                progress,
                intensity=intensity,
                frame_count=effect['frame_count']
            )
            
            # Apply effect to frame
            if effect_frame is not None:
                # Create mask for blending
                mask = np.sum(effect_frame, axis=2) > 0
                mask = np.expand_dims(mask, axis=2).astype(np.float32)
                mask = np.repeat(mask, 3, axis=2)
                
                # Apply alpha blending
                alpha = (1.0 - progress * 0.7) * effect['intensity']  # Fade out effect
                frame = cv2.addWeighted(
                    frame, 1.0, 
                    np.where(mask > 0, effect_frame, frame).astype(np.uint8), 
                    alpha, 0
                )
        
        return frame
    
    def handle_wave_progression(self):
        # Check if current wave is cleared
        if self.enemies_remaining <= 0 and len(self.targets) == 0:
            self.spawn_wave()
    
    def handle_hero_selection(self, hand_landmarks):
        # Use horizontal hand movement to cycle through heroes
        wrist_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
        heroes = list(self.hero_effect.hero_assets.keys())
        
        # Map hand position to hero selection
        section_width = 1.0 / len(heroes)
        selected_index = min(int(wrist_x / section_width), len(heroes)-1)
        
        # Get the selected hero_id and convert to proper format
        hero_id = heroes[selected_index]
        self.current_hero = self.hero_effect.hero_assets[hero_id]["name"]
        
        # Confirm selection with fist gesture
        if self.is_fist_gesture(hand_landmarks):
            print(f"Selected hero: {self.current_hero}")  # Debug print
            self.exit_menu()
    
    def exit_menu(self):
        self.in_menu = False
    
    def show_menu(self, frame, hand_landmarks=None):
        # Draw semitransparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw title
        cv2.putText(frame, "AVENGERS HERO SELECTION", (self.width//2 - 250, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Draw hero options
        heroes = list(self.hero_effect.hero_assets.keys())
        section_width = self.width / len(heroes)
        
        for i, hero_id in enumerate(heroes):
            hero_data = self.hero_effect.hero_assets[hero_id]
            x = int(i * section_width + section_width/2)
            y = self.height // 2
            
            # Get hero mask image for icon
            mask_img = hero_data.get("mask_image")
            if mask_img is not None:
                # Calculate icon size and position
                icon_size = 100
                icon_x = x - icon_size//2
                icon_y = y - icon_size//2
                
                # Resize mask image to icon size
                icon = cv2.resize(mask_img, (icon_size, icon_size))
                
                # Highlight selected hero
                current_hero_id = self.current_hero.lower().replace(" ", "_").replace("-", "_")
                if hero_id == current_hero_id:
                    # Draw highlight border
                    cv2.rectangle(frame, (icon_x-5, icon_y-5), 
                                (icon_x+icon_size+5, icon_y+icon_size+5), 
                                (0, 255, 0), 2)
                    blend_alpha = 1.0  # Full opacity for selected hero
                else:
                    blend_alpha = 0.7  # Slightly transparent for unselected heroes
                
                # Ensure icon_x and icon_y are within frame bounds
                if (icon_x >= 0 and icon_y >= 0 and 
                    icon_x + icon_size <= frame.shape[1] and 
                    icon_y + icon_size <= frame.shape[0]):
                    
                    # Create ROI
                    roi = frame[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size]
                    
                    # Handle alpha channel if present
                    if icon.shape[2] == 4:
                        # Extract alpha channel and normalize to 0-1
                        alpha_channel = icon[:, :, 3:] / 255.0
                        # Get RGB channels
                        icon_rgb = icon[:, :, :3]
                        
                        # Blend based on alpha channel
                        blended = (1.0 - alpha_channel * blend_alpha) * roi + (alpha_channel * blend_alpha) * icon_rgb
                        frame[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = blended.astype(np.uint8)
                    else:
                        # If no alpha channel, use simple alpha blending
                        blended = cv2.addWeighted(roi, 1 - blend_alpha, icon, blend_alpha, 0)
                        frame[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = blended
            
            # Draw hero name
            name = hero_data["name"]
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = x - text_size[0]//2
            text_y = y + 70
            
            # Highlight selected hero name
            color = (0, 255, 0) if hero_id == self.current_hero.lower().replace(" ", "_") else (200, 200, 200)
            cv2.putText(frame, name, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw power type
            power_text = f"Power: {hero_data['effect_name']}"
            text_size = cv2.getTextSize(power_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = x - text_size[0]//2
            cv2.putText(frame, power_text, (text_x, text_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Instructions
        cv2.putText(frame, "Move hand left/right to select", (self.width//2 - 200, self.height - 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Make a fist to confirm selection", (self.width//2 - 200, self.height - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def show_game_over_screen(self):
        # Create black frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Game over text
        cv2.putText(frame, "GAME OVER", (self.width//2 - 150, self.height//2 - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # Show score
        cv2.putText(frame, f"Final Score: {self.score}", (self.width//2 - 150, self.height//2 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Instructions to restart
        cv2.putText(frame, "Press 'R' to restart or 'Q' to quit", (self.width//2 - 200, self.height//2 + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Avengers Game', frame)
        key = cv2.waitKey(1)
        
        if key == ord('r'):
            self.__init__()  # Reset game
        elif key == ord('q'):
            self.cleanup()
            exit()
    
    def render_ui(self, frame):
        # Function to draw text with background
        def draw_text_with_background(img, text, pos, font, font_scale, text_color, bg_color):
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness=2)
            # Draw background rectangle
            cv2.rectangle(img, 
                        (pos[0], pos[1] - text_h - baseline), 
                        (pos[0] + text_w, pos[1] + baseline), 
                        bg_color, -1)
            # Draw text
            cv2.putText(img, text, pos, font, font_scale, text_color, 2)

        # Display score with background
        draw_text_with_background(frame, f"Score: {self.score}", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                (255, 255, 255), (0, 0, 0))
        
        # Display current hero with background
        draw_text_with_background(frame, f"Hero: {self.current_hero}", 
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                (255, 255, 255), (0, 0, 0))
        
        # Display wave number with background
        draw_text_with_background(frame, f"Wave: {self.wave_number}", 
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                (255, 255, 255), (0, 0, 0))
        
        # Display health bar with improved visibility
        health_width = int((self.player_health / 100) * 200)
        # Draw background for health bar
        cv2.rectangle(frame, (self.width - 220, 20), (self.width - 20, 60), (0, 0, 0), -1)
        # Draw health bar background
        cv2.rectangle(frame, (self.width - 220, 30), (self.width - 20, 50), (50, 50, 50), -1)
        # Draw actual health bar
        cv2.rectangle(frame, (self.width - 220, 30), (self.width - 220 + health_width, 50), (0, 0, 255), -1)
        # Draw health text with better contrast
        draw_text_with_background(frame, f"Health: {int(self.player_health)}%", 
                                (self.width - 210, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                (255, 255, 255), (0, 0, 0))
        
        # Display collected powers with background
        for i in range(self.max_powers):
            center_x = self.width - 50 - i*30
            center_y = 80
            if i < self.collected_powers:
                color = self.hero_effect.get_hero_power(self.current_hero)["power_color"]
                # Draw power circle with background
                cv2.circle(frame, (center_x, center_y), 12, (0, 0, 0), -1)  # Background
                cv2.circle(frame, (center_x, center_y), 10, color, -1)  # Power
            else:
                # Draw empty power slot with background
                cv2.circle(frame, (center_x, center_y), 12, (0, 0, 0), -1)  # Background
                cv2.circle(frame, (center_x, center_y), 10, (100, 100, 100), 2)  # Empty slot
        
        return frame
    
    def render_targets(self, frame):
        # Draw each target as a drone instead of a simple circle
        for target in self.targets:
            # Get drone image for this enemy type
            drone_img = self.drone_images.get(target['enemy_type'], self.drone_images["Drone"])
            
            # Calculate position for overlay
            x, y = int(target['x']), int(target['y'])
            
            # Highlight if locked
            if target['locked']:
                # Draw targeting brackets around drone
                size = target['size']
                cv2.line(frame, (x - size, y - size), (x - size//2, y - size), (0, 255, 0), 2)
                cv2.line(frame, (x - size, y - size), (x - size, y - size//2), (0, 255, 0), 2)
                cv2.line(frame, (x + size, y - size), (x + size//2, y - size), (0, 255, 0), 2)
                cv2.line(frame, (x + size, y - size), (x + size, y - size//2), (0, 255, 0), 2)
                cv2.line(frame, (x - size, y + size), (x - size//2, y + size), (0, 255, 0), 2)
                cv2.line(frame, (x - size, y + size), (x - size, y + size//2), (0, 255, 0), 2)
                cv2.line(frame, (x + size, y + size), (x + size//2, y + size), (0, 255, 0), 2)
                cv2.line(frame, (x + size, y + size), (x + size, y + size//2), (0, 255, 0), 2)
            
            # Calculate drone image size for overlay
            drone_h, drone_w = drone_img.shape[:2]
            
            # Overlay drone image
            alpha_mask = drone_img[:, :, 3] / 255.0
            alpha_mask = np.expand_dims(alpha_mask, axis=2)
            alpha_mask = np.repeat(alpha_mask, 3, axis=2)
            
            # Calculate placement coordinates
            x_start = max(0, x - drone_w//2)
            y_start = max(0, y - drone_h//2)
            x_end = min(frame.shape[1], x_start + drone_w)
            y_end = min(frame.shape[0], y_start + drone_h)
            
            # Calculate the portion of the drone image to use
            img_x_start = max(0, drone_w//2 - x if x < drone_w//2 else 0)
            img_y_start = max(0, drone_h//2 - y if y < drone_h//2 else 0)
            img_x_end = drone_w - max(0, (x + drone_w//2) - frame.shape[1] if (x + drone_w//2) > frame.shape[1] else 0)
            img_y_end = drone_h - max(0, (y + drone_h//2) - frame.shape[0] if (y + drone_h//2) > frame.shape[0] else 0)
            
            # Ensure we have valid dimensions
            if x_end > x_start and y_end > y_start and img_x_end > img_x_start and img_y_end > img_y_start:
                # Get the region of the frame where we'll place the drone
                roi = frame[y_start:y_end, x_start:x_end]
                
                # Get the region of the drone image
                drone_region = drone_img[img_y_start:img_y_end, img_x_start:img_x_end]
                alpha_region = alpha_mask[img_y_start:img_y_end, img_x_start:img_x_end]
                
                # Make sure dimensions match before blending
                if roi.shape[:2] == drone_region.shape[:2]:
                    # Blend drone with frame
                    blended = (1.0 - alpha_region) * roi + alpha_region * drone_region[:, :, :3]
                    frame[y_start:y_end, x_start:x_end] = blended
            
            # Draw health bar above target
            health_width = int((target['health'] / 100) * target['size'] * 2)
            cv2.rectangle(frame, (int(target['x'] - target['size']), int(target['y'] - target['size'] - 10)), 
                         (int(target['x'] + target['size']), int(target['y'] - target['size'] - 5)), 
                         (50, 50, 50), -1)
            cv2.rectangle(frame, (int(target['x'] - target['size']), int(target['y'] - target['size'] - 10)), 
                         (int(target['x'] - target['size'] + health_width), int(target['y'] - target['size'] - 5)), 
                         (0, 0, 255), -1)
        
        return frame
    
    def render_powers(self, frame):
        # Draw each power with hero-specific visuals
        for power in self.powers:
            hero = power.get('hero', 'Iron Man')  # Default to Iron Man if not specified
            hero_effect = power.get('effect', 'repulsor')
            
            # Get power position and size
            x, y = int(power['x']), int(power['y'])
            size = power['size']
            color = power['color']
            
            # Draw different effects based on hero
            if hero == "Iron Man":
                # Iron Man - Repulsor beam with energy rings
                if power['target'] in self.targets:
                    target_x, target_y = int(power['target']['x']), int(power['target']['y'])
                    
                    # Draw energy beam core
                    cv2.line(frame, (x, y), (target_x, target_y), (0, 200, 255), 3, cv2.LINE_AA)
                    
                    # Add pulsing energy rings along beam
                    beam_length = math.hypot(target_x - x, target_y - y)
                    num_rings = int(beam_length / 20)
                    for i in range(num_rings):
                        t = i / num_rings
                        ring_x = int(x + (target_x - x) * t)
                        ring_y = int(y + (target_y - y) * t)
                        ring_size = 3 + 2 * math.sin(time.time() * 10 + i)  # Pulsing effect
                        cv2.circle(frame, (ring_x, ring_y), int(ring_size), (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Add impact explosion
                    cv2.circle(frame, (target_x, target_y), size, (0, 150, 255), -1)
                    cv2.circle(frame, (target_x, target_y), size//2, (255, 255, 255), -1)
                
                # Draw repulsor glow at hand
                for i in range(3):
                    cv2.circle(frame, (x, y), size + i*5, (0, 100 + i*50, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), size//2, (255, 255, 255), -1)
                
            elif hero == "Spider-Man":
                # Spider-Man - Web shooter with detailed web patterns
                if power['target'] in self.targets:
                    target_x, target_y = int(power['target']['x']), int(power['target']['y'])
                    
                    # Calculate main web strand points
                    points = []
                    num_segments = 8
                    for i in range(num_segments + 1):
                        t = i / num_segments
                        # Add slight curve to web
                        mid_x = (x + target_x) / 2
                        mid_y = (y + target_y) / 2 - 20 * math.sin(math.pi * t)
                        if i == 0:
                            seg_x, seg_y = x, y
                        elif i == num_segments:
                            seg_x, seg_y = target_x, target_y
                        else:
                            seg_x = x + (mid_x - x) * (2 * t) if t < 0.5 else mid_x + (target_x - mid_x) * (2 * t - 1)
                            seg_y = y + (mid_y - y) * (2 * t) if t < 0.5 else mid_y + (target_y - mid_y) * (2 * t - 1)
                        points.append((int(seg_x), int(seg_y)))
                    
                    # Draw main web strand
                    for i in range(len(points) - 1):
                        cv2.line(frame, points[i], points[i+1], (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Add web details
                    for i in range(1, len(points) - 1):
                        # Add crossing strands
                        angle = math.atan2(points[i+1][1] - points[i-1][1],
                                         points[i+1][0] - points[i-1][0]) + math.pi/2
                        length = 10 + 5 * math.sin(i * math.pi / 2)
                        
                        x1 = int(points[i][0] + length * math.cos(angle))
                        y1 = int(points[i][1] + length * math.sin(angle))
                        x2 = int(points[i][0] - length * math.cos(angle))
                        y2 = int(points[i][1] - length * math.sin(angle))
                        
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
                        
                        # Add small connecting webs
                        if i < len(points) - 2:
                            mid_x = (points[i][0] + points[i+1][0]) // 2
                            mid_y = (points[i][1] + points[i+1][1]) // 2
                            cv2.line(frame, (x1, y1), (mid_x, mid_y), (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.line(frame, (x2, y2), (mid_x, mid_y), (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw web shooter effect at hand
                cv2.circle(frame, (x, y), size//2, (200, 200, 200), -1)
                cv2.circle(frame, (x, y), size//4, (255, 255, 255), -1)
                
            elif hero == "Thor":
                # Thor - Lightning with dynamic branching and thunder effects
                if power['target'] in self.targets:
                    target_x, target_y = int(power['target']['x']), int(power['target']['y'])
                    
                    # Generate main lightning bolt path
                    points = [(x, y)]
                    current_x, current_y = x, y
                    segments = 8
                    
                    for i in range(1, segments + 1):
                        t = i / segments
                        next_x = int(x + (target_x - x) * t)
                        next_y = int(y + (target_y - y) * t)
                        
                        # Add randomness to path
                        if i != segments:  # Don't randomize final point
                            offset = 15 * math.sin(time.time() * 10 + i * math.pi)  # Dynamic movement
                            next_x += int(offset)
                            next_y += int(offset * 0.5)
                        
                        points.append((next_x, next_y))
                        current_x, current_y = next_x, next_y
                    
                    # Draw main lightning bolt
                    for i in range(len(points) - 1):
                        # Outer glow
                        cv2.line(frame, points[i], points[i+1], (255, 255, 100), 4, cv2.LINE_AA)
                        # Inner core
                        cv2.line(frame, points[i], points[i+1], (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Add branching lightning
                    for i in range(1, len(points) - 1):
                        if random.random() < 0.7:  # 70% chance for each branch
                            # Create two branches at each point
                            for _ in range(2):
                                branch_length = random.randint(15, 30)
                                angle = math.atan2(points[i+1][1] - points[i-1][1],
                                                 points[i+1][0] - points[i-1][0])
                                angle += random.uniform(-math.pi/3, math.pi/3)
                                
                                end_x = int(points[i][0] + branch_length * math.cos(angle))
                                end_y = int(points[i][1] + branch_length * math.sin(angle))
                                
                                # Draw branch with glow effect
                                cv2.line(frame, points[i], (end_x, end_y), (255, 255, 100), 3, cv2.LINE_AA)
                                cv2.line(frame, points[i], (end_x, end_y), (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Add impact effect at target
                    impact_size = size * 2
                    for i in range(3):
                        cv2.circle(frame, (target_x, target_y), impact_size - i*5,
                                 (255, 255, 100), 2, cv2.LINE_AA)
                
                # Draw Mjolnir at hand
                hammer_size = size
                # Handle
                cv2.rectangle(frame, (x - hammer_size//6, y - hammer_size),
                            (x + hammer_size//6, y - hammer_size//2),
                            (150, 150, 150), -1)
                # Head
                cv2.rectangle(frame, (x - hammer_size//2, y - hammer_size//2),
                            (x + hammer_size//2, y), (200, 200, 200), -1)
                # Details
                cv2.rectangle(frame, (x - hammer_size//3, y - hammer_size//2),
                            (x + hammer_size//3, y), (150, 150, 150), 2)
                
            elif hero == "Hulk":
                # Hulk - Smash with shockwave and debris
                if power['target'] in self.targets:
                    target_x, target_y = int(power['target']['x']), int(power['target']['y'])
                    
                    # Draw smash path with growing effect
                    dist = math.hypot(target_x - x, target_y - y)
                    num_segments = int(dist / 10)
                    
                    for i in range(num_segments):
                        t = i / num_segments
                        # Create growing effect along path
                        current_x = int(x + (target_x - x) * t)
                        current_y = int(y + (target_y - y) * t)
                        current_size = int(size * (0.5 + t * 0.5))  # Grows as it moves
                        
                        # Draw fist silhouette with motion blur
                        alpha = 0.3 * (1 - t)  # Fade out trail
                        overlay = frame.copy()
                        cv2.circle(overlay, (current_x, current_y), current_size,
                                 (0, 200, 0), -1)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    
                    # Draw impact effect at target
                    # Shockwave rings
                    for i in range(3):
                        radius = int(size * (2 + i) * (1 + math.sin(time.time() * 5)))
                        cv2.circle(frame, (target_x, target_y), radius,
                                 (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Ground cracks
                    for i in range(8):
                        angle = i * math.pi / 4 + random.uniform(-0.2, 0.2)
                        length = size * 3
                        end_x = int(target_x + length * math.cos(angle))
                        end_y = int(target_y + length * math.sin(angle))
                        
                        # Draw crack with varying thickness
                        pts = []
                        for t in range(10):
                            t_val = t / 9
                            crack_x = int(target_x + length * t_val * math.cos(angle))
                            crack_y = int(target_y + length * t_val * math.sin(angle))
                            offset = random.randint(-5, 5)
                            pts.append([crack_x + offset, crack_y + offset])
                        
                        pts = np.array(pts, np.int32)
                        cv2.polylines(frame, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Add debris particles
                    for _ in range(10):
                        angle = random.uniform(0, 2 * math.pi)
                        speed = random.uniform(0.5, 1.0)
                        particle_x = int(target_x + size * 2 * speed * math.cos(angle))
                        particle_y = int(target_y + size * 2 * speed * math.sin(angle))
                        particle_size = random.randint(2, 6)
                        cv2.circle(frame, (particle_x, particle_y), particle_size,
                                 (0, 200, 0), -1)
                
                # Draw fist at hand
                cv2.circle(frame, (x, y), size, (0, 200, 0), -1)
                # Knuckle details
                for i in range(4):
                    offset_x = size//3 * math.cos(i * math.pi/6)
                    cv2.circle(frame, (int(x + offset_x), int(y - size//3)),
                             size//6, (0, 150, 0), -1)
                
            elif hero == "Captain America":
                # Captain America - Shield with star and dynamic spinning
                if power['target'] in self.targets:
                    target_x, target_y = int(power['target']['x']), int(power['target']['y'])
                    
                    # Calculate shield spin based on time
                    spin_angle = (time.time() * 720) % 360  # Two rotations per second
                    
                    # Draw shield trail
                    dist = math.hypot(target_x - x, target_y - y)
                    num_trails = int(dist / 20)
                    
                    for i in range(num_trails):
                        t = i / num_trails
                        trail_x = int(x + (target_x - x) * t)
                        trail_y = int(y + (target_y - y) * t)
                        trail_size = size * (1 - t * 0.3)  # Shield gets slightly smaller along trail
                        trail_angle = spin_angle - t * 360  # Spin along the trail
                        
                        # Draw simplified shield for trail
                        alpha = 0.3 * (1 - t)
                        overlay = frame.copy()
                        
                        # Rotate and draw shield
                        M = cv2.getRotationMatrix2D((trail_x, trail_y), trail_angle, 1)
                        
                        # Draw basic shield shape for trail
                        cv2.circle(overlay, (trail_x, trail_y), int(trail_size),
                                 (0, 0, 150), -1)
                        cv2.circle(overlay, (trail_x, trail_y), int(trail_size * 0.8),
                                 (200, 0, 0), -1)
                        cv2.circle(overlay, (trail_x, trail_y), int(trail_size * 0.6),
                                 (255, 255, 255), -1)
                        
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    
                    # Draw impact effect at target
                    cv2.circle(frame, (target_x, target_y), size * 2, (0, 0, 150), 2)
                    for i in range(8):
                        angle = i * math.pi / 4
                        end_x = int(target_x + size * 2 * math.cos(angle))
                        end_y = int(target_y + size * 2 * math.sin(angle))
                        cv2.line(frame, (target_x, target_y), (end_x, end_y),
                                (200, 0, 0), 2, cv2.LINE_AA)
                
                # Draw detailed shield at hand
                # Base circles
                cv2.circle(frame, (x, y), size, (0, 0, 150), -1)  # Outer blue
                cv2.circle(frame, (x, y), int(size * 0.8), (200, 0, 0), -1)  # Red
                cv2.circle(frame, (x, y), int(size * 0.6), (255, 255, 255), -1)  # White
                cv2.circle(frame, (x, y), int(size * 0.4), (200, 0, 0), -1)  # Red
                cv2.circle(frame, (x, y), int(size * 0.2), (0, 0, 150), -1)  # Blue center
                
                # Draw star
                star_size = size * 0.3
                star_points = []
                for i in range(5):
                    # Outer points
                    angle = math.pi/2 + i * 2*math.pi/5
                    px = x + star_size * math.cos(angle)
                    py = y + star_size * math.sin(angle)
                    star_points.append((int(px), int(py)))
                    
                    # Inner points
                    angle += math.pi/5
                    px = x + star_size * 0.4 * math.cos(angle)
                    py = y + star_size * 0.4 * math.sin(angle)
                    star_points.append((int(px), int(py)))
                
                # Draw star
                star_points = np.array(star_points, np.int32)
                star_points = star_points.reshape((-1, 1, 2))
                cv2.fillPoly(frame, [star_points], (255, 255, 255))
            
            else:
                # Default power effect
                cv2.circle(frame, (x, y), size, color, -1)
        
        return frame
    
    def render_danger_zone(self, frame, faces):
        if len(faces) > 0 and self.face_position:
            # Draw danger zone around face
            cv2.circle(frame, self.face_position, self.danger_zone_radius, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Apply hero mask to face
            for (x, y, w, h) in faces:
                # Convert hero name to hero_id format
                hero_id = self.current_hero.lower().replace(" ", "_").replace("-", "_")
                # Apply mask with proper scaling
                frame = self.hero_effect.apply_face_mask(frame, hero_id, [(x, y, w, h)])
        
        return frame
    
    def render_targeting_reticle(self, frame, hand_landmarks):
        if self.aiming_mode and hand_landmarks:
            hand_pos = self.get_hand_position(hand_landmarks)
            
            # Draw aiming reticle
            cv2.line(frame, (hand_pos[0] - 20, hand_pos[1]), (hand_pos[0] - 5, hand_pos[1]), (0, 255, 0), 2)
            cv2.line(frame, (hand_pos[0] + 5, hand_pos[1]), (hand_pos[0] + 20, hand_pos[1]), (0, 255, 0), 2)
            cv2.line(frame, (hand_pos[0], hand_pos[1] - 20), (hand_pos[0], hand_pos[1] - 5), (0, 255, 0), 2)
            cv2.line(frame, (hand_pos[0], hand_pos[1] + 5), (hand_pos[0], hand_pos[1] + 20), (0, 255, 0), 2)
            cv2.circle(frame, hand_pos, 30, (0, 255, 0), 1)
            cv2.circle(frame, hand_pos, self.target_lock_distance, (0, 255, 0), 1)
            
            # Draw lock line to target and confirmation text
            if self.current_target:
                cv2.line(frame, hand_pos, (int(self.current_target['x']), int(self.current_target['y'])), 
                        (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "TARGET LOCKED", (hand_pos[0]+20, hand_pos[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def check_game_state(self):
        # Check if player health is depleted
        if self.player_health <= 0:
            self.game_over = True
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face detection
            faces = self.detect_faces(frame)
            
            # Process hand detection
            hand_results = self.hands.process(rgb_frame)
            hand_landmarks = None
            
            if self.game_over:
                self.show_game_over_screen()
                continue
                
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                if self.in_menu:
                    self.handle_hero_selection(hand_landmarks)
                    frame = self.show_menu(frame, hand_landmarks)
                else:
                    self.handle_gestures(hand_landmarks)
            
            if not self.in_menu:
                # Update game elements
                self.update_targets(faces)
                self.update_powers()
                self.handle_wave_progression()
                self.check_game_state()
                
                # Render elements
                frame = self.render_targets(frame)
                frame = self.render_powers(frame)
                frame = self.render_danger_zone(frame, faces)
                frame = self.update_hit_effects(frame)
                frame = self.update_kill_effects(frame)  # Add kill effects
                if hand_landmarks:
                    frame = self.render_targeting_reticle(frame, hand_landmarks)
                frame = self.render_ui(frame)
            
            # Show frame
            cv2.imshow('Avengers Game', frame)
            
            # Check for key presses
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.in_menu = True
            elif key == ord('c'):  # 'C' key now opens hero selection menu
                self.in_menu = True
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = AvengersGame()
    game.run() 