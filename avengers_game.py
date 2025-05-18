import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import random
from hero_effects import HeroEffect

class HeroEffects:
    def __init__(self):
        self.hero_assets = {
            "Iron Man": {
                "power_color": (30, 90, 255),  # Red-orange
                "power_size": 25,
                "attack_speed": 15,
                "power_effect": "repulsor",
            },
            "Spider-Man": {
                "power_color": (0, 0, 255),  # Red
                "power_size": 15,
                "attack_speed": 20,
                "power_effect": "web",
            },
            "Thor": {
                "power_color": (255, 215, 0),  # Lightning
                "power_size": 35,
                "attack_speed": 12,
                "power_effect": "lightning",
            },
            "Hulk": {
                "power_color": (0, 200, 0),  # Green
                "power_size": 40,
                "attack_speed": 10,
                "power_effect": "smash",
            },
            "Captain America": {
                "power_color": (255, 0, 0),  # Blue
                "power_size": 30,
                "attack_speed": 17,
                "power_effect": "shield",
            }
        }
        
    def get_hero_power(self, hero_name):
        return self.hero_assets.get(hero_name, self.hero_assets["Iron Man"])

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
        
        # Enemy types
        self.enemy_types = [
            {"name": "Drone", "size": 40, "speed": 0.5, "health": 1, "color": (0, 0, 255), "points": 10},
            {"name": "Sentinel", "size": 50, "speed": 0.3, "health": 2, "color": (0, 50, 255), "points": 20},
            {"name": "Ultron Bot", "size": 60, "speed": 0.2, "health": 3, "color": (0, 100, 255), "points": 30}
        ]
        
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
        self.menu_gesture_start = None  # New for menu activation cooldown
        
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
        
        # Initialize the first wave
        self.spawn_wave()
    
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
        
        if self.detect_menu_gesture(hand_landmarks):
            self.in_menu = True
            return
            
        if self.is_targeting_gesture(hand_landmarks):
            self.aiming_mode = True
            self.handle_targeting(hand_landmarks)
            
        if self.is_power_throw_gesture(hand_landmarks):
            self.handle_power_throw(hand_landmarks)
            
        if self.is_power_collect_gesture(hand_landmarks):
            self.collecting_power = True
            self.collect_powers()
    
    def detect_menu_gesture(self, hand_landmarks):
        # More specific "rock on" gesture (index and pinky up, others down)
        index_ext = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
                   hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y
        middle_closed = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > \
                   hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_closed = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y > \
                 hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_ext = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y < \
               hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y
        
        # Require 1 second hold for menu activation
        if index_ext and pinky_ext and middle_closed and ring_closed:
            if not self.menu_gesture_start:
                self.menu_gesture_start = time.time()
            return time.time() - self.menu_gesture_start > 1.0
        else:
            self.menu_gesture_start = None
            return False

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
        
        self.current_hero = heroes[selected_index]
        
        # Confirm selection with fist gesture
        if self.is_fist_gesture(hand_landmarks):
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
        
        for i, hero in enumerate(heroes):
            x = int(i * section_width + section_width/2)
            y = self.height // 2
            
            # Highlight selected hero
            color = (0, 255, 0) if hero == self.current_hero else (200, 200, 200)
            size = 30 if hero == self.current_hero else 20
            
            cv2.putText(frame, hero, (x - len(hero)*5, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw hero power example
            hero_power = self.hero_effect.get_hero_power(hero)
            cv2.circle(frame, (x, y + 50), size, hero_power["power_color"], -1)
        
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
        # Display score
        cv2.putText(frame, f"Score: {self.score}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display current hero
        cv2.putText(frame, f"Hero: {self.current_hero}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display wave number
        cv2.putText(frame, f"Wave: {self.wave_number}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display health bar
        health_width = int((self.player_health / 100) * 200)
        cv2.rectangle(frame, (self.width - 220, 30), (self.width - 20, 50), (50, 50, 50), -1)
        cv2.rectangle(frame, (self.width - 220, 30), (self.width - 220 + health_width, 50), (0, 0, 255), -1)
        cv2.putText(frame, f"Health: {int(self.player_health)}%", (self.width - 210, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display collected powers
        for i in range(self.max_powers):
            if i < self.collected_powers:
                color = self.hero_effect.get_hero_power(self.current_hero)["power_color"]
                cv2.circle(frame, (self.width - 50 - i*30, 80), 10, color, -1)
            else:
                cv2.circle(frame, (self.width - 50 - i*30, 80), 10, (100, 100, 100), 2)
        
        return frame
    
    def render_targets(self, frame):
        # Draw each target
        for target in self.targets:
            # Draw target
            color = (0, 255, 0) if target['locked'] else target['color']
            cv2.circle(frame, (int(target['x']), int(target['y'])), target['size'], color, 2)
            
            # Draw health bar above target
            health_width = int((target['health'] / 100) * target['size'] * 2)
            cv2.rectangle(frame, (int(target['x'] - target['size']), int(target['y'] - target['size'] - 10)), 
                         (int(target['x'] + target['size']), int(target['y'] - target['size'] - 5)), 
                         (50, 50, 50), -1)
            cv2.rectangle(frame, (int(target['x'] - target['size']), int(target['y'] - target['size'] - 10)), 
                         (int(target['x'] - target['size'] + health_width), int(target['y'] - target['size'] - 5)), 
                         (0, 0, 255), -1)
            
            # Draw lock indicator if targeted
            if target['locked']:
                cv2.line(frame, (int(target['x'] - target['size'] - 10), int(target['y'] - target['size'] - 10)), 
                        (int(target['x'] - target['size']), int(target['y'] - target['size'])), (0, 255, 0), 2)
                cv2.line(frame, (int(target['x'] + target['size'] + 10), int(target['y'] - target['size'] - 10)), 
                        (int(target['x'] + target['size']), int(target['y'] - target['size'])), (0, 255, 0), 2)
                cv2.line(frame, (int(target['x'] - target['size'] - 10), int(target['y'] + target['size'] + 10)), 
                        (int(target['x'] - target['size']), int(target['y'] + target['size'])), (0, 255, 0), 2)
                cv2.line(frame, (int(target['x'] + target['size'] + 10), int(target['y'] + target['size'] + 10)), 
                        (int(target['x'] + target['size']), int(target['y'] + target['size'])), (0, 255, 0), 2)
        
        return frame
    
    def render_powers(self, frame):
        # Draw each power
        for power in self.powers:
            cv2.circle(frame, (int(power['x']), int(power['y'])), power['size'], power['color'], -1)
            
            # Draw trail based on effect
            if power['effect'] == "repulsor":
                cv2.circle(frame, (int(power['x'] - power['dx']), int(power['y'] - power['dy'])), 
                          power['size']//2, power['color'], -1)
            elif power['effect'] == "web":
                cv2.line(frame, (int(power['x']), int(power['y'])), 
                        (int(power['x'] - power['dx']*2), int(power['y'] - power['dy']*2)), 
                        power['color'], 2)
            elif power['effect'] == "lightning":
                for i in range(3):
                    offset_x = random.randint(-10, 10)
                    offset_y = random.randint(-10, 10)
                    cv2.line(frame, (int(power['x']), int(power['y'])), 
                            (int(power['x'] - power['dx'] + offset_x), int(power['y'] - power['dy'] + offset_y)), 
                            power['color'], 2)
            elif power['effect'] == "smash":
                cv2.circle(frame, (int(power['x']), int(power['y'])), 
                          power['size'] + random.randint(0, 5), power['color'], 2)
            elif power['effect'] == "shield":
                cv2.ellipse(frame, (int(power['x']), int(power['y'])), 
                           (power['size']*2, power['size']), 0, 0, 360, power['color'], 2)
        
        return frame
    
    def render_danger_zone(self, frame, faces):
        if len(faces) > 0 and self.face_position:
            # Draw danger zone around face
            cv2.circle(frame, self.face_position, self.danger_zone_radius, (0, 0, 255), 2)
        
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
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = AvengersGame()
    game.run() 