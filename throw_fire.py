import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
from fire_effects import FireEffect

class FireThrower:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize fire effects
        self.fire_effect = FireEffect(self.width, self.height)
        
        # Store fire locations (x, y, scale, rotation, frame_created)
        self.fire_locations = []
        
        # Tracking previous finger positions
        self.prev_thumb_tip = None
        self.prev_index_tip = None
        
        # State variables
        self.collecting_fire = False
        self.throwing_fire = False
        self.throw_start_time = 0
        self.throw_duration = 1.5  # seconds
        
        # Animation parameters
        self.collected_fires = []
        self.thrown_fires = []
        
        # Frame counter for effects
        self.frame_count = 0
    
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
    
    def run(self):
        last_pinch_time = 0
        pinch_cooldown = 0.2  # seconds between pinch detection
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            results = self.hands.process(rgb_frame)
            
            # Increment frame counter
            self.frame_count += 1
            
            # Draw fires that have already been placed
            for i, loc in enumerate(self.fire_locations):
                x, y, scale, rotation, _ = loc
                
                # Get animated fire with subtle movement
                fire_img, scale_mod, rot_mod = self.fire_effect.create_small_fire(self.frame_count + i * 10)
                
                # Apply animation
                current_scale = scale * scale_mod
                current_rotation = rotation + rot_mod
                
                # Draw the fire
                frame = self.fire_effect.overlay_image(frame, fire_img, x, y, current_scale, current_rotation)
            
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
                        x, y, scale, rotation = self.fire_effect.create_thrown_fire(
                            start_pos, target_pos, fire_progress, size)
                        
                        # Get animated fire
                        fire_img, _, _ = self.fire_effect.create_small_fire(self.frame_count)
                        
                        # Draw the fire
                        frame = self.fire_effect.overlay_image(frame, fire_img, x, y, scale, rotation)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Check for pinch gesture
                    is_pinching, pinch_x, pinch_y = self.is_pinch_gesture(hand_landmarks)
                    
                    # If pinching and cooldown has elapsed
                    current_time = time.time()
                    if is_pinching and current_time - last_pinch_time > pinch_cooldown and not self.collecting_fire and not self.throwing_fire:
                        # Add a new fire at the pinch location
                        self.fire_locations.append((pinch_x, pinch_y, 1.0, 0.0, self.frame_count))
                        last_pinch_time = current_time
                    
                    # Check for fist gesture
                    if self.is_fist_gesture(hand_landmarks) and not self.collecting_fire and not self.throwing_fire and len(self.fire_locations) > 0:
                        self.collecting_fire = True
                        
                        # Calculate hand position for collecting fires
                        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        hand_x, hand_y = int(wrist.x * self.width), int(wrist.y * self.height)
                        
                        # Store collected fires
                        self.collected_fires = self.fire_locations.copy()
                        self.fire_locations = []
                        
                        print("Collecting fires into fist!")
                    
                    # Check for open palm gesture to throw collected fires
                    if self.is_open_palm_gesture(hand_landmarks) and self.collecting_fire and not self.throwing_fire:
                        self.collecting_fire = False
                        self.throwing_fire = True
                        self.throw_start_time = time.time()
                        
                        # Get hand center position
                        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        hand_x, hand_y = int(wrist.x * self.width), int(wrist.y * self.height)
                        
                        # Create thrown fires with start and target positions
                        self.thrown_fires = self.fire_effect.create_fire_particle_system(
                            hand_x, hand_y, num_particles=30, spread=300, intensity=1.5)
                        
                        print("Throwing fire!")
            
            # Display instructions
            cv2.putText(frame, "Pinch: Create fire", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Fist: Collect fires", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Open palm: Throw fire", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show fire counter
            cv2.putText(frame, f"Fires: {len(self.fire_locations)}", (self.width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Throw Fire", frame)
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    fire_thrower = FireThrower()
    fire_thrower.run() 