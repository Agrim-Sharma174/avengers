# Customization Guide for Throw Fire

This guide explains how to customize and extend the "Throw Fire" application.

## Fire Appearance

### Changing Fire Colors

In both versions of the application, you can modify the fire colors by editing the `create_fire_image` function.

Look for this section in the code:

```python
# Base (orange)
cv2.ellipse(img, (width//2, int(height*0.65)), (int(width*0.3), int(height*0.4)), 
           0, 0, 360, (0, 69, 255, 255), -1)

# Middle (red)
cv2.ellipse(img, (width//2, int(height*0.5)), (int(width*0.2), int(height*0.35)), 
           0, 0, 360, (0, 0, 255, 255), -1)

# Top (yellow)
cv2.ellipse(img, (width//2, int(height*0.4)), (int(width*0.1), int(height*0.25)), 
           0, 0, 360, (0, 165, 255, 255), -1)
```

The color values are in BGRA format (Blue, Green, Red, Alpha). For example, to make a blue fire, change:
- For the base: `(0, 69, 255, 255)` to `(255, 69, 0, 255)`
- For the middle: `(0, 0, 255, 255)` to `(255, 0, 0, 255)`
- For the top: `(0, 165, 255, 255)` to `(255, 165, 0, 255)`

### Using Custom Fire Images

Instead of generating the fire programmatically, you can use custom images:

1. Create a PNG image with transparency (for the flame)
2. Place it in the `assets` directory
3. In `throw_fire.py` or `simple_fire.py`, modify the initialization code:

```python
# Instead of generating fire image
self.fire_img = cv2.imread("assets/your_custom_fire.png", cv2.IMREAD_UNCHANGED)
if self.fire_img is None:
    # Fallback to generated fire
    self.fire_img = self.create_fire_image(100, 100)
self.small_fire_img = cv2.resize(self.fire_img, (50, 50))
```

## Animation Parameters

### Fire Size and Speed

To adjust fire size:
- Change the resize values: `cv2.resize(self.fire_img, (50, 50))` to a different size
- Modify the scale parameters in the animation code

To adjust fire throwing speed:
- Change `self.throw_duration = 1.5  # seconds` to a higher or lower value

### Adding More Fire Particles

To increase the number of particles when throwing fire:
```python
self.thrown_fires = self.create_fire_particle_system(
    hand_x, hand_y, num_particles=30, spread=300, intensity=1.5)
```

Increase `num_particles` for more particles, and adjust `spread` to control how far they travel.

## Gesture Recognition

### Adjusting Pinch Sensitivity

In the full version (`throw_fire.py`), you can modify the pinch detection threshold:

```python
# If distance is small enough, consider it a pinch
return distance < 30, (thumb_x + index_x) // 2, (thumb_y + index_y) // 2
```

Change `distance < 30` to a higher value for easier pinch detection, or lower for more precise detection.

### Adding New Gestures

To add a new gesture:

1. Create a new detection function, following the pattern of the existing ones:
```python
def is_new_gesture(self, hand_landmarks):
    # Detection logic here
    return is_detected
```

2. Add the gesture detection to the main loop and implement the action

## Advanced Customizations

### Adding Sound Effects

To add sound effects, install the `pygame` library:

```python
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load sounds
fire_sound = pygame.mixer.Sound("assets/fire_sound.wav")
throw_sound = pygame.mixer.Sound("assets/throw_sound.wav")

# Then play the sounds at appropriate moments
fire_sound.play()
```

### Creating Particle Effects

You can enhance the fire effects by adding more varied particles:

1. Add a particle class to manage individual particles
2. Add properties like lifetime, color variation, opacity
3. Update particles in each frame

Example particle implementation:
```python
class FireParticle:
    def __init__(self, x, y, velocity_x, velocity_y, size, lifetime):
        self.x = x
        self.y = y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.size = size
        self.lifetime = lifetime
        self.age = 0
        
    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.age += 1
        # Add gravity effect
        self.velocity_y += 0.1
        
    def is_alive(self):
        return self.age < self.lifetime
```

### Adding Color Modes

You can implement different color themes for the fire:
1. Create several fire generation functions for different colors
2. Add a keyboard shortcut to switch between them

Example:
```python
if key == ord('b'):  # Press 'b' for blue fire
    self.fire_mode = "blue"
elif key == ord('g'):  # Press 'g' for green fire
    self.fire_mode = "green"
```

## Troubleshooting Customizations

- If your custom image doesn't display properly, check that it has a proper alpha channel
- If animations are too slow, reduce particle count or simplify effects
- If gesture detection becomes unreliable after modifications, revert to the original detection parameters 