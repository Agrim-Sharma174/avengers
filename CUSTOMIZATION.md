# Customization Guide for Marvel Avengers Game

This guide explains how to customize and extend the Marvel Avengers game.

## Hero Assets

### Custom Hero Masks

Each hero can have a custom mask image that appears over the player's face:

1. Create a PNG image with transparency (recommended size: 200x200 pixels)
2. Save it as `mask.png` in the hero's directory:
   ```
   assets/heroes/iron_man/mask.png
   assets/heroes/spider_man/mask.png
   assets/heroes/thor/mask.png
   assets/heroes/hulk/mask.png
   assets/heroes/captain_america/mask.png
   ```

### Custom Power Effects

Each hero can have custom power effect images:

1. Create a PNG image with transparency (recommended size: 100x100 pixels)
2. Save it as `effect.png` in the hero's directory:
   ```
   assets/heroes/iron_man/effect.png
   assets/heroes/spider_man/effect.png
   assets/heroes/thor/effect.png
   assets/heroes/hulk/effect.png
   assets/heroes/captain_america/effect.png
   ```

## Game Modifications

### Adjusting Hero Powers

In `hero_effects.py`, you can modify each hero's power attributes:

```python
self.hero_assets = {
    "iron_man": {
        "power_type": "beam",
        "power_speed": 2.0,
        "power_damage": 3,
        "power_color": (0, 200, 255)
    },
    # ... other heroes ...
}
```

Parameters you can adjust:
- `power_type`: Effect type ("beam", "web", "lightning", "smash", "shield")
- `power_speed`: Projectile speed (higher = faster)
- `power_damage`: Damage dealt to enemies
- `power_color`: RGB color tuple for power effects

### Modifying Game Difficulty

In `avengers_game.py`, you can adjust these game parameters:

```python
# Enemy settings
self.max_targets = 3          # Maximum enemies at once
self.target_spawn_interval = 3.0  # Seconds between spawns
self.enemies_per_wave = 3     # Enemies per wave
self.danger_zone_radius = 150 # Safe distance from enemies

# Player settings
self.max_powers = 5           # Maximum stored powers
self.power_cooldown = 0.5     # Seconds between power uses
self.player_health = 100      # Starting health
```

### Enemy Types

Customize enemy drones in `avengers_game.py`:

```python
self.enemy_types = [
    {
        "name": "Drone",
        "size": 40,
        "speed": 0.5,
        "health": 1,
        "color": (0, 0, 255),
        "points": 10
    },
    # ... add more enemy types ...
]
```

## Visual Effects

### Modifying Hero Effects

Each hero's power effect can be customized in `hero_effects.py`:

#### Iron Man
```python
# Repulsor beam
cv2.line(frame, (x, y), (target_x, target_y), (0, 200, 255), 3)
cv2.circle(frame, (x, y), size, (255, 255, 255), -1)
```

#### Spider-Man
```python
# Web pattern
cv2.line(frame, points[i], points[i+1], (255, 255, 255), 2)
cv2.circle(frame, (x, y), size//2, (200, 200, 200), -1)
```

#### Thor
```python
# Lightning effect
cv2.line(frame, points[i], points[i+1], (255, 255, 100), 4)
cv2.circle(frame, (target_x, target_y), impact_size, (255, 255, 100), 2)
```

#### Hulk
```python
# Smash effect
cv2.circle(frame, (target_x, target_y), radius, (0, 255, 0), 2)
cv2.polylines(frame, [pts], False, (0, 255, 0), 2)
```

#### Captain America
```python
# Shield effect
cv2.circle(frame, (x, y), size, (0, 0, 150), -1)
cv2.circle(frame, (x, y), int(size * 0.8), (200, 0, 0), -1)
```

## UI Customization

### Game Interface

Modify the UI elements in `avengers_game.py`:

```python
def render_ui(self, frame):
    # Score display
    cv2.putText(frame, f"Score: {self.score}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (255, 255, 255), 2)
    
    # Health bar
    cv2.rectangle(frame, (self.width - 220, 30), 
                 (self.width - 220 + health_width, 50), 
                 (0, 0, 255), -1)
```

### Menu Customization

Modify the hero selection menu in `avengers_game.py`:

```python
def show_menu(self, frame, hand_landmarks=None):
    # Menu title
    cv2.putText(frame, "AVENGERS HERO SELECTION", 
                (self.width//2 - 250, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
```

## Advanced Customization

### Adding New Heroes

1. Add hero assets to the `assets/heroes/` directory
2. Add hero configuration to `hero_assets` in `hero_effects.py`
3. Create custom power effects in `create_targeted_effect()`
4. Add hero to the selection menu in `show_menu()`

### Adding New Enemy Types

1. Create new enemy design in `create_drone_image()`
2. Add enemy type to `enemy_types` list
3. Customize enemy behavior in `update_targets()`

### Creating Custom Effects

1. Implement effect generation in `hero_effects.py`
2. Add effect rendering in `render_powers()`
3. Trigger effects in appropriate game events

## Troubleshooting

- If custom images don't load, check file paths and image formats
- For performance issues, reduce effect complexity or number of particles
- If gestures aren't detecting well, adjust detection thresholds
- For memory issues, optimize image sizes and effect durations 