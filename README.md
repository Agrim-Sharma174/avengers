# Marvel Avengers Gesture Control Game

An interactive game that lets you become your favorite Marvel Avenger using computer vision and hand gestures. Fight against evil drones using hero-specific powers and stunning visual effects!

## Features

- **5 Playable Heroes**:
  - Iron Man: Repulsor beam with energy rings
  - Spider-Man: Web shooter with detailed web patterns
  - Thor: Lightning bolts with dynamic branching
  - Hulk: Powerful smash with shockwaves
  - Captain America: Shield throw with bounce effects

- **Gesture Controls**:
  - Point with index finger to target enemies
  - Make a fist to collect power (up to 5 charges)
  - Open palm to unleash your hero's power
  - Press 'C' to open hero selection menu
  - Press 'M' to return to menu
  - Press 'Q' to quit game

- **Game Elements**:
  - Dynamic enemy drones with unique behaviors
  - Wave-based progression system
  - Health and power management
  - Score tracking
  - Hero-specific kill animations
  - Face detection for hero mask overlay

## Requirements

- Python 3.7 or higher
- Webcam
- Required Python packages (install via pip):
  ```
  pip install -r requirements.txt
  ```

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the game:
   ```
   python run_avengers.py
   ```

## Game Structure

- `run_avengers.py`: Main entry point and game launcher
- `avengers_game.py`: Core game logic and mechanics
- `hero_effects.py`: Hero-specific visual effects and powers
- `assets/`: Directory containing hero masks and effect images

## Gameplay Tips

- Keep distance from enemies to prevent health loss
- Collect powers when not in immediate danger
- Use hero powers strategically - each hero has unique strengths
- Watch your health bar and avoid getting surrounded
- Target enemies that are closest to you first

## Hero Powers

Each hero has unique abilities and effects:

- **Iron Man**: Repulsor beam with energy rings and pulsing effects
- **Spider-Man**: Web shooter with detailed strand patterns and connecting webs
- **Thor**: Lightning strikes with dynamic branching and thunder effects
- **Hulk**: Ground-shaking smash with shockwaves and debris
- **Captain America**: Shield throw with star effects and motion trails

## Customization

The game can be customized in various ways:
- Add custom hero masks in `assets/heroes/<hero_name>/mask.png`
- Add custom effect images in `assets/heroes/<hero_name>/effect.png`
- Modify hero powers and effects in `hero_effects.py`
- Adjust game difficulty settings in `avengers_game.py`

See `CUSTOMIZATION.md` for detailed customization instructions.

## Credits

- Developed using OpenCV and MediaPipe for computer vision
- Inspired by Marvel's Avengers characters
- Created for educational and entertainment purposes

## License

This project is for educational purposes only. All Marvel characters and related elements are property of Marvel Entertainment and Disney.

## Support

If you encounter any issues or have suggestions:
1. Check the known issues in the documentation
2. Ensure your webcam is working properly
3. Verify you have all required dependencies installed
4. Make sure you have sufficient lighting for gesture detection 