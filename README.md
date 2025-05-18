# Marvel Avengers Gesture-Based Game

A fun, interactive game that lets you become your favorite Marvel hero using computer vision and gesture controls. This version features easier gameplay for beginners, drone enemies, and spectacular hero-specific visual effects!

![Avengers Logo](https://i.imgur.com/YbqKVo4.png)

## Overview

The Avengers Game uses your webcam and hand gestures to let you control Marvel superheroes and defeat enemy drones. With enhanced visual effects and simplified gameplay, it's accessible for players of all skill levels.

## Features

- **5 Playable Heroes**: Iron Man, Spider-Man, Thor, Hulk, and Captain America
- **Hero-Specific Powers**: Each hero has unique attack animations and visual effects:
  - Iron Man: Repulsor beam with blue energy glow
  - Spider-Man: White web shooters with detailed strand effects
  - Thor: Lightning bolts with branches and glow effects
  - Hulk: Green fist with impact shockwaves
  - Captain America: Shield with star and motion trail
- **Enemy Drones**: Visually detailed drone enemies instead of simple circles
- **Gesture Controls**: Use intuitive hand gestures to target enemies and unleash superhero powers
- **Dynamic Difficulty**: Enemies spawn in waves with slight increases in difficulty
- **Spectacular Kill Effects**: Satisfy your inner hero with amazing visual effects when defeating enemies
- **Easy Mode**: Reduced enemy numbers and speed for a more accessible experience

## Requirements

- Python 3.7+
- Webcam
- Libraries:
  - OpenCV
  - Mediapipe
  - NumPy

## Installation

1. Clone this repository:

```
git clone https://github.com/yourusername/avengers-game.git
cd avengers-game
```

2. Install required packages:

```
pip install opencv-python mediapipe numpy
```

3. Run the game:

```
python run_avengers.py
```

## Gameplay Controls

| Control | Action |
|---------|--------|
| Point with index finger | Target enemies |
| Make a fist | Collect power (up to 5 charges) |
| Open palm | Use power to attack targeted enemy |
| Press 'C' key | Open hero selection menu |
| Press 'q' | Quit game |
| Press 'm' | Return to menu |

## Hero Powers

Each hero has a unique power with different visual effects:

- **Iron Man**: Repulsor Beam - A bright blue energy beam with glowing rings
- **Spider-Man**: Web Shooter - White webs with detailed strand patterns
- **Thor**: Lightning - Jagged lightning bolts with multiple branches and glow effects
- **Hulk**: Smash - Green fist with motion blur and impact shockwaves
- **Captain America**: Shield - Animated shield with star and motion trail

## Game Objective

Protect yourself from incoming drone enemies by defeating them before they reach you. Each wave will spawn a small number of enemies that move toward your position. Target them with your index finger and unleash your hero's power to defeat them!

## Customization

The game can be further customized by editing these files:
- `avengers_game.py`: Main game logic
- `hero_effects.py`: Visual effects for each hero
- `run_avengers.py`: Game launcher and intro display

## Credits

- Developed using OpenCV and MediaPipe
- Inspired by Marvel's Avengers characters
- Created for educational and entertainment purposes

## License

This project is for educational purposes only. All Marvel characters are property of Marvel Entertainment and Disney.

---

Enjoy the game! If you have suggestions or issues, please submit them on GitHub! 