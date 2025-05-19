#!/usr/bin/env python
import os
import sys
import time

def print_intro():
    """Print intro message with ASCII art."""
    print(r"""
  _____                                         _____                      
 |  _  |                                       |  ___|                     
 | | | |_   _ _ __    | |_| |__   ___  | | | | | |__  _ __   __ _  ___ ___ 
 | | | | | | | '_ \   | __| '_ \ / _ \ | | | | |  __|| '_ \ / _` |/ _ / __|
 \ \_/ | |_| | | | |  | |_| | | |  __/ \ \_/ / | |___| | | | (_| |  __\__ \
  \___/ \__,_|_| |_|   \__|_| |_|\___|  \___/  \____/|_| |_|\__, |\___|___/
                                                              __/ |         
                                                             |___/          
    """)
    print("\n🌟 MARVEL AVENGERS EXPERIENCE 🌟")
    print("=========================================")
    print("Join your favorite heroes to fight against evil targets in this beginner-friendly version!")
    print("\n✨ FEATURES:")
    print("- Choose from 5 heroes: Iron Man, Spider-Man, Thor, Hulk, Captain America")
    print("- Hero-specific powers with enhanced visual effects")
    print("- Stunning kill animations unique to each hero")
    print("- Reduced enemy numbers and slower speeds for easier gameplay")
    print("- Improved targeting system for better accessibility")
    print("\n🎮 CONTROLS:")
    print("- Point with your index finger: Target enemies")
    print("- Make a fist: Collect powers (now collect up to 5)")
    print("- Open palm: Use powers to attack targeted enemies")
    print("- Rock-on gesture (index and pinky up): Open hero selection menu")
    print("- Press 'q' to quit, 'm' for menu")
    print("\n💡 TIPS:")
    print("- Enemies move slower now, giving you more time to react")
    print("- Powers recharge faster for more frequent attacks")
    print("- Look for the special effects when you defeat enemies!")
    print("- Stay away from enemies to prevent health loss")
    print("\nStarting game in 3 seconds...")
    time.sleep(3)

def check_requirements():
    """Check if required libraries are installed."""
    try:
        import cv2
        import mediapipe
        import numpy
    except ImportError as e:
        print(f"Error: Missing required library - {e.name}")
        print("\nPlease install required libraries with:")
        print("pip install opencv-python mediapipe numpy")
        return False
    
    # Check camera availability
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera")
        print("Please check your camera connection and permissions")
        cap.release()
        return False
    cap.release()
    
    return True

def main():
    """Main function to run the game."""
    # Create necessary directories
    heroes = ["iron_man", "spider_man", "thor", "hulk", "captain_america"]
    
    # Create base assets directory
    if not os.path.exists("assets"):
        os.makedirs("assets")
    
    # Create heroes directory
    heroes_dir = os.path.join("assets", "heroes")
    if not os.path.exists(heroes_dir):
        os.makedirs(heroes_dir)
    
    # Create individual hero directories
    for hero in heroes:
        hero_dir = os.path.join(heroes_dir, hero)
        if not os.path.exists(hero_dir):
            os.makedirs(hero_dir)
            print(f"Created directory for {hero}")
    
    print("\nAssets directories are ready!")
    print("Place your custom mask.png and effect.png files in each hero's directory")
    print("Default placeholder images will be used if custom assets are not found\n")
    
    # Print intro
    print_intro()
    
    # Check requirements
    if not check_requirements():
        print("\nExiting due to missing requirements.")
        return
    
    print("\nStarting Avengers game in easy mode...")
    
    # Run the game
    try:
        from avengers_game import AvengersGame
        game = AvengersGame()
        game.run()
    except Exception as e:
        print(f"Error running game: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease make sure all files are in the correct location.")

if __name__ == "__main__":
    main() 