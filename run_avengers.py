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
    print("\n🌟 MARVEL AVENGERS EXPERIENCE - EASY MODE 🌟")
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
    # Check if assets folder exists, create if not
    if not os.path.exists("assets"):
        os.makedirs("assets")
        os.makedirs("assets/heroes", exist_ok=True)
        print("Created assets folders for storing hero effects")
    
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
        print("\nPlease make sure all files are in the correct location.")

if __name__ == "__main__":
    main() 