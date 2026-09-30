import cv2
import numpy as np
import os

def generate_image(filename, text, size, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """
    Generates a simulated OCR image with specific dimensions and text.
    No rotation or angle changes applied.
    """
    # Create background image
    img = np.ones((size[1], size[0], 3), dtype=np.uint8)
    img[:] = bg_color
    
    # Font settings (same as templates for best matching)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    # Get text size to perfectly center it within whatever aspect ratio we give it
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Calculate position to center the text
    text_x = (size[0] - text_size[0]) // 2
    text_y = (size[1] + text_size[1]) // 2
    
    # Draw text
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # Save image
    cv2.imwrite(filename, img)
    print(f"[Success] Generated {filename} | Size: {size[0]}x{size[1]} | Text: '{text}'")

if __name__ == "__main__":
    print("--- Generating Simulated Images with Different Frame Orientations ---")
    
    # Generate 1:1 ratio (Square)
    generate_image("simulated_square_1x1.png", "AERO", (400, 400))
    
    # Generate 3:4 ratio (Portrait)
    generate_image("simulated_portrait_3x4.png", "VISION", (300, 400))
    
    # Generate 4:3 ratio (Landscape)
    generate_image("simulated_landscape_4x3.png", "DATA", (400, 300))
    
    # Generate 16:9 ratio (Widescreen)
    generate_image("simulated_widescreen_16x9.png", "SYSTEM", (640, 360))
    
    # Generate another 1:1 ratio with a different background shade (dark gray)
    generate_image("simulated_dark_1x1.png", "VIPIN", (500, 500), bg_color=(100, 100, 100), text_color=(255, 255, 255))
    
    print("--- Done! You can now test these on the frontend. ---")
