import cv2
import numpy as np
import os

class TraditionalOCR:
    """
    A pure traditional AI / Computer Vision OCR system.
    NO Neural Networks. NO Machine Learning. NO Deep Learning.
    
    How it works:
    1. Template Generation: Renders standard characters (A-Z, 0-9) as templates.
    2. Segmentation: Uses OpenCV contours to find characters in the input image.
    3. Template Matching: Compares the cropped input character against all templates
       using pixel-wise Mean Squared Error (MSE) to find the closest match.
    """
    
    def __init__(self):
        self.chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.templates = {}
        self.generate_templates()
        
    def generate_templates(self):
        """Generates standard font templates for all characters."""
        print("[System] Generating pure AI templates...")
        for char in self.chars:
            # Create a blank black image
            img = np.zeros((50, 50), dtype=np.uint8)
            
            # Put white text on it
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Get text size to center it
            text_size = cv2.getTextSize(char, font, 1.5, 3)[0]
            text_x = (50 - text_size[0]) // 2
            text_y = (50 + text_size[1]) // 2
            
            cv2.putText(img, char, (text_x, text_y), font, 1.5, 255, 3)
            
            # Crop tight bounding box around the character
            x, y, w, h = cv2.boundingRect(img)
            if w > 0 and h > 0:
                cropped = img[y:y+h, x:x+w]
                # Standardize all templates to 28x28
                resized = cv2.resize(cropped, (28, 28))
                self.templates[char] = resized

    def preprocess_and_segment(self, image_path):
        """Reads image, applies robust CV processing, and finds character bounding boxes."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # 1. Blur to remove high frequency noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 2. Adaptive Binarization (Handles varying lighting/shadows in random images)
        # Detect if the image is mostly dark (light text on dark background) or light (dark text on light background)
        avg_brightness = np.mean(img)
        print(f"[System] Average brightness: {avg_brightness:.2f}")
        
        if avg_brightness < 127:
            # Dark background, light text -> Use standard binary threshold (text becomes white)
            thresh_type = cv2.THRESH_BINARY
            c_val = -5
            print("[System] Detected dark image. Using THRESH_BINARY with C=-5.")
        else:
            # Light background, dark text -> Invert it so text becomes white
            thresh_type = cv2.THRESH_BINARY_INV
            c_val = 5
            print("[System] Detected light image. Using THRESH_BINARY_INV with C=5.")
            
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       thresh_type, 15, c_val)
                                       
        # 3. Morphological operations to clean up noise and connect broken character parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # Remove small noise specks
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # Close small holes in letters
        
        # 4. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from left to right
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
        
        img_h, img_w = img.shape
        characters = []
        
        for x, y, w, h in bounding_boxes:
            # Filter out tiny noise and gigantic contours (e.g. bounding box of the whole image)
            if w > 2 and h > 10 and w < img_w * 0.9 and h < img_h * 0.9:
                char_crop = thresh[y:y+h, x:x+w]
                # Resize to match template size
                char_resized = cv2.resize(char_crop, (28, 28))
                characters.append(char_resized)
                
        return characters

    def recognize_character(self, char_img):
        """Finds the best matching template using Normalized Cross-Correlation."""
        best_match = "?"
        highest_score = -1.0
        
        for char, template in self.templates.items():
            # cv2.matchTemplate with TM_CCOEFF_NORMED is robust to varying contrast/brightness
            result = cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]
            
            if score > highest_score:
                highest_score = score
                best_match = char
                
        return best_match

    def recognize_image(self, image_path):
        """Full pipeline: segment -> match -> output text."""
        print(f"\n[System] Reading image: {image_path}")
        try:
            characters = self.preprocess_and_segment(image_path)
            print(f"[System] Found {len(characters)} characters.")
            
            result_text = ""
            for char_img in characters:
                pred = self.recognize_character(char_img)
                result_text += pred
                
            print(f">>> FINAL RECOGNITION RESULT: '{result_text}'")
            return result_text
        except Exception as e:
            print(f"Error: {e}")
            return None

# ----- HOW TO RUN -----
if __name__ == "__main__":
    print("==================================================")
    print(" PURE TRADITIONAL AI - TEXT RECOGNITION SYSTEM")
    print("==================================================")
    ocr = TraditionalOCR()
    
    print("\n--- SIMULATING 5 PERFECTLY PRINTED PHOTOS ---")
    phrases = ["HELLO", "WORLD", "AI 2026", "ABCD", "12345"]
    
    for i, phrase in enumerate(phrases, 1):
        img_name = f"simulated_{i}.png"
        
        # Create a white background image
        test_img = np.ones((100, 400), dtype=np.uint8) * 255
        
        # Write the text using the exact same font the templates use
        cv2.putText(test_img, phrase, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
        cv2.imwrite(img_name, test_img)
        print(f"[System] Created simulated image: {img_name} with text: {phrase}")
        
        # Run recognition
        ocr.recognize_image(img_name)
    
    print("\n--- RUNNING USER UPLOADED IMAGES ---")
    ocr.recognize_image("test_input.png")
    ocr.recognize_image("simulated_1.png")
    ocr.recognize_image("simulated_2.png")
    ocr.recognize_image("simulated_3.png")
    ocr.recognize_image("simulated_4.png")
    ocr.recognize_image("simulated_5.png")