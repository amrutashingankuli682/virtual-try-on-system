import cv2
import numpy as np
import os
# Load the garment image
garment_path = "myntra_garment2.jpg"
garment = cv2.imread(garment_path)

# Check if the image is loaded properly
if garment is None or garment.size == 0:
    print("Error: Failed to load garment image or empty dimensions")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(garment, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to create alpha channel
    _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Split the BGR channels
    b, g, r = cv2.split(garment)
    
    # Merge BGR channels with the alpha channel to get RGBA
    rgba = [b, g, r, alpha]
    garment_rgba = cv2.merge(rgba)

    directory_path = "C:\\Users\\HP\\Desktop\\myntra\\"

    if os.access(directory_path, os.W_OK):
      print(f"Write permission is granted for directory: {directory_path}")
    else:
      print(f"Write permission is not granted for directory: {directory_path}")

    # Save the RGBA image
    output_path = "C:\\Users\\HP\\Desktop\\myntra\\myntra_garment2_rgba.png"
    cv2.imwrite(output_path, garment_rgba)
    print(f"Image saved successfully at {output_path}")
    garment_width = 100  # Example width
    garment_height = 150  # Example height

    # Resize the garment image
    resized_garment = cv2.resize(garment, (garment_width, garment_height), interpolation=cv2.INTER_AREA)

    # Display or save the resized garment image
    cv2.imshow('Resized Garment', resized_garment)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
