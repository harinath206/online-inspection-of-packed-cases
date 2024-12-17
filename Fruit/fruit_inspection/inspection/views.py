from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from django.core.files.storage import default_storage
# from sklearn.cluster import KMeans
from .models import Feedback
from .forms import FeedbackForm
import cv2
import numpy as np
import joblib
import os

# Model save/load path
MODEL_DIR = 'inspection/models'
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_fruit_quality_model.pkl')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image temporarily
            image_file = request.FILES['image']
            image_path = default_storage.save('tmp/' + image_file.name, image_file)
            image_path = default_storage.path(image_path)
            
            # Analyze the image and get the result
            result = analyze_apple(image_path)

            # Clean up the temporary file
            default_storage.delete(image_path)

            print(f"Result dictionary: {result}")  # Ensure this prints correctly

            return render(request, 'inspection/result.html', {'result': result})
    else:
        form = ImageUploadForm()

    # Render the upload form for a GET request, no result is passed yet
    return render(request, 'inspection/index.html', {'form': form})


def analyze_apple(image_path):
    """Analyze the uploaded image and return the analysis results for apples."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return {"status": "Image not found"}

    # Convert to RGB and HSV color spaces
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for ripe (red), unripe (green), and damaged (black) apples
    lower_red = np.array([0, 70, 50])      # Adjusted for ripe apples
    upper_red = np.array([10, 255, 255]) 
    
    lower_green = np.array([35, 50, 50])   # Adjusted for unripe apples
    upper_green = np.array([85, 255, 255])

    lower_black = np.array([0, 0, 0])      # Adjusted for damaged apples (dark/black)
    upper_black = np.array([50, 50, 50])

    # Create masks for ripe (red), unripe (green), and damaged (black) apples
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

    # Calculate the percentage of red, green, and black pixels in the image
    red_percentage = np.sum(red_mask) / (image.shape[0] * image.shape[1]) * 100
    green_percentage = np.sum(green_mask) / (image.shape[0] * image.shape[1]) * 100
    black_percentage = np.sum(black_mask) / (image.shape[0] * image.shape[1]) * 100

    # Analyze color and ripeness based on thresholds
    if red_percentage > 50:
        color_status = "Red"
        ripeness_status = "Ripe"
    elif green_percentage > 50:
        color_status = "Green"
        ripeness_status = "Unripe"
    elif black_percentage > 0:
        color_status = "Black"
        ripeness_status = "Ripe"  # Considered ripe but bad due to damage
    else:
        color_status = "Unknown"
        ripeness_status = "Unknown"

    # Analyze overall status
    if black_percentage > 20:
        overall_status = "Bad"
    elif ripeness_status == "Ripe" and color_status == "Red":
        overall_status = "Good"
    elif ripeness_status == "Unripe" and color_status == "Green":
        overall_status = "Good"
    else:
        overall_status = "Bad"

    result = {

        'color_check': 'True',
        'ripeness_check': 'True',
        'uniformity_check': 'True',
        'color_status': color_status,
        'ripeness_status': ripeness_status,
        'overall_status': overall_status
    }

    return result

def submit_feedback(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        image_name = request.POST.get('image_name')  # Retrieve image name

        if form.is_valid():
            feedback_value = form.cleaned_data['feedback']

            # Save the feedback to the database
            Feedback.objects.create(image_name=image_name, feedback=feedback_value)

            # Show a response based on feedback
            if feedback_value == 'yes':
                response_message = "Thank you for your feedback! We're glad you found it helpful."
            else:
                response_message = "Oops! We'll learn from this and strive to improve."

            return render(request, 'inspection/thank_you.html', {
                'response_message': response_message,
                'image_name': image_name
            })

    return redirect('upload_image')

# def analyze_grapes(image_path):
#     """Analyze the uploaded image and return the analysis results for grapes."""
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         return {"status": "Image not found"}

#     # Convert to RGB and HSV color spaces
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define color ranges for ripe (green), ripe (black), and unripe (green) grapes
#     lower_green = np.array([35, 50, 50])   # Adjusted for unripe apples
#     upper_green = np.array([85, 255, 255])

#     lower_black = np.array([0, 0, 0])       # Adjusted for ripe (black) grapes
#     upper_black = np.array([50, 50, 50])

#     # Create masks for ripe (green), unripe (green), and ripe (black) grapes
#     green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
#     black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

#     # Calculate the percentage of green and black pixels in the image
#     green_percentage = np.sum(green_mask) / (image.shape[0] * image.shape[1]) * 100
#     black_percentage = np.sum(black_mask) / (image.shape[0] * image.shape[1]) * 100

#     # Analyze color and ripeness based on thresholds
#     if green_percentage > 50:
#         if green_percentage > 70:  # Higher percentage of green for ripe status
#             color_status = "Green"
#             ripeness_status = "Ripe"
#         else:
#             color_status = "Green"
#             ripeness_status = "Unripe"
#     elif black_percentage > 0:
#         color_status = "Black"
#         ripeness_status = "Ripe"
#     else:
#         color_status = "Unknown"
#         ripeness_status = "Unknown"

#     # Analyze if the grape bunch is damaged based on black color dominance or unusual patterns
#     if black_percentage > 20 and ripeness_status == "Ripe":
#         overall_status = "Good"
#     elif color_status == "Green" and ripeness_status == "Ripe":
#         overall_status = "Good"
#     elif color_status == "Green" and ripeness_status == "Unripe":
#         overall_status = "Good"
#     else:
#         overall_status = "Bad"

#     result = {
#         'color_status': color_status,
#         'ripeness_status': ripeness_status,
#         'overall_status': overall_status
#     }

#     return result

# def analyze_orange(image_path):
#     """Analyze the uploaded image and return the analysis results for oranges."""
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         return {"status": "Image not found"}

#     # Convert to RGB and HSV color spaces
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define color ranges for ripe (orange), unripe (green), and damaged (black) oranges
#     lower_orange = np.array([10, 100, 100])   # Adjusted for ripe oranges
#     upper_orange = np.array([25, 255, 255])
    
#     lower_green = np.array([35, 50, 50])   # Adjusted for unripe apples
#     upper_green = np.array([85, 255, 255])

#     lower_black = np.array([0, 0, 0])         # Adjusted for damaged oranges (black)
#     upper_black = np.array([50, 50, 50])

#     # Create masks for ripe (orange), unripe (green), and damaged (black) oranges
#     orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
#     green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
#     black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

#     # Calculate the percentage of orange, green, and black pixels in the image
#     orange_percentage = np.sum(orange_mask) / (image.shape[0] * image.shape[1]) * 100
#     green_percentage = np.sum(green_mask) / (image.shape[0] * image.shape[1]) * 100
#     black_percentage = np.sum(black_mask) / (image.shape[0] * image.shape[1]) * 100

#     # Analyze color and ripeness based on thresholds
#     if orange_percentage > 100:
#         color_status = "Orange"
#         ripeness_status = "Ripe"
#         overall_status = "Good"
#     elif green_percentage > 50:
#         color_status = "Green"
#         ripeness_status = "Unripe"
#         overall_status = "Good"
#     elif black_percentage > 0:
#         color_status = "Black"
#         ripeness_status = "Ripe"
#         overall_status = "Bad"
#     else:
#         color_status = "Unknown"
#         ripeness_status = "Unknown"
#         overall_status = "Bad"

#     result = {
#         'color_status': color_status,
#         'ripeness_status': ripeness_status,
#         'overall_status': overall_status
#     }

#     return result


# def analyze_images(image_paths):
#     """Analyze a list of images and return their results."""
#     results = []

#     for image_path in image_paths:
#         # Load the image
#         image = cv2.imread(image_path)
#         if image is None:
#             continue

#         # Convert to RGB and HSV color spaces
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#         # Define color ranges for ripe (red), unripe (green), and damaged (dark/black) apples
#         lower_red = np.array([0, 50, 50])      # Adjusted for ripe apples
#         upper_red = np.array([10, 255, 255])

#         lower_green = np.array([35, 50, 50])   # Adjusted for unripe apples
#         upper_green = np.array([85, 255, 255])

#         lower_black = np.array([0, 0, 0])      # Adjusted for damaged apples (dark/black)
#         upper_black = np.array([50, 50, 50])

#         # Create masks for ripe (red), unripe (green), and damaged (black/dark) apples
#         red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
#         green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
#         black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

#         # Calculate the percentage of red, green, and black pixels in the image
#         red_percentage = np.sum(red_mask) / (image.shape[0] * image.shape[1]) * 100
#         green_percentage = np.sum(green_mask) / (image.shape[0] * image.shape[1]) * 100
#         black_percentage = np.sum(black_mask) / (image.shape[0] * image.shape[1]) * 100

#         print(f"Red percentage: {red_percentage}%")
#         print(f"Green percentage: {green_percentage}%")
#         print(f"Black percentage: {black_percentage}%")

#         # Determine color status and ripeness status
#         if red_percentage > 50:
#             color_status = "Red"
#             ripeness_status = "Ripe"
#         elif green_percentage > 50:
#             color_status = "Green"
#             ripeness_status = "Unripe"
#         elif black_percentage > 20:
#             color_status = "Black"
#             ripeness_status = "Ripe"
#         else:
#             color_status = "Unknown"
#             ripeness_status = "Unknown"

#         # Determine overall status based on the color and ripeness
#         if color_status == "Red" and ripeness_status == "Ripe":
#             overall_status = "Good"
#         elif color_status == "Black" and ripeness_status == "Ripe":
#             overall_status = "Bad"
#         elif color_status == "Green" and ripeness_status == "Unripe":
#             overall_status = "Good"
#         else:
#             overall_status = "Bad"

#         # Prepare the result dictionary
#         result = {
#             'color_check': 'True',
#             'ripeness_check': 'True',
#             'uniformity_check': 'True',
#             'color_status': color_status,
#             'ripeness_status': ripeness_status,
#             'overall_status': overall_status
#         }

#         print(f"Result for {image_path}: {result}")
#         results.append(result)

#     return results

def analyze_images(image_paths, fruit_type=""):
    """Analyze a list of images and return their results for the specified fruit type."""
    results = []

    for image_path in image_paths:
        if fruit_type == "apple":
            result = analyze_apple(image_path)
        # elif fruit_type == "grape":
        #     result = analyze_grapes(image_path)
        # elif fruit_type == "orange":
        #     result = analyze_orange(image_path)
        else:
            result = {"status": "Unknown fruit type"}

        print(f"Result for {image_path}: {result}")
        results.append(result)

    return results

# Example usage for analyzing multiple images of all fruits
apple_image_paths = [
    r"C:\Users\user\Documents\Dataset\train\Apples\ripe\Apples ripe\image_13.jpg",  # Replace with actual image paths
    r"C:\Users\user\Documents\Dataset\train\Apples\unripe\Apples unripe\image_1.jpg"
]

# grape_image_paths = [
#     r"path_to_your_grape_image1.jpg",  # Replace with actual image paths
#     r"path_to_your_grape_image2.jpg"
# ]

# orange_image_paths = [
#     r"path_to_your_orange_image1.jpg",  # Replace with actual image paths
#     r"path_to_your_orange_image2.jpg"
# ]

# Analyze the images and print the results for apples
print("Apple Analysis Results:")
apple_results = analyze_images(apple_image_paths, fruit_type="apple")
for result in apple_results:
    print(result)

# # Analyze the images and print the results for grapes
# print("Grape Analysis Results:")
# grape_results = analyze_images(grape_image_paths, fruit_type="grape")
# for result in grape_results:
#     print(result)

# # Analyze the images and print the results for oranges
# print("Orange Analysis Results:")
# orange_results = analyze_images(orange_image_paths, fruit_type="orange")
# for result in orange_results:
#     print(result)


# # Example usage for analyzing multiple images
# image_paths = [
#     r"C:\Users\user\Documents\Dataset\train\Apples\worst condition\Apples worst condition\Image_12.jpg",  # Replace with actual image paths
#     r"C:\Users\user\Documents\Dataset\train\Apples\ripe\Apples ripe\Image_3.jpg",
#     r"C:\Users\user\Documents\Dataset\train\Apples\unripe\Apples unripe\Image_3.jpg"
# ]

# # Analyze the images and print the results
# analysis_results = analyze_images(image_paths)
# for result in analysis_results:
#     print(result)

# def analyze_images(image_paths):
#     """Analyze a list of images and return their results."""
#     results = []

#     for image_path in image_paths:
#         # Analyze each image
#         analysis_result = analyze_apple(image_path)

#         # Add the result to the list
#         results.append(analysis_result)

#         # Additional Feature Extraction for color and intensity
#         image = cv2.imread(image_path)
#         if image is None:
#             continue

#         # Convert to RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Extract mean colors (R, G, B)
#         mean_colors = cv2.mean(image_rgb)[:3]
#         mean_colors = [round(c, 2) for c in mean_colors]
#         print(f"Mean colors: {mean_colors}")

#         # Convert to grayscale for texture analysis
#         gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
#         mean_intensity = np.mean(gray_image)
#         print(f"Mean intensity: {mean_intensity}")

#         # Determine Color Status
#         color_status = 'Red' if mean_colors[0] > 150 else 'Green'
#         print(f"Color Status: {color_status}")

#         # Determine Ripeness based on color status
#         if color_status == 'Red':
#             ripeness_status = 'Ripe'
#         elif color_status == 'Green':
#             ripeness_status = 'Unripe'
#         else:
#             ripeness_status = 'Unknown'
#         print(f"Ripeness Status: {ripeness_status}")

#         # Load the pre-trained model and predict the overall status
#         try:
#             model = joblib.load(MODEL_PATH)
#             features = np.array([mean_colors[0], mean_colors[1], mean_colors[2], mean_intensity]).reshape(1, -1)
#             overall_status = model.predict(features)[0]  # Assuming model returns 'Good' or 'Bad'
#             print(f"Model-based Overall Status: {overall_status}")
#         except Exception as e:
#             print(f"Model loading or prediction error: {e}")

#     return results


# # Example usage for analyzing multiple images
# image_paths = [
#     r"C:\Users\user\Documents\Dataset\train\Apples\worst condition\Apples worst condition\Image_12.jpg",  # Replace with actual image paths
#     r"C:\Users\user\Documents\Dataset\train\Apples\ripe\Apples ripe\Image_3.jpg",
#     r"C:\Users\user\Documents\Dataset\train\Apples\unripe\Apples unripe\Image_3.jpg"
# ]

# # Analyze the images and print the results
# analysis_results = analyze_images(image_paths)
# for result in analysis_results:
#     print(result)

# from django.shortcuts import render
# from .forms import ImageUploadForm
# from django.core.files.storage import default_storage
# import cv2
# import numpy as np
# import joblib
# import os

# # Model save/load path
# MODEL_DIR = 'inspection/models'
# MODEL_PATH = os.path.join(MODEL_DIR, 'svm_fruit_quality_model.pkl')

# # Ensure model directory exists
# os.makedirs(MODEL_DIR, exist_ok=True)

# def upload_image(request):
#     if request.method == 'POST':
#         form = ImageUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             # Save the uploaded image temporarily
#             image_file = request.FILES['image']
#             image_path = default_storage.save('tmp/' + image_file.name, image_file)
#             image_path = default_storage.path(image_path)
            
#             # Analyze the image and get the result
#             result = analyze_image(image_path)

#             # Clean up the temporary file
#             default_storage.delete(image_path)

#             print(f"Result dictionary: {result}")  # Ensure this prints correctly

#             return render(request, 'inspection/result.html', {'result': result})
#     else:
#         form = ImageUploadForm()
    
#     # Render the upload form for a GET request, no result is passed yet
#     return render(request, 'inspection/index.html', {'form': form})

# def analyze_apple(image_path):
#     """Analyze the uploaded image and return the analysis results."""
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         return {"status": "Image not found"}

#     # Convert to RGB and HSV color spaces
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define color ranges for ripe (red) and unripe (green) apples
#     lower_red = np.array([0, 50, 50])
#     upper_red = np.array([10, 255, 255])
#     lower_green = np.array([35, 50, 50])
#     upper_green = np.array([85, 255, 255])

#     # Create masks for ripe (red) and unripe (green) apples
#     red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
#     green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

#     # Calculate the percentage of red and green pixels in the image
#     red_percentage = np.sum(red_mask) / (image.shape[0] * image.shape[1]) * 100
#     green_percentage = np.sum(green_mask) / (image.shape[0] * image.shape[1]) * 100

#     # Analyze ripeness based on color
#     if red_percentage > 50:
#         color_status = "Ripe"
#     elif green_percentage > 50:
#         color_status = "Unripe"
#     else:
#         color_status = "Unknown"

#     # Analyze if the apple is damaged based on dark spots
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, threshold = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
#     dark_spots_percentage = (1 - np.sum(threshold) / (threshold.shape[0] * threshold.shape[1])) * 100

#     # If dark spots cover a significant portion, classify the apple as bad
#     if dark_spots_percentage > 10:
#         overall_status = "Bad"
#     else:
#         overall_status = "Good"

#     result = {
        # 'color_check': 'True',
        # 'ripeness_check': 'True',
        # 'uniformity_check': 'True',
#         'color_status': color_status,
#         'ripeness_status': 'Ripe' if color_status == 'Ripe' else 'Unripe',
#         'overall_status': overall_status
#     }

#     # Return the result dictionary

#     return result


# def analyze_images(image_paths):
#     """Analyze a list of images and print the results."""
#     for img_path in image_paths:
#         analysis_result = analyze_apple(img_path)
#         print(f"Analysis result for {img_path}: {analysis_result}")

#         # Additional Feature Extraction for color and intensity
#         image = cv2.imread(image_path)

#     # Convert to RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Extract mean colors (R, G, B)
#         mean_colors = cv2.mean(image_rgb)[:3]
#         mean_colors = [round(c, 2) for c in mean_colors]
#         print(f"Mean colors: {mean_colors}")

#         # Convert to grayscale for texture analysis
#         gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
#         mean_intensity = np.mean(gray_image)
#         print(f"Mean intensity: {mean_intensity}")

#         # Determine Color Status
#         color_status = 'Red' if mean_colors[0] > 150 else 'Green'
#         print(f"Color Status: {color_status}")

#         # Determine Ripeness based on color status
#         if color_status == 'Red':
#             ripeness_status = 'Ripe'
#         elif color_status == 'Green':
#             ripeness_status = 'Unripe'
#         else:
#             ripeness_status = 'Unknown'
#         print(f"Ripeness Status: {ripeness_status}")

#         # Load the pre-trained model and predict the overall status
#         try:
#             model = joblib.load(MODEL_PATH)
#             features = np.array([mean_colors[0], mean_colors[1], mean_colors[2], mean_intensity]).reshape(1, -1)
#             overall_status = model.predict(features)[0]  # Assuming model returns 'Good' or 'Bad'
#             print(f"Model-based Overall Status: {overall_status}")
#         except Exception as e:
#             print(f"Model loading or prediction error: {e}")


# # List of image paths to analyze
# image_paths = [
#     "/mnt/data/Image_1.jpg",  # Replace with actual image paths
#     "/mnt/data/Image_2.jpg",
#     "/mnt/data/Image_3.jpg"
# ]

# # Analyze the images
# analysis_results = analyze_images(image_paths)
# for result in analysis_results:
#     print(result)