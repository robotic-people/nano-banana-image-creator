import datetime
import os
from dotenv import load_dotenv, find_dotenv # type: ignore
from google import genai
from google.genai import types # type: ignore
from PIL import Image # type: ignore
import json

load_dotenv(find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load config file
with open('config.json', 'r') as f:
    config = json.load(f)

model_id = config["model_id"]
prompt = config["prompt"]
aspect_ratio = config["aspect_ratio"]
resolution = config["resolution"]
num_images = config["num_images"]
input_dir = config["input_dir"]
output_dir = config["output_dir"]
output_name = config["output_name"]

client = genai.Client()

# Define safety settings
safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
    )
]

generate_content_config = types.GenerateContentConfig(
    response_modalities=['Text', 'Image'],
    image_config=types.ImageConfig(
        aspect_ratio=aspect_ratio,
        image_size=resolution
    ),
    safety_settings=safety_settings
)

contents = [prompt]

# Load all of the images in the input directory if they exist and add them to the contents
if os.path.exists(input_dir):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_dir, filename)
            contents.append(Image.open(image_path))

def generate_image(image_path, index):
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=generate_content_config,
        )

        if response.candidates is None:
            print("BLOCKED. prompt_feedback:", response.prompt_feedback)
            return

        # Save the image
        for part in response.parts:
            if image:= part.as_image():
                image.save(image_path)

            print(f"Output number {index+1} saved to {image_path}")

    except Exception as e:
        print(f"An error occurred during image number {index + 1}: {e}")

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get time and date for unique directory for each run in the output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_new = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir_new)

    for i in range(num_images):
        image_path = os.path.join(output_dir_new, f"{output_name}_{i+1}.png")
        generate_image(image_path, i)

    # Check if the output directory is empty
    if not os.listdir(output_dir_new):
        os.rmdir(output_dir_new)
        print(f"No images were generated. Removed empty directory: {output_dir_new}")