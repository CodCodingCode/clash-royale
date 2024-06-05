import cv2
from inference.core.interfaces.camera.entities import VideoFrame
from typing import Union, List, Optional
from inference import InferencePipeline
import pygame
import time

api_key = "ROBOFLOW_API_KEY"

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

card_zones =  {
    "Player-Card1": ([0, 1650]), 
    "Player-Card2": ([1400, 1900]), 
    "Opponent-Card1": ([0, 0]), 
    "Opponent-Card2": ([1400, 300]),
    "Player-princess1-tower1": ([405, 1083]),
    "Player-princess1-tower2": ([536, 1415]),
    "Player-princess2-tower1": ([900, 1115]),
    "Player-princess2-tower2": ([1039, 1415]),
    "Opponent-princess1-tower1": ([405, 610]),
    "Opponent-princess1-tower2": ([545, 942]),
    "Opponent-princess2-tower1": ([912, 608]),
    "Opponent-princess2-tower2": ([1042, 942]),
                }

# Initialize Pygame
pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (173, 216, 230)

size = (600, 800)
screen = pygame.display.set_mode(size)

# Buffer to store predictions for drawing
predictions_buffer = []

def on_prediction(
    predictions: Union[dict, List[Optional[dict]]],
    video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
) -> None:
    global predictions_buffer

    if not isinstance(predictions, list):
        predictions = [predictions]
        video_frame = [video_frame]

    for prediction, frame in zip(predictions, video_frame):
        if prediction is None:
            continue

        for obj in prediction['predictions']:
            # Get bounding box coordinates
            x = obj['x']
            y = obj['y']
            width = obj['width']
            height = obj['height']
            label = obj['class']
            predictions_buffer.append((x, y, width, height, label))

pipeline = InferencePipeline.init(
    model_id="clash-royale-detection-cysig/7",
    max_fps=60,
    confidence=0.6,
    video_reference='/Users/owner/Downloads/clash-royale/test.mp4',
    on_prediction=on_prediction,
    api_key=api_key
)

start_time = time.time()
print("Starting pipeline...")
pipeline.start()

# Event loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(WHITE)

    # Draw all predictions
    for (x, y, width, height, label) in predictions_buffer:
        pygame.draw.rect(screen, GREEN, (x, y, width, height), 2)

    # Update the display
    pygame.display.flip()

    # Clear the buffer after drawing
    predictions_buffer.clear()

pipeline.join()
pygame.quit()
cv2.destroyAllWindows()
