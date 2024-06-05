import cv2
from inference.core.interfaces.camera.entities import VideoFrame
from typing import Union, List, Optional
from inference import InferencePipeline
import time
from openai import OpenAI
import pygame


api_key = "ROBOFLOW_API_KEY"
client = OpenAI(api_key = 'OPENAI_API_KEY')

my_cardsc = []
enemies_cardsc = []
my_cardsa = []
enemies_cardsa = []
my_hand = []
enemy_hand = []
detected_labels = []
prev_message = ""

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)

prev_detections = []
detections = []

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

correlation = {
    "C-hog-rider": "A-hog-rider",
    "C-ice-spirit": "A-ice-spirit",
    "C-ice-golem": "A-ice-golem",
    "C-minion-horde": "A-minion-horde",
    "C-canon": "A-canon",
    "C-fireball": "A-fireball",
    "C-log": "A-log",
    "C-balloon": "A-balloon",
    "C-freeze": "A-freeze",
    "C-musketeer": "A-musketeer",
    "C-royal-ghost": "A-royal-ghost",
    "C-skeletons": "A-skeletons",
    "C-valkyrie": "A-valkyrie",
    "C-wall-breakers": "A-wall-breakers",
    "C-wizard": "A-wizard",
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

screen_pygame = (600, 800)

def on_prediction(
    predictions: Union[dict, List[Optional[dict]]],
    video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
) -> None:
    if not isinstance(predictions, list):
        predictions = [predictions]
        video_frame = [video_frame]
    
    global prev_detections, detections, my_hand, enemy_hand, prev_message

    for prediction, frame in zip(predictions, video_frame):
        if prediction is None:
            continue

        # Extract image from the VideoFrame object
        image = frame.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for obj in prediction['predictions']:
            # Get bounding box coordinates
            x = obj['x']
            y = obj['y']
            width = obj['width']
            height = obj['height']
            label = obj['class']
            confidence = obj['confidence']
            if x in range(0, 1400) and y in range(1600, 1900):
                if label not in my_cardsc:
                    print(f"Player has {label} in deck")
                    my_cardsc.append(label)
                    my_cardsa.append(correlation[label])
            elif x in range(0, 1400) and y in range(0, 300):
                if label not in enemies_cardsc:
                    print(f"Opponent has {label} in deck")
                    enemies_cardsc.append(label)
                    enemies_cardsa.append(correlation[label])
            
            if y in range(1000) and label not in prev_detections and label in enemies_cardsa and label != prev_message:
                print(f"Opponent placed {label}")
                prev_message = label
            if y not in range(1000) and label not in prev_detections and label in my_cardsa and label != prev_message:
                print(f"Player placed {label}")
                prev_message = label

            if len(enemies_cardsc) == 4:
                print("Cool")

            # Calculate top-left and bottom-right coordinates of the bounding box
            start_point = (int(x - width / 2), int(y - height / 2))
            end_point = (int(x + width / 2), int(y + height / 2))

            # Draw the bounding box
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

            # Prepare label text
            label_text = f"{label}: {confidence:.2f}"

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw background rectangle for the label
            cv2.rectangle(image, (start_point[0], start_point[1] - text_height - baseline),
                          (start_point[0] + text_width, start_point[1]), (0, 255, 0), cv2.FILLED)

            # Card zones
            cv2.rectangle(image, card_zones["Player-Card1"], card_zones['Player-Card2'], WHITE, 3)
            cv2.rectangle(image, card_zones['Opponent-Card1'], card_zones["Opponent-Card2"], WHITE, 3)

            detections.append(label)
            
            # Princess tower danger zones
            if label not in detected_labels:
                if x in range(405, 536) and y in range(1083, 1415) and label != "T-archer-tower" and label not in my_cardsa:
                    print(f"Player princess tower 1 is attacking {label}")
                    detected_labels.append(label)
                if x in range(900, 1039) and y in range(1115, 1415) and label != "T-archer-tower" and label not in my_cardsa:
                    print(f"Player princess tower 2 is attacking {label}")
                    detected_labels.append(label)
                if x in range(405, 545) and y in range(610, 942) and label != "T-archer-tower" and label not in enemies_cardsa:
                    print(f"Opponent princess tower 1 is attacking {label}")
                    detected_labels.append(label)
                if x in range(912, 1042) and y in range(608, 942) and label != "T-archer-tower" and label not in enemies_cardsa:
                    print(f"Opponent princess tower 2 is attacking {label}")
                    detected_labels.append(label)

            cv2.rectangle(image, card_zones["Player-princess1-tower1"], card_zones['Player-princess1-tower2'], BLUE, 3)
            cv2.rectangle(image, card_zones["Player-princess2-tower1"], card_zones['Player-princess2-tower2'], BLUE, 3)
            cv2.rectangle(image, card_zones["Opponent-princess1-tower1"], card_zones['Opponent-princess1-tower2'], BLUE, 3)
            cv2.rectangle(image, card_zones["Opponent-princess2-tower1"], card_zones['Opponent-princess2-tower2'], BLUE, 3)

            cv2.line(image, (0, 1000), (1400, 1000), RED, 3)
            # Draw label text
            cv2.putText(image, label_text, (start_point[0], start_point[1] - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        prev_detections = detections.copy()
        detections = []
        

        # Display the resulting frame
        cv2.imshow('Frame', image)

        # Press 'q' to exit the video display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

pipeline = InferencePipeline.init(
    model_id="clash-royale-detection-cysig/7",
    max_fps=60,
    confidence=0.5,
    video_reference='/Users/owner/Downloads/clash-royale/test1.mp4',
    on_prediction=on_prediction,
    api_key=api_key
)

start_time = time.time()
print("Starting pipeline...")
pipeline.start()

pipeline.join()
cv2.destroyAllWindows()
