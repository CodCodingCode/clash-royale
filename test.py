import cv2
from inference.core.interfaces.camera.entities import VideoFrame
from typing import Union, List, Optional
from inference import InferencePipeline
import time
from openai import OpenAI
api_key = "Zw9s4qJmfSsVpb4IerO9"

my_cards = []
enemies_cards = []

client = OpenAI(api_key = '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')

def on_prediction(
    predictions: Union[dict, List[Optional[dict]]],
    video_frame: Union[VideoFrame, List[Optional[VideoFrame]]]
) -> None:
    if not isinstance(predictions, list):
        predictions = [predictions]
        video_frame = [video_frame]

    for prediction, frame in zip(predictions, video_frame):
        if prediction is None:
            continue

        # Extract image from the VideoFrame object
        image = frame.image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for obj in prediction['predictions']:
            # Get bounding box coordinates
            x = obj['x']
            y = obj['y']
            width = obj['width']
            height = obj['height']
            label = obj['class']
            confidence = obj['confidence']
            if x in range(0, 1400) and y in range(1600, 1900):
                if label not in my_cards:
                    print(f"Player placed {label}")
                    my_cards.append(label)
            elif x in range(0, 1400) and y in range(0, 300):
                if label not in enemies_cards:
                    print(f"Opponent placed {label}")
                    enemies_cards.append(label)
          
            '''if len(enemies_cards) == 4 and prev_len != len(my_cards):

                completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a clash royale expert, that knows the best decks to play."},
                    {"role": "user", "content": I am playing clash royale right now and need to know the deck of my opponent. 
                     What deck includes a minion horde and ice spirit. Please answer in this format:
                    Possible Cards: [YOUR ANSWER HERE]
                    Win Rate with Deck: [YOUR ANSWER HERE]
                    Strategies to counter deck: [YOUR ANSWER HERE]}
                ]
                )
                print(completion.choices[0].message)'''
            # prev_len = len(my_cards)
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
            
            cv2.rectangle(image, ([0, 1650]),([1400, 1900]), (255, 255, 255), 3)
            cv2.rectangle(image, ([0, 0]),([1400, 300]), (255, 255, 255), 3)

            # Draw label text
            cv2.putText(image, label_text, (start_point[0], start_point[1] - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Display the resulting frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        cv2.imshow('Frame', image)

        # Press 'q' to exit the video display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

pipeline = InferencePipeline.init(
    model_id="clash-royale-detection-cysig/6",
    max_fps=60,
    confidence=0.5,
    video_reference='/Users/owner/Downloads/clash-royale/test.mp4',
    on_prediction=on_prediction,
    api_key=api_key
)

print("Starting pipeline...") 
pipeline.start()

pipeline.join()
cv2.destroyAllWindows()
