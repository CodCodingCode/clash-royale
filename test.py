import cv2
from inference.core.interfaces.camera.entities import VideoFrame
from typing import Union, List, Optional
from inference import InferencePipeline
import time

api_key = "Zw9s4qJmfSsVpb4IerO9"

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
    model_id="clash-royale-detection-cysig/4",
    max_fps=60,
    confidence=0.3,
    video_reference='/Users/owner/Downloads/clash-royale/test.mp4',
    on_prediction=on_prediction,
    api_key=api_key
)

print("Starting pipeline...") 
pipeline.start()

pipeline.join()
cv2.destroyAllWindows()
