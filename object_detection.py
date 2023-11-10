import cv2

def detect_objects():
    # Load the pre-trained YOLO model and classes
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    classes = []
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Set the input image size and scale factor
    input_size = (416, 416)
    scale = 0.00392

    # Open the live video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the video capture
        ret, frame = cap.read()

        # Create a blob from the input frame
        blob = cv2.dnn.blobFromImage(frame, scale, input_size, (0, 0, 0), True, crop=False)

        # Set the blob as the input to the YOLO network
        net.setInput(blob)

        # Perform object detection
        detections = net.forward()

        for detection in detections:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > 0.5:
                # Get the object class label
                label = classes[class_id]

                # Display the object label on the frame
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    detect_objects()

if __name__ == '__main__':
    main()
