import cv2
import numpy as np

def detect_poses(input_path):
    # Load the pre-trained OpenPose model
    net = cv2.dnn.readNetFromTensorflow('pose/coco/pose_iter_440000.caffemodel', 'pose/coco/pose_deploy_linevec.prototxt')

    if input_path.endswith('.mp4'):
        # Video input
        cap = cv2.VideoCapture(input_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))

            output = net.forward()

            H, W = frame.shape[:2]
            points = []
            for i in range(18):
                probMap = output[0, i, :, :]
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                x = int((W * point[0]) / output.shape[3])
                y = int((H * point[1]) / output.shape[2])
                if prob > 0.1:
                    points.append((x, y))
                else:
                    points.append(None)

            for pair in [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]:
                if points[pair[0]] and points[pair[1]]:
                    cv2.line(frame, points[pair[0]], points[pair[1]], (0, 255, 0), 2)

            cv2.imshow('Pose Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Image input
        image = cv2.imread(input_path)

        frameWidth = image.shape[1]
        frameHeight = image.shape[0]
        net.setInput(cv2.dnn.blobFromImage(image, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))

        output = net.forward()

        H, W = image.shape[:2]
        points = []
        for i in range(18):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = int((W * point[0]) / output.shape[3])
            y = int((H * point[1]) / output.shape[2])
            if prob > 0.1:
                points.append((x, y))
            else:
                points.append(None)

        for pair in [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]:
            if points[pair[0]] and points[pair[1]]:
                cv2.line(image, points[pair[0]], points[pair[1]], (0, 255, 0), 2)

        cv2.imshow('Pose Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Usage example:
# For video: detect_poses('input.mp4')
# For image: detect_poses('input.jpg')
