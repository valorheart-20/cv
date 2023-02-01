import cv2
import mediapipe as mp
import numpy as np

def visualize(input, faces, fps, org, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            cv2.putText(input, 'Face {}, box: ({:.0f}, {:.0f} {:.0f}, {:.0f}), \
                score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]), 
                (1, org + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, org), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def bounding_from_mp_result(shape, mp_detection_result):
    results = np.zeros([len(mp_detection_result), 15], dtype = np.int32)
    width = shape[1]
    height = shape[0]
    for i, bounding in enumerate(mp_detection_result):
        old_bounding = bounding.location_data.relative_bounding_box
        results[i][0] = old_bounding.xmin * width
        results[i][1] = old_bounding.ymin * height
        results[i][2] = old_bounding.width * width
        results[i][3] = old_bounding.height * height
    return results

def run():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    detector1 = cv2.FaceDetectorYN.create(
        'detection.onnx',
        "",
        (320, 320), 0.9,
        0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create('recognition.onnx', '')
    tm = cv2.TickMeter()


    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            tm.start()
            detector1.setInputSize((image.shape[1], image.shape[0]))
            result1 = detector1.detect(image)
            visualize(image, result1, tm.getFPS(), 18)
            tm.stop()


            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            tm.start()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            tm.stop()
            mp_detections = results.detections
            if mp_detections:
                bounding = bounding_from_mp_result(image.shape, mp_detections)
                for detection in mp_detections:
                    mp_drawing.draw_detection(image, detection)
                    cv2.putText(image, 'FPS: {:.2f}, Box: ({:.4f}, {:.4f} {:.4f}, {:.4f}), \
                        score: {:.2f}'.format(tm.getFPS(), 
                        bounding[0][0], bounding[0][1], bounding[0][2], bounding[0][3],0), 
                        (1, 36 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #         aligned = recognizer.alignCrop(image, detection.location_data.relative_bounding_box)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Detection', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


if __name__ == '__main__':
    run()
