import cv2
import mediapipe as mp
import helper

hp = helper.Helper()
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(4)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened() and cap2.isOpened():
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        img.flags.writeable = False
        img2.flags.writeable = False

        result = pose.process(img)
        result2 = pose.process(img2)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        img.flags.writeable = True
        img2.flags.writeable = True

        try:
            landmarks = result.pose_landmarks.landmark

            # Get coordinates for angle_wrist
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]

            # Get coordinates for angle_elbow
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            # Calculate angle
            angle_elbow = hp.calculate_angle(left_shoulder, left_elbow, left_wrist)
            angle_wrist = hp.calculate_angle(left_index, left_wrist, left_pinky)

            # Wave patient attention
            hp.patient_attention(angle_elbow, angle_wrist)

            # Visualize to camera
            cv2.rectangle(img, (0, 0), (300, 110), (245, 117, 16), -1)

            cv2.putText(img, "ELBOW", (15, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(int(angle_elbow)), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(img, "WRIST", (140, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(int(angle_wrist)), (143, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # ========================= CAMERA 2 =====================

            landmarks2 = result2.pose_landmarks.landmark

            # Get coordinates for angle_wrist
            left_index2 = [landmarks2[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                           landmarks2[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            left_wrist2 = [landmarks2[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks2[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_pinky2 = [landmarks2[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
                           landmarks2[mp_pose.PoseLandmark.LEFT_PINKY.value].y]

            # Get coordinates for angle_elbow
            left_shoulder2 = [landmarks2[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks2[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow2 = [landmarks2[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks2[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            # Calculate angle
            angle_elbow2 = hp.calculate_angle(left_shoulder2, left_elbow2, left_wrist2)
            angle_wrist2 = hp.calculate_angle(left_index2, left_wrist2, left_pinky2)

            # Wave patient attention
            hp.patient_attention(angle_elbow2, angle_wrist2)

            # Visualize to camera
            cv2.rectangle(img2, (0, 0), (300, 110), (245, 117, 16), -1)

            cv2.putText(img2, "ELBOW", (15, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img2, str(int(angle_elbow)), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(img2, "WRIST", (140, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img2, str(int(angle_wrist)), (143, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        except:
            pass

        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(img2, result2.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe Feed', img)
        cv2.imshow("Second Camera", img2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cap2.release()
cv2.destroyAllWindows()
