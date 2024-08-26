import cv2
import mediapipe as mp

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)

    def draw_landmarks(self, frame, pose_landmarks):
        if pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def get_landmarks(self, result):
        if result.pose_landmarks:
            return result.pose_landmarks.landmark
        return None

    def get_coordinates(self, frame, landmark):
        h, w, _ = frame.shape
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        return x, y

# Test the PoseEstimator class
if __name__ == "__main__":
    pose_estimator = PoseEstimator()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = pose_estimator.process_frame(frame)
        frame = pose_estimator.draw_landmarks(frame, result.pose_landmarks)
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
