import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the garment image with alpha channel
garment_path = "myntra_garment2-removebg-preview.png"
garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)

# Check if the image is loaded properly
if garment is None or garment.size == 0:
    raise ValueError("Error: Failed to load garment image or empty dimensions")
else:
    print(f"Garment loaded successfully with shape: {garment.shape}")

# Global flag to stop the thread
stop_thread = False

def virtual_try_on():
    global stop_thread
    cap = cv2.VideoCapture(0)
    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            # Get relevant landmarks (adjust as needed)
            left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            neck = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]  # Example, adjust as needed

            # Calculate coordinates
            left_x = int(left_shoulder.x * frame.shape[1])
            left_y = int(left_shoulder.y * frame.shape[0])
            right_x = int(right_shoulder.x * frame.shape[1])
            right_y = int(right_shoulder.y * frame.shape[0])
            neck_x = int(neck.x * frame.shape[1])
            neck_y = int(neck.y * frame.shape[0])

            # Calculate garment height to cover upper body
            garment_height = abs(neck_y - min(left_y, right_y)) * 4  # Adjust as needed

            # Calculate garment width based on shoulder width
            garment_width = int(1.5 * abs(right_x - left_x))

            # Debug statements
            print(f"Left shoulder: ({left_x}, {left_y})")
            print(f"Right shoulder: ({right_x}, {right_y})")
            print(f"Garment width: {garment_width}, Garment height: {garment_height}")

            if garment_width > 0 and garment_height > 0:
                # Resize garment image
                resized_garment = cv2.resize(garment, (garment_width, garment_height), interpolation=cv2.INTER_AREA)

                # Calculate center point of upper body
                # center_x = (left_x + right_x) // 2
                # center_y = (left_y + right_y) // 2

                # Calculate positioning offsets to center the garment
                # x_offset = center_x - garment_width // 2
                # y_offset = center_y - garment_height // 2 + 80  # Adjust +50 to move slightly below center
                x_offset = neck_x - garment_width // 2
                y_offset = neck_y
                # Adjust offsets to fit within frame boundaries
                if x_offset < 0:
                    x_offset = 0
                if y_offset < 0:
                    y_offset = 0
                if x_offset + garment_width > frame.shape[1]:
                    x_offset = frame.shape[1] - garment_width
                if y_offset + garment_height > frame.shape[0]:
                    y_offset = frame.shape[0] - garment_height

                # Overlay the garment on the frame
                for i in range(garment_height):
                    for j in range(garment_width):
                        alpha = resized_garment[i, j, 3] / 255.0
                        if 0 <= y_offset + i < frame.shape[0] and 0 <= x_offset + j < frame.shape[1]:
                            frame[y_offset + i, x_offset + j] = alpha * resized_garment[i, j, :3] + (1 - alpha) * frame[y_offset + i, x_offset + j]

        cv2.imshow('Virtual Try-On', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_thread = True
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    virtual_try_on()
