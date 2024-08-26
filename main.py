import threading

import cv2
import pose_estimation
import virtual_try_on

if __name__ == "__main__":
    try:
        # Create the threads
        pose_thread = threading.Thread(target=pose_estimation.pose_estimation)
        try_on_thread = threading.Thread(target=virtual_try_on.virtual_try_on)

        # Start the threads
        pose_thread.start()
        try_on_thread.start()

        # Wait for the threads to complete
        pose_thread.join()
        try_on_thread.join()
    except Exception as e:
        print(f"Error in main.py: {e}")
    finally:
        # Ensure all windows are closed
        cv2.destroyAllWindows()
