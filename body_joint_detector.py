import cv2
import mediapipe as mp

def detect_joints_from_video(video_path, body_or_face='body'):
    """
    Reads a video, detects joints on the body or face using MediaPipe,
    and returns the video frames with landmarks overlaid.

    Args:
        video_path (str): Path to the video file.
        body_or_face (str, optional):  'body' for body pose estimation, 
                                        'face' for face mesh. Defaults to 'body'.

    Returns:
        list: A list of frames (NumPy arrays) with landmarks overlaid.
              Returns None if the video cannot be opened.
    """

    mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
    
    if body_or_face == 'body':
        mp_pose = mp.solutions.pose  # Pose estimation model
    elif body_or_face == 'face':
        mp_face_mesh = mp.solutions.face_mesh # Face mesh model
    else:
        raise ValueError("body_or_face must be 'body' or 'face'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None

    frames_with_landmarks = []

    with (mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) if body_or_face == 'body' else mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)) as model:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # To improve performance
        
            # Make detection
            results = model.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            if body_or_face == 'body':
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                         )
            elif body_or_face == 'face':                 
                 mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                         mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
                                         mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1)
                                         )

            frames_with_landmarks.append(image)

    cap.release()
    return frames_with_landmarks

if __name__ == '__main__':
    # Example usage:
    video_path = '/home/nishant/project/uco/clips_mp4/8/16/cam0.mp4'  # Replace with your video file path
    
    # Body pose estimation
    body_frames = detect_joints_from_video(video_path, body_or_face='body')
    
    # Face mesh estimation
    # face_frames = detect_joints_from_video(video_path, body_or_face='face')

    if body_frames:
        # Display the first frame with body landmarks (for example)
        cv2.imshow('Body Pose Estimation', body_frames[0])
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()

    #if face_frames:
    #   Display the first frame with face landmarks (for example)
    #   cv2.imshow('Face Mesh Estimation', face_frames[0])
    #   cv2.waitKey(0)
    #   cv2.destroyAllWindows()
