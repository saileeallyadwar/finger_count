import cv2
import numpy as np
import math

# Global variables
background = None
accumulated_weight = 0.5
roi_top, roi_bottom = 100, 400  # Larger ROI for better hand detection
roi_left, roi_right = 200, 500
calibration_frames = 120  # More frames for better background calibration

def calc_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment(frame, threshold=30):  # Increased default threshold
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Enhanced morphological operations
    kernel = np.ones((5,5), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Filter contours by area and convexity
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:  # Minimum hand area threshold
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area)/hull_area
            if solidity < 0.9:  # Hand contours are typically not perfectly convex
                valid_contours.append(cnt)
    
    if not valid_contours:
        return None
    
    hand_segment = max(valid_contours, key=cv2.contourArea)
    return (thresholded, hand_segment)

def count_fingers(thresholded, hand_segment):
    # Convex hull and defects approach
    hull = cv2.convexHull(hand_segment, returnPoints=False)
    defects = cv2.convexityDefects(hand_segment, hull)
    
    if defects is None:
        return 0
    
    finger_count = 0
    angle_threshold = 80  # Degrees
    depth_threshold = 10000  # Minimum defect depth
    
    # Get the center of the hand
    M = cv2.moments(hand_segment)
    if M["m00"] == 0:
        return 0
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = (cx, cy)
    
    # Analyze each convexity defect
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i,0]
        start = tuple(hand_segment[s][0])
        end = tuple(hand_segment[e][0])
        far = tuple(hand_segment[f][0])
        
        # Calculate distances
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        
        # Calculate angle
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 180/math.pi
        
        # Filter defects based on angle and depth
        if angle <= angle_threshold and d > depth_threshold:
            # Additional check to ensure defect is between fingers
            if far[1] < center[1]:  # Only count defects above palm center
                finger_count += 1
    
    # The number of fingers is defects + 1 (since each space between fingers creates a defect)
    total_fingers = min(finger_count + 1, 5)  # Cap at 5 fingers
    
    # Special case for closed fist (0 fingers)
    if total_fingers == 1:
        # Check if it's really a fist by comparing contour area to hull area
        hull_area = cv2.contourArea(cv2.convexHull(hand_segment))
        contour_area = cv2.contourArea(hand_segment)
        if hull_area > 0 and (contour_area / hull_area) > 0.85:  # Fist is more convex
            total_fingers = 0
    
    return total_fingers

def main():
    global background
    
    cam = cv2.VideoCapture(0)
    num_frames = 0
    
    # Create windows
    cv2.namedWindow("Finger Counter", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)  # Increased blur for better background subtraction
        
        if num_frames < calibration_frames:
            calc_accum_avg(gray, accumulated_weight)
            progress = int((num_frames/calibration_frames)*100)
            cv2.putText(frame_copy, f"Calibrating... {progress}%", (50, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame_copy, "Keep hand out of frame", (50, 100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            hand = segment(gray)
            
            if hand is not None:
                thresholded, hand_segment = hand
                
                # Draw contours and convex hull
                cv2.drawContours(frame_copy, [hand_segment + (roi_left, roi_top)], -1, (255, 0, 0), 2)
                hull = cv2.convexHull(hand_segment)
                cv2.drawContours(frame_copy, [hull + (roi_left, roi_top)], -1, (0, 255, 0), 2)
                
                fingers = count_fingers(thresholded, hand_segment)
                
                # Display finger count with confidence indicator
                cv2.putText(frame_copy, f"Fingers: {fingers}", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # Show thresholded image
                cv2.imshow("Thresholded", thresholded)
        
        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 3)
        num_frames += 1
        cv2.imshow("Finger Counter", frame_copy)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()