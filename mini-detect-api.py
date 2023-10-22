from ultralytics import YOLO
import torch
import cv2
import numpy as np
import math
import sys
import json
from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.responses import Response, JSONResponse

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def is_rectish(contour):
    contour_len = len(contour)
    max_angle_delta_deg = 15
    
    # Needs to be 4-sided
    if contour_len != 4:
        return False
    
    # Future better way: get "top-left" and go clockwise, then ensure that
    # top are obtuse, bottom are acute due to perspective
    
    p1 = contour[0]
    p2 = contour[1]
    p3 = contour[2]
    p4 = contour[3]
    
    angles = (
        math.degrees(angle_between(p2 - p1, p4 - p1)),
        math.degrees(angle_between(p1 - p2, p3 - p2)),
        math.degrees(angle_between(p4 - p3, p2 - p3)),
        math.degrees(angle_between(p3 - p4, p1 - p4)),
    )
    print(f"Angles: {angles}")

    if abs(angles[0] - 115) > max_angle_delta_deg:
        return False
    
    if abs(angles[3] - 115) > max_angle_delta_deg:
        return False
    
    if abs(angles[1] - 65) > max_angle_delta_deg:
        return False
    
    if abs(angles[2] - 65) > max_angle_delta_deg:
        return False
        
    return True

def reorder_upper_left(points):
    closest_dist = sys.float_info.max
    closest_idx = -1
    for idx, point in enumerate(points):
        dist = math.sqrt(math.pow(point[0],2) + math.pow(point[1],2))
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = idx
    
    for i in range(0, closest_idx):
        point = points[0]
        points = np.delete(points, 0, 0)
        points = np.append(points, [point], axis=0)
        
    return points
        

def detect_field(img, min_area_ratio, max_area_ratio):
    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    area = img.shape[0] * img.shape[1]
    center_point = (round(img.shape[0] / 2), round(img.shape[1] / 2))
    
    _, thresh = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)

    # using a findContours() function
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x) / area, reverse=True)
    for contour in contours_sorted:       
        contour_area = cv2.contourArea(contour)
        if contour_area / area > min_area_ratio and contour_area / area < max_area_ratio and cv2.pointPolygonTest(contour, center_point, False) == 1.0:
            # cv2.approxPloyDP() function to approximate the shape
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            approx = cv2.convexHull(approx)

            while len(approx) > 4:
                epsilon *= 1.1
                approx = cv2.approxPolyDP(contour, epsilon, True)
            
            approx = approx[:,0]
            approx = reorder_upper_left(approx)
            
            if not is_rectish(approx):
                continue

            M = cv2.moments(approx)
            if M['m00'] == 0.0:
                continue
            
            return approx

    return np.empty(0)

def get_pct_along_field(field, center):
    # Assume that field is drawn from upper left, counterclockwise, with 4 points
    return False

torch.cuda.set_device(0)
MODEL_NAME = "best-reduced-imagery.pt"
MODEL_PATH = f"G:/My Drive/ml/minitracking"
#SOURCE_VIDEO_PATH = f"{HOME}/drive/MyDrive/ml/minitracking/WIN_20230923_13_17_21_Pro.mp4"

model = YOLO(f"{MODEL_PATH}/{MODEL_NAME}")
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [0]

print("Starting video capture...")
capture = cv2.VideoCapture(0)
print("Video capture online...")

min_area_ratio = 0.2
max_area_ratio = 0.8
p_dim = 500

app = FastAPI()

@app.get("/image", responses={200:{"content":{"image/png":{}}}},response_class=Response)
def get_camera_annotated():
    has_frame, frame = capture.read()

    if not has_frame:
        return { "error": "No video detected" }

    field = detect_field(frame, min_area_ratio, max_area_ratio)
    if field.any():
        cv2.drawContours(frame, [field], -1, (0,255,255), 3)
        results = model.track(frame, persist=True)
        frame = results[0].plot()
        bytes = cv2.imencode('.png', frame)[1].tobytes()
        return Response(content=bytes, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Item not found")
    
@app.get("/perspective", responses={200:{"content":{"image/png":{}}}},response_class=Response)
def get_perspected_annotated():
    has_frame, frame = capture.read()

    if not has_frame:
        return { "error": "No video detected" }

    field = detect_field(frame, min_area_ratio, max_area_ratio)
    if field.any():
        results = model.track(frame, persist=True)
        frame = results[0].plot()
        dst_pts = np.array([[0,0],[0,500],[500,500],[500,0]])
        M, _ = cv2.findHomography(field, dst_pts)
        frame_mod = cv2.warpPerspective(frame, M, (500,500))
        bytes = cv2.imencode('.png', frame_mod)[1].tobytes()
        return Response(content=bytes, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Item not found")

@app.get("/coords")
def get_positions():
    has_frame, frame = capture.read()

    if not has_frame:
        return { "error": "No video detected" }
    
    positions = []

    field = detect_field(frame, min_area_ratio, max_area_ratio)
    if field.any():
        results = model.track(frame, persist=True)
        dst_pts = np.array([[0,0],[0,p_dim],[p_dim,p_dim],[p_dim,0]])
        M, _ = cv2.findHomography(field, dst_pts)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for xyxy, track_id in zip(boxes, track_ids):
            xyxy_i = xyxy.reshape(2,2)
            xyxy_m = cv2.perspectiveTransform(np.array([xyxy_i]), M)[0,:,:].flatten()
            xyxy_pct = (xyxy_m/p_dim).tolist()
            b_c = ((xyxy_pct[0] + xyxy_pct[2])/2, xyxy_pct[3])
            positions.append({
                "track": track_id,
                "b_c": b_c,
                "xyxy_m": xyxy_m.astype(np.int32).tolist(),
                "xyxy_pct": xyxy_pct,
                "xyxy" : xyxy.astype(np.int32).tolist()
            })

    return JSONResponse(content=positions)

if __name__ == "__main__":
    uvicorn.run("mini-detect-api:app", host="0.0.0.0", port=5000, log_level="info")