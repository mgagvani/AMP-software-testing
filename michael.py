import cv2
import numpy as np
import glob
import os

def pixel_to_ground(u, v, fx, fy, cx, cy, H):
    # Ray direction in camera coordinates (normalized such that Z_c = 1)
    x_dir = (u - cx) / fx
    y_dir = (v - cy) / fy
    z_dir = 1.0
    # Avoid division by zero (e.g., points exactly at horizon)
    if y_dir == 0:
        return None  # ray is parallel to ground (no intersection within finite distance)
    # Solve for intersection scale (lambda)
    lam = H / y_dir  # negative because Y_dir is negative for ground points (camera Y up)
    # Compute ground intersection in camera coords
    X_ground = lam * x_dir
    Z_ground = lam * z_dir
    return X_ground, Z_ground

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        fx, fy, cx, cy, H = param
        pt = pixel_to_ground(x, y, fx, fy, cx, cy, H)
        print(f"Clicked pixel = ({x:3d}, {y:3d}) → ground = {pt}")   
#(657,563)

def main():
    fx, fy = 701.12, 701.12
    cx, cy = 610.83, 380.3405
    H = 0.7633

    params = {'fx': 701.12, 'fy': 701.12, 'cx': 610.83, 'cy': 380.3405, 'k1': -0.175609, 'k2': 0.0273627, 'p1': 0.000324635, 'p2': 0.00135292, 'k3': 0.0}
    k = [params['k1'], params['k2'], params['p1'], params['p2']]

    # x =646
    # y =563
    # print(pixel_to_ground(x, y, fx, fy, cx, cy, H))
    # return
    
    # 1) Open camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 2) Create window & bind callback (pass intrinsics+H in 'param')
    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", mouse_callback, param=(fx, fy, cx, cy, H))

    # 3) Grab-and-show loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: frame grab failed.")
            break

        # Show left half only, if that’s what you want
        left = frame[:, : frame.shape[1] // 2]
        # print(left.shape)
        # resize to 1280x720
        left = cv2.resize(left, (1280, 720))
        left = cv2.undistort(left, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array(k))
        cv2.imshow("Camera", left)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()