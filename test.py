import numpy as np

# ----- calibration -----
K = np.array([[701.12,   0.0, 610.83],
              [  0.0, 701.12, 380.3405],
              [  0.0,   0.0,   1.0 ]])
K_inv = np.linalg.inv(K)

R_wc = np.array([[ 0,  0,  1],
                 [-1,  0,  0],
                 [ 0, -1,  0]])       # see derivation above

C_w = np.array([0.0, 0.0, 0.7366])    # optical centre, metres

# ----- pixel you clicked (OpenCV convention: col=x, row=y) -----
col, row = 300, 606          # <-- adjust to your actual click

# ----- 1. pixel --> camera ray -----
ray_cam = K_inv @ np.array([col, row, 1.0])

# ----- 2. camera ray --> world -----
ray_world = R_wc @ ray_cam

# ----- 3. intersect with ground Z=0 -----
if abs(ray_world[2]) < 1e-9:
    raise RuntimeError("Ray parallel to ground plane")
lam = -C_w[2] / ray_world[2]
P_w = C_w + lam * ray_world
P_w[2] = 0.0                 # enforce planarity

print(P_w)                   # -> [ 2.288  1.015  0.   ]
