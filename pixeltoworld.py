import numpy as np
import cv2

class PixelToPoint:
    """
    Parameters
    ----------
    fx, fy : float
        Focal lengths (pixels).
    cx, cy : float
        Principal‑point coordinates (pixels).
    dist_coeffs : (N,) array_like
        Radial & tangential distortion coefficients understood by OpenCV.
        If your images are already undistorted, pass an empty tuple.
    camera_height : float
        Height of the camera above the plane (metres, >0).
    hfov, vfov : float, optional
        Horizontal/vertical field of view *in degrees*.
        *Only* used for sanity checks; not required for the maths.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist_coeffs=(),
        *,
        camera_height: float,
        hfov: float | None = None,
        vfov: float | None = None,
    ) -> None:
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32).reshape(-1, 1)
        self.H = float(camera_height)

        # 3×3 intrinsic matrix and its inverse for fast back‑projection
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]], dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K)

        # Optional FOV checks
        self.hfov = float(hfov) if hfov is not None else None
        self.vfov = float(vfov) if vfov is not None else None

        if self.H <= 0:
            raise ValueError("camera_height must be positive.")

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def __call__(self, uv: np.ndarray) -> np.ndarray:
        """
        Vectorised conversion from (u, v) → (x, y).

        Parameters
        ----------
        uv : (N,2) array_like
            Pixel coordinates in the image (u horizontal, v vertical).

        Returns
        -------
        xy : (N,2) ndarray
            World‑plane coordinates in metres.  x is rightward, y is forward.
        """
        uv = np.asarray(uv, dtype=np.float32).reshape(-1, 1, 2)

        # 1. Optional undistortion (returns normalised points)
        if self.dist_coeffs.size:
            pts_n = cv2.undistortPoints(
                uv, cameraMatrix=self.K, distCoeffs=self.dist_coeffs,
                P=self.K  # re‑apply intrinsics so we stay in pixel space
            )                          # shape (N,1,2)
            uv = pts_n

        # 2. Homogeneous: [u,v,1]^T
        ones = np.ones((uv.shape[0], 1, 1), dtype=np.float32)
        uv_h = np.concatenate([uv, ones], axis=2)   # (N,1,3)

        # 3. Back‑project to a direction vector in camera coords
        dirs = (self.K_inv @ uv_h.transpose(0, 2, 1)).squeeze(-1)  # (N,3)

        # 4. Intersection with Z = –H  (camera looks along +Z, plane at Z=0)
        #    Solve  h + t*Zc = 0  →  t = -H / Zc
        t = -self.H / dirs[:, 2]                          # (N,)
        xy = (dirs[:, :2].T * t).T                        # scale X,Y

        return xy.astype(np.float32)
    
    def run(self, uv: np.ndarray) -> np.ndarray:
        return self.__call__(uv)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def pixels_per_metre(self, v: float | int | None = None) -> float:
        """
        Convenience: approximate ground sample distance at image row **v**.

        Using the pin-hole model with no distortion.

        Parameters
        ----------
        v : int | float | None
            Image row.  If *None* uses the principal point (cy).

        Returns
        -------
        float
            Pixels per metre at that scanline.
        """
        v = self.cy if v is None else v
        theta = np.arctan((v - self.cy) / self.fy)  # angle below optical axis
        GSD = self.H * np.tan(theta + 1e-12)        # avoid div‑by‑zero
        return 1.0 / GSD if GSD != 0 else np.inf
