"""撮影画像から画面内のstego画像領域を検出し、homography補正して返す。

手法:オリジナル cover.png との ORB 特徴点マッチング + RANSAC homography。
→ 壁・家具・タスクバーなど画像外の背景に影響されない。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def align_to_reference(
    capture_path: str | Path,
    reference_path: str | Path,
    target_w: int | None = None,
    target_h: int | None = None,
    debug_dir: Path | None = None,
    n_features: int = 5000,
    ratio_thresh: float = 0.8,
    ransac_thresh: float = 5.0,
    use_sift: bool = True,
) -> tuple[Image.Image, np.ndarray, int] | None:
    """撮影画像を reference 画像の座標系に揃える。

    Returns:
        (warped_image, homography, num_inliers) or None
    """
    ref_bgr = cv2.imread(str(reference_path))
    cap_bgr = cv2.imread(str(capture_path))
    if ref_bgr is None or cap_bgr is None:
        return None

    tw = target_w or ref_bgr.shape[1]
    th = target_h or ref_bgr.shape[0]

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    cap_gray = cv2.cvtColor(cap_bgr, cv2.COLOR_BGR2GRAY)

    # 撮影画像は通常解像度が高いので適度に縮小して高速化
    cap_h, cap_w = cap_gray.shape
    max_dim = 2000
    if max(cap_h, cap_w) > max_dim:
        s = max_dim / max(cap_h, cap_w)
        cap_gray_small = cv2.resize(cap_gray, (int(cap_w * s), int(cap_h * s)))
        cap_scale = s
    else:
        cap_gray_small = cap_gray
        cap_scale = 1.0

    if use_sift:
        detector = cv2.SIFT_create(nfeatures=n_features)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=n_features)
        norm = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(ref_gray, None)
    kp2, des2 = detector.detectAndCompute(cap_gray_small, None)
    if des1 is None or des2 is None:
        return None

    matcher = cv2.BFMatcher(norm)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 10:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # H: capture(縮小) → ref
    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None

    inliers = int(mask.sum())
    # フルサイズ cap 座標を縮小 cap 座標に落とすには cap_scale を掛ける(cap_scale < 1)
    scale_mat = np.array([[cap_scale, 0, 0], [0, cap_scale, 0], [0, 0, 1]], dtype=np.float64)
    H_full = H @ scale_mat

    warped_bgr = cv2.warpPerspective(cap_bgr, H_full, (tw, th))
    warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        name = Path(capture_path).stem
        # マッチング可視化
        good_inlier = [g for g, m in zip(good, mask.flatten()) if m]
        # 縮小キャプチャで描画
        vis = cv2.drawMatches(
            ref_gray, kp1, cap_gray_small, kp2, good_inlier[:80], None,
            matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=2,
        )
        cv2.imwrite(str(debug_dir / f"{name}_matches.jpg"), vis)
        cv2.imwrite(str(debug_dir / f"{name}_warped.png"), warped_bgr)

    return Image.fromarray(warped_rgb), H_full, inliers
