import numpy as np
from collections import deque


# ============================================================
# Hilfsfunktionen: Resize, Faltungen, Filter
# ============================================================


def resize_bilinear(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """
    Bilineares Resize (ähnlich OpenCV INTER_LINEAR), vektorisiert.
    img: 2D uint8/float32, shape (H, W)
    """
    img = img.astype(np.float32)
    h, w = img.shape

    # Zielkoordinaten
    ys = (np.arange(new_h) + 0.5) * (h / new_h) - 0.5
    xs = (np.arange(new_w) + 0.5) * (w / new_w) - 0.5

    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    wy = (ys - y0)[..., None]  # (new_h, 1)
    wx = (xs - x0)[None, ...]  # (1, new_w)

    # Vier Eckwerte via fancy indexing
    Ia = img[y0[:, None], x0[None, :]]  # top-left
    Ib = img[y0[:, None], x1[None, :]]  # top-right
    Ic = img[y1[:, None], x0[None, :]]  # bottom-left
    Id = img[y1[:, None], x1[None, :]]  # bottom-right

    top = Ia * (1 - wx) + Ib * wx
    bottom = Ic * (1 - wx) + Id * wx
    out = top * (1 - wy) + bottom * wy

    return out.astype(np.uint8)


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2D-Faltung (Cross-Correlation) mit reflect-Padding.
    Vektorisiert über Kernel (keine Pixel-Schleifen).
    """
    img = img.astype(np.float32)
    kernel = kernel.astype(np.float32)

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float32)

    # Schleife nur über Kernel-Offsets, nicht über Pixel
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i : i + H, j : j + W]

    return out


def gaussian_kernel_1d(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    1D-Gausskern, normalisiert.
    """
    ax = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Separable Gaussian Blur: erst horizontal, dann vertikal.
    """
    img = img.astype(np.float32)
    kernel = gaussian_kernel_1d(ksize, sigma)

    # Horizontal
    pad = ksize // 2
    padded = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    H, W = img.shape
    tmp = np.zeros_like(img, dtype=np.float32)
    for j in range(ksize):
        tmp += kernel[j] * padded[:, j : j + W]

    # Vertikal
    padded2 = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(tmp, dtype=np.float32)
    for i in range(ksize):
        out += kernel[i] * padded2[i : i + H, :]

    return out


def sobel_gradients(img: np.ndarray):
    """
    Sobel-Gradienten in x/y, Magnitude und Winkel.
    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = convolve2d(img, Kx)
    gy = convolve2d(img, Ky)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    return mag, ang


# ============================================================
# Entropy
# ============================================================


def entropy_check(img: np.ndarray, threshold: float = 5.0):
    """
    Shannon-Entropie über 256-Bin-Histogramm.
    """
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float32)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))
    passed = entropy >= threshold
    return entropy, passed


# ============================================================
# FFT-Ratio
# ============================================================


def fft_check(img_gray: np.ndarray, threshold: float = 0.15):
    """
    Low-Frequency-Ratio im FFT-Spektrum.
    Erwartet beliebige Größe; intern auf 128x128 rescaled.
    """
    small = resize_bilinear(img_gray, 128, 128)

    f = np.fft.fft2(small)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    center = magnitude[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    ratio = float(center.sum() / magnitude.sum())
    passed = ratio < threshold
    return ratio, passed


# ============================================================
# Otsu Threshold
# ============================================================


def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """
    Otsu-Thresholding, NumPy-only.
    """
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = img.size

    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = t

    binary = (img > threshold).astype(np.uint8) * 255
    return binary


# ============================================================
# Connected Components (Union-Find)
# ============================================================


def components_check(binary_img: np.ndarray, threshold: int = 10):
    """
    8-connectivity Connected Components, Union-Find-basierter Zwei-Pass-Ansatz.
    binary_img: 0 Hintergrund, !=0 Vordergrund.
    """
    img = binary_img != 0
    h, w = img.shape
    labels = np.zeros((h, w), dtype=np.int32)

    # Union-Find Strukturen
    max_labels = h * w // 2 + 1
    parent = np.arange(max_labels, dtype=np.int32)
    rank = np.zeros(max_labels, dtype=np.int32)
    next_label = 1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Erster Pass
    for y in range(h):
        for x in range(w):
            if not img[y, x]:
                continue

            neighbors = []
            # 8er-Nachbarn links/oben
            for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if labels[ny, nx] > 0:
                        neighbors.append(labels[ny, nx])

            if not neighbors:
                labels[y, x] = next_label
                next_label += 1
            else:
                m = min(neighbors)
                labels[y, x] = m
                for n in neighbors:
                    if n != m:
                        union(m, n)

    # Zweiter Pass: Label-Flachlegung
    label_map = {}
    current = 1
    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                root = find(labels[y, x])
                if root not in label_map:
                    label_map[root] = current
                    current += 1
                labels[y, x] = label_map[root]

    components = current - 1
    passed = components >= threshold
    return components, passed


# ============================================================
# Canny (vektorisierte NMS + Hysteresis)
# ============================================================


def nonmax_suppression(mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """
    Non-Maximum Suppression, größtenteils vektorisiert.
    """
    H, W = mag.shape
    Z = np.zeros((H, W), dtype=np.float32)

    ang_deg = ang * 180.0 / np.pi
    ang_deg[ang_deg < 0] += 180

    # Richtungs-Quantisierung
    # 0°, 45°, 90°, 135°
    # Masken
    dir0 = ((0 <= ang_deg) & (ang_deg < 22.5)) | ((157.5 <= ang_deg) & (ang_deg <= 180))
    dir45 = (22.5 <= ang_deg) & (ang_deg < 67.5)
    dir90 = (67.5 <= ang_deg) & (ang_deg < 112.5)
    dir135 = (112.5 <= ang_deg) & (ang_deg < 157.5)

    # Hilfsfunktion: vergleicht mit zwei Nachbarn in gegebener Richtung
    def suppress(mask, dy1, dx1, dy2, dx2):
        yy, xx = np.where(mask)
        y1 = np.clip(yy + dy1, 0, H - 1)
        x1 = np.clip(xx + dx1, 0, W - 1)
        y2 = np.clip(yy + dy2, 0, H - 1)
        x2 = np.clip(xx + dx2, 0, W - 1)

        m0 = mag[yy, xx]
        m1 = mag[y1, x1]
        m2 = mag[y2, x2]

        keep = (m0 >= m1) & (m0 >= m2)
        Z[yy[keep], xx[keep]] = m0[keep]

    suppress(dir0, 0, -1, 0, 1)
    suppress(dir45, -1, 1, 1, -1)
    suppress(dir90, -1, 0, 1, 0)
    suppress(dir135, -1, -1, 1, 1)

    return Z


def hysteresis_thresholding(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Hysteresis mit iterativer Nachbarschaftsaktivierung, vektorisiert.
    """
    strong_val = 255
    weak_val = 50

    strong = img >= high
    weak = (img >= low) & (img < high)

    result = np.zeros_like(img, dtype=np.uint8)
    result[strong] = strong_val
    result[weak] = weak_val

    changed = True
    H, W = img.shape

    while changed:
        changed = False

        # Nachbarschaft eines strong-Pixels:
        strong_mask = result == strong_val

        # 8er-Nachbarschaft via Shifts
        neigh = np.zeros_like(strong_mask, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.zeros_like(strong_mask, dtype=bool)
                if dy >= 0:
                    ys = slice(dy, H)
                    yd = slice(0, H - dy)
                else:
                    ys = slice(0, H + dy)
                    yd = slice(-dy, H)
                if dx >= 0:
                    xs = slice(dx, W)
                    xd = slice(0, W - dx)
                else:
                    xs = slice(0, W + dx)
                    xd = slice(-dx, W)
                shifted[yd, xd] = strong_mask[ys, xs]
                neigh |= shifted

        # Weak-Pixel, die an strong angrenzen, werden zu strong
        promote = (result == weak_val) & neigh
        if np.any(promote):
            result[promote] = strong_val
            changed = True

    result[result != strong_val] = 0
    return result


def canny_numpy(img: np.ndarray, low: float = 50.0, high: float = 100.0) -> np.ndarray:
    """
    Vollständiger Canny-Edge-Detector, NumPy-only.
    """
    blur = gaussian_blur(img, ksize=5, sigma=1.0)
    mag, ang = sobel_gradients(blur)
    nms = nonmax_suppression(mag, ang)
    edges = hysteresis_thresholding(nms, low, high)
    return edges


# ============================================================
# Edge-Dichte
# ============================================================


def edge_density_check(edges: np.ndarray, threshold: float = 0.2):
    """
    Edge-Dichte: mean(edges)/255.0, erwartet 0/255-Bild.
    """
    density = float(edges.mean() / 255.0)
    passed = density >= threshold
    return density, passed


# ============================================================
# Gesamtanalyse mit Gewichtung
# ============================================================


def analyze_image(img_gray: np.ndarray):
    """
    Führt alle vier Checks durch und berechnet den gewichteten Score.
    img_gray: 2D uint8-Array.
    """
    # 1) Entropy
    entropy_val, entropy_ok = entropy_check(img_gray)

    # 2) FFT ratio
    fft_ratio, fft_ok = fft_check(img_gray)

    # 3) Components
    binary = otsu_threshold(img_gray)
    components_cnt, components_ok = components_check(binary)

    # 4) Edges
    edges = canny_numpy(img_gray)
    edge_density, edges_ok = edge_density_check(edges)

    # Gewichteter Score
    score = 0
    if components_ok:
        score += 2
    if edges_ok:
        score += 2
    if entropy_ok:
        score += 1
    if fft_ok:
        score += 1

    return {
        "entropy": (entropy_val, entropy_ok),
        "fft_ratio": (fft_ratio, fft_ok),
        "components": (components_cnt, components_ok),
        "edge_density": (edge_density, edges_ok),
        "score": score,
    }
