"""Synthetic cut-end images of a U 0-24/77-like profile with known dimensions.

Shape (part frame, mm, y up from the outer base): flat-ish base of thickness T, tapered keyhole cavity that is
widest W at T+4 mm and closes at the tips, legs leaning inward, domed tips that touch (thin dark seam) or,
with gap_mm > 0, leave the mouth open. Drawn blue on a dark brown background with foam texture and a glare
spot in the cavity.
"""
import cv2
import numpy as np

PX_PER_MM = 20.0
IMAGE_SIZE = 2000
BG_BGR = (35, 45, 60)
FOAM_BGR = (185, 125, 45)
CAVITY_BGR = (22, 22, 26)


def profile_mm(length=78.0, inner_width=25.5, base_thickness=12.0, outer_width=46.0, lean=4.0,
               tip_solid=7.0, n=400, lean_right=None):
    """(outer, cavity) polygons in mm, y up, x centred, base at y=0. `lean_right` (default = lean) lets the
    right leg's outside lean differently, making the part asymmetric (so mirror images can be told apart)."""
    L, W, T, Wo = length, inner_width, base_thickness, outer_width
    r = 6.0                                  # base corner radius
    y = np.linspace(0.0, L, n)
    dome = 8.0

    def half_width(lean_):
        h = np.where(y < r, Wo / 2 - r + r * np.clip(1 - (np.clip(r - y, 0, r) / r) ** 2.5, 0, 1) ** (1 / 2.5),
                     Wo / 2 - lean_ * (y - r) / (L - r))
        return np.where(y > L - dome, h * np.sqrt(np.clip(1 - ((y - (L - dome)) / dome) ** 2, 0, 1)), h)

    h_r = half_width(lean if lean_right is None else lean_right)
    h_l = half_width(lean)
    outer = np.vstack([np.column_stack([h_r, y]), np.column_stack([-h_l[::-1], y[::-1]])])

    y_top = L - tip_solid
    yc = np.linspace(T, y_top, n)
    yw = T + 4.0
    c = np.where(yc < yw, W / 2 * np.clip(1 - ((yw - yc) / 4.0) ** 4, 0, 1) ** 0.25,
                 W / 2 * np.clip((y_top - yc) / (y_top - yw), 0, 1) ** 0.7)
    cavity = np.vstack([np.column_stack([c, yc]), np.column_stack([-c[::-1], yc[::-1]])])
    return outer, cavity


def _to_image(pts_mm, centre_px, angle_deg, ppm, sx=1.0, sy=1.0, y_mid=39.0):
    p = np.column_stack([pts_mm[:, 0] * sx, -(pts_mm[:, 1] - y_mid) * sy])     # y down, centred
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (p @ rot.T) * ppm + centre_px


def _fill(shape, polys, ppm_shift=4):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(p * (1 << ppm_shift)).astype(np.int32) for p in polys], 255,
                 lineType=cv2.LINE_AA, shift=ppm_shift)
    return m.astype(np.float32) / 255


def render(angle_deg=0.0, sx=1.0, sy=1.0, gap_mm=0.0, seam_mm=0.15, glare=True, seed=0, offset_px=(0, 0),
           size=IMAGE_SIZE, ppm=PX_PER_MM, **shape):
    """BGR image of the profile. sx/sy scale the part's width/length (for out-of-spec variants)."""
    rng = np.random.default_rng(seed)
    outer, cavity = profile_mm(**shape)
    L = shape.get("length", 78.0)
    centre = np.array([size / 2, size / 2]) + offset_px     # offset_px moves the part (e.g. tips out of frame)
    tf = lambda pts: _to_image(pts, centre, angle_deg, ppm, sx, sy, y_mid=L / 2)
    hw = shape.get("inner_width", 25.5)

    a_outer = _fill((size, size), [tf(outer)])
    a_cav = _fill((size, size), [tf(cavity)])
    holes = []
    y_top = L - shape.get("tip_solid", 7.0)
    if gap_mm > 0:   # open legs: slot from the cavity tip through the domed tips
        g = gap_mm / 2
        holes.append(np.array([[-g, y_top - 3], [g, y_top - 3], [g, L + 2], [-g, L + 2]]))
    elif seam_mm > 0:  # touching tips: thin dark seam
        g = seam_mm / 2
        holes.append(np.array([[-g, y_top - 1], [g, y_top - 1], [g, L + 0.5], [-g, L + 0.5]]))
    a_slot = _fill((size, size), [tf(h) for h in holes]) if holes else np.zeros((size, size), np.float32)
    # An open mouth shows the background; a closed seam is a dark crack in the foam.
    seam = 0.0 if gap_mm > 0 else 1.0

    img = np.empty((size, size, 3), np.float32)
    img[:] = BG_BGR
    img += rng.normal(0, 5, img.shape)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 3)
    foam = np.array(FOAM_BGR, np.float32) * (1 + 0.12 * texture / texture.std())[..., None]
    cav = np.array(CAVITY_BGR, np.float32) + rng.normal(0, 3, img.shape)

    a_foam = np.clip(a_outer - a_cav - a_slot, 0, 1)[..., None]
    a_dark = np.clip(a_cav + seam * a_slot * a_outer, 0, 1)[..., None]
    img = img * (1 - a_foam - a_dark) + foam * a_foam + cav * a_dark

    if glare:  # bright, unsaturated spot in the cavity near its widest part
        gm = np.zeros((size, size), np.float32)
        gc = tf(np.array([[hw * 0.1, shape.get("base_thickness", 12.0) + 9.0]]))[0]
        cv2.ellipse(gm, (int(gc[0]), int(gc[1])), (int(2.5 * ppm), int(1.5 * ppm)), angle_deg, 0, 360, 1.0, -1)
        gm = cv2.GaussianBlur(gm, (0, 0), 4)[..., None]
        img = img * (1 - gm) + np.array([235, 238, 240], np.float32) * gm

    return np.clip(img, 0, 255).astype(np.uint8)


def warp(img, angle_deg=0.0, scale=1.0, mirror=False):
    """Image-level variant: rotate/scale about the centre, optional horizontal mirror."""
    if mirror:
        img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=BG_BGR)


# Ground truth of the default shape
TRUE = dict(length=78.0, inner_width_max=25.5, base_thickness=12.0, outer_width_max=46.0)

# Slightly different "good" parts for building a reference
GOOD_PART_SHAPES = [
    dict(length=78.0, inner_width=25.5, base_thickness=12.0, outer_width=46.0),
    dict(length=78.3, inner_width=25.3, base_thickness=12.2, outer_width=46.2),
    dict(length=77.7, inner_width=25.7, base_thickness=11.9, outer_width=45.8),
    dict(length=78.2, inner_width=25.6, base_thickness=12.1, outer_width=46.1, lean=4.3),
    dict(length=77.9, inner_width=25.4, base_thickness=11.8, outer_width=45.9, lean=3.8),
]
GOOD_PART_ANGLES = [0.0, 37.0, 101.0, 158.0, 233.0]
