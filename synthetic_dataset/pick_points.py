"""Click the key distances on a reference image and save them to shapes/distance.txt.

python pick_points.py shapes/shape9.png [--distances shapes/distance.txt]

  left click        place the next point (two clicks = one distance: D1, D2, ...)
  right click / u   undo the last point
  mouse wheel       zoom in / out around the cursor
  middle drag       pan (or W / A / D / X = up / left / right / down)
  s                 save: replaces this image's section in distance.txt (old file kept as distance.txt.bak)
  q / Esc           quit (asks to save if there are unsaved points)

Coordinates are full-resolution pixels of the image file (0,0 = top-left, x right, y down), which is
what make_dataset.py expects.  A magnifier in the top-left corner shows the area under the cursor at 4x.
Existing points for this image in distance.txt are loaded, so you can add to or correct them.
Section name: shape9.png -> "9:", shape9_2.png -> "9_2:".
"""
import argparse, os, re, shutil
import cv2
import numpy as np

COLORS = [(60, 60, 235), (80, 200, 60), (40, 215, 245), (235, 140, 40), (220, 80, 220),
          (230, 230, 60), (40, 140, 255), (160, 160, 255), (150, 255, 150), (255, 255, 255)]
WIN, VIEW = 'pick points', (1400, 900)          # window name, window size (w, h)


def section_name(path):
    m = re.fullmatch(r'shape(\d+)(?:_(\d+))?', os.path.splitext(os.path.basename(path))[0])
    if not m:
        raise SystemExit('image must be named shapeN.png or shapeN_K.png')
    return m.group(1) + (f'_{m.group(2)}' if m.group(2) and m.group(2) != '1' else '')


def read_section(txt, name):
    """Points of section `name` in distance.txt (comments after '#' ignored)."""
    pts, cur = [], None
    for line in txt.splitlines():
        line = line.split('#')[0].strip()
        h = re.fullmatch(r'(\d+(?:_\d+)?)\s*[:.]', line)
        if h:
            cur = h.group(1)
            continue
        if cur == name:
            for x, y in re.findall(r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', line):
                pts.append((float(x), float(y)))
    return pts


def write_section(txt, name, pts):
    """distance.txt text with section `name` replaced by `pts` (appended if it did not exist)."""
    body = [f'({p[0]:.0f}, {p[1]:.0f}) ({q[0]:.0f}, {q[1]:.0f})' for p, q in zip(pts[0::2], pts[1::2])]
    out, skip, done = [], False, False
    for line in txt.splitlines():
        h = re.fullmatch(r'(\d+(?:_\d+)?)\s*[:.]', line.split('#')[0].strip())
        if h:
            skip = h.group(1) == name
            if skip:
                out += [f'{name}:'] + body
                done = True
                continue
        if not skip:
            out.append(line)
    if not done:
        while out and not out[-1].strip():
            out.pop()
        out += ([''] if out else []) + [f'{name}:'] + body
    return '\n'.join(out) + '\n'


def render(img, pts, zoom, ox, oy, cursor):
    """Window picture: image at `zoom` with top-left image pixel (ox, oy), points, lines and a magnifier."""
    w, h = VIEW
    M = np.float32([[zoom, 0, -ox * zoom], [0, zoom, -oy * zoom]])
    view = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_AREA if zoom < 1 else cv2.INTER_NEAREST,
                          borderValue=(40, 40, 40))

    def to_view(p):
        return int(round((p[0] + .5 - ox) * zoom - .5)), int(round((p[1] + .5 - oy) * zoom - .5))
    for k, p in enumerate(pts):
        c = COLORS[(k // 2) % len(COLORS)]
        if k % 2:
            cv2.line(view, to_view(pts[k - 1]), to_view(p), c, 2, cv2.LINE_AA)
            m = ((np.array(to_view(pts[k - 1])) + to_view(p)) // 2)
            cv2.putText(view, f'D{k // 2 + 1}', tuple(int(v) for v in m), 0, .6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, f'D{k // 2 + 1}', tuple(int(v) for v in m), 0, .6, c, 1, cv2.LINE_AA)
        cv2.circle(view, to_view(p), 5, c, 2, cv2.LINE_AA)
    if cursor is not None:                                   # 4x magnifier of the full-res pixels under the cursor
        cx, cy = int(cursor[0]), int(cursor[1])
        r = 40
        patch = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT, value=(40, 40, 40))[cy:cy + 2 * r + 1,
                                                                                              cx:cx + 2 * r + 1]
        mag = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        for k, p in enumerate(pts):                          # points that fall inside the magnifier
            q = (int((p[0] - cx + r) * 4 + 2), int((p[1] - cy + r) * 4 + 2))
            if 0 <= q[0] < mag.shape[1] and 0 <= q[1] < mag.shape[0]:
                cv2.circle(mag, q, 6, COLORS[(k // 2) % len(COLORS)], 2, cv2.LINE_AA)
        c = mag.shape[0] // 2
        cv2.line(mag, (c, c - 12), (c, c + 12), (0, 255, 255), 1)
        cv2.line(mag, (c - 12, c), (c + 12, c), (0, 255, 255), 1)
        cv2.rectangle(mag, (0, 0), (mag.shape[1] - 1, mag.shape[0] - 1), (255, 255, 255), 1)
        view[:mag.shape[0], :mag.shape[1]] = mag
    n = len(pts)
    status = (f'x={cursor[0]:.0f} y={cursor[1]:.0f}   ' if cursor is not None else '') + \
             f'{n // 2} distances' + (' + 1 point' if n % 2 else '') + \
             '   [click] point  [right/u] undo  [wheel] zoom  [WADX] pan  [s] save  [q] quit'
    cv2.rectangle(view, (0, h - 26), (w, h), (0, 0, 0), -1)
    cv2.putText(view, status, (8, h - 8), 0, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image')
    ap.add_argument('--distances', default=os.path.join(here, 'shapes', 'distance.txt'))
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f'could not read {a.image}')
    name = section_name(a.image)
    txt = open(a.distances).read() if os.path.exists(a.distances) else ''
    pts = read_section(txt, name)
    saved = list(pts)
    H, W = img.shape[:2]
    zoom = min(VIEW[0] / W, (VIEW[1] - 26) / H)
    st = dict(zoom=zoom, ox=0.0, oy=0.0, cursor=None, pan=None, dirty=True)

    def img_xy(x, y):
        return (x + .5) / st['zoom'] - .5 + st['ox'], (y + .5) / st['zoom'] - .5 + st['oy']

    def on_mouse(ev, x, y, flags, _):
        p = img_xy(x, y)
        st['cursor'], st['dirty'] = (p if 0 <= p[0] < W and 0 <= p[1] < H else None), True
        if ev == cv2.EVENT_LBUTTONDOWN and st['cursor'] is not None:
            pts.append((round(p[0]), round(p[1])))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
        elif ev == cv2.EVENT_MOUSEWHEEL:
            # wheel direction is the sign of the high 16 bits of `flags` (cv2.getMouseWheelDelta is
            # missing from some OpenCV builds); flags arrives as a signed 32-bit int, so its sign is that sign
            if flags >= 2 ** 31:                         # some builds pass it unsigned
                flags -= 2 ** 32
            f = 1.25 if flags > 0 else 0.8
            z = min(max(st['zoom'] * f, 0.05), 16)
            st['ox'], st['oy'] = p[0] + .5 - (x + .5) / z, p[1] + .5 - (y + .5) / z   # keep p under the cursor
            st['zoom'] = z
        elif ev == cv2.EVENT_MBUTTONDOWN:
            st['pan'] = (x, y, st['ox'], st['oy'])
        elif ev == cv2.EVENT_MOUSEMOVE and st['pan'] and flags & cv2.EVENT_FLAG_MBUTTON:
            x0, y0, ox, oy = st['pan']
            st['ox'], st['oy'] = ox - (x - x0) / st['zoom'], oy - (y - y0) / st['zoom']
        elif ev == cv2.EVENT_MBUTTONUP:
            st['pan'] = None

    def save():
        nonlocal txt, saved
        if len(pts) % 2:
            print('odd number of points - the last point has no partner and is not saved')
        if os.path.exists(a.distances):
            shutil.copyfile(a.distances, a.distances + '.bak')
        txt = write_section(txt, name, pts[:len(pts) // 2 * 2])
        open(a.distances, 'w').write(txt)
        saved = list(pts[:len(pts) // 2 * 2])
        print(f'saved {len(saved) // 2} distances as "{name}:" in {a.distances}')

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    print(f'{a.image}: {len(pts) // 2} distances loaded from {a.distances}' if pts else f'{a.image}: no points yet')
    while True:
        if st['dirty']:
            cv2.imshow(WIN, render(img, pts, st['zoom'], st['ox'], st['oy'], st['cursor']))
            st['dirty'] = False
        k = cv2.waitKey(20) & 0xFF
        step = 100 / st['zoom']
        if k in (ord('q'), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            if pts[:len(pts) // 2 * 2] != saved and input('save changes? [y/n] ').strip().lower().startswith('y'):
                save()
            break
        elif k == ord('s'):
            save()
        elif k == ord('u') and pts:
            pts.pop()
        elif k in (ord('a'), 81):
            st['ox'] -= step
        elif k in (ord('d'), 83):
            st['ox'] += step
        elif k in (ord('w'), 82):
            st['oy'] -= step
        elif k in (ord('x'), 84):
            st['oy'] += step
        else:
            continue
        st['dirty'] = True
    cv2.destroyAllWindows()
    for i, (p, q) in enumerate(zip(pts[0::2], pts[1::2])):
        print(f'D{i + 1}: ({p[0]:.0f}, {p[1]:.0f}) ({q[0]:.0f}, {q[1]:.0f})')


if __name__ == '__main__':
    main()
