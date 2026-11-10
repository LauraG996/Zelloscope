"""Plot the key distances (reference points) on the transformed images.

The key points of every transformed image are the reference points pushed through that image's
geometric transform (computed by make_dataset.py and stored in transformed_points.csv).

python plot_reference_points.py [--csv dataset_out/transformed_points.csv] [--images dataset_out/raw]
                                [--out dataset_out/plotted] [--shapes 1 2] [--n 10]
"""
import argparse, csv, os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = [(60, 60, 235), (80, 200, 60), (40, 215, 245), (235, 140, 40), (220, 80, 220),   # BGR: D1 red, D2 green,
          (230, 230, 60), (40, 140, 255), (160, 160, 255), (150, 255, 150), (255, 255, 255)]  # D3 yellow, ... (cycled)


def draw_distances(img, kp):
    out = img.copy()
    for i, (a, b) in enumerate(kp):
        pa, pb = tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b)
        cv2.line(out, pa, pb, COLORS[i % len(COLORS)], 2, cv2.LINE_AA)
        for p in (pa, pb):
            cv2.circle(out, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, p, 5, (0, 0, 0), 1, cv2.LINE_AA)
        m = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
        txt = f'D{i + 1} {np.linalg.norm(np.asarray(a) - np.asarray(b)):.0f}px'
        cv2.putText(out, txt, m, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, m, cv2.FONT_HERSHEY_SIMPLEX, .5, COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(HERE, 'dataset_out', 'transformed_points.csv'))
    ap.add_argument('--images', default=os.path.join(HERE, 'dataset_out', 'raw'))
    ap.add_argument('--out', default=os.path.join(HERE, 'dataset_out', 'plotted'))
    ap.add_argument('--shapes', type=int, nargs='*', help='only these shape numbers')
    ap.add_argument('--n', type=int, help='max images per shape')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seen = {}
    for r in csv.DictReader(open(a.csv)):
        s = int(r['shape'])
        if (a.shapes and s not in a.shapes) or (a.n and seen.get(s, 0) >= a.n):
            continue
        seen[s] = seen.get(s, 0) + 1
        ds = sorted(int(k[1:-3]) for k in r if k.startswith('D') and k.endswith('_x1') and r[k] != '')
        kp = np.array([[[float(r[f'D{d}_x{e}']), float(r[f'D{d}_y{e}'])] for e in (1, 2)] for d in ds])
        img = cv2.imread(os.path.join(a.images, r['image']))
        cv2.imwrite(os.path.join(a.out, r['image'].replace('.png', '.jpg')), draw_distances(img, kp),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('plotted', sum(seen.values()), 'images to', a.out)


if __name__ == '__main__':
    main()
