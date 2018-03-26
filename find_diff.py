import argparse
import cv2
import glob
import numpy as np
import os
import subprocess
from collections import deque

from model import createModel


def merge_rects(rects):
    if not rects:
        return []

    unions = [rects[0]]

    for rect in rects[1:]:
        found = False

        idx = 0
        merged_at = None

        while idx < len(unions):
            union = unions[idx]

            if rect[2] <= union[2] and union[0] <= rect[0]:
                found = True
                # print('Merging2', rect, 'with', union)
                break

            if union[2] <= rect[2] and rect[0] <= union[0]:
                found = True
                # print('Merging1', rect, 'with', union)
                if merged_at is None:
                    union[0], union[1], union[2], union[3] = rect
                    merged_at = idx

                else:
                    unions[idx] = unions[-1]
                    unions.pop()
                    continue

            idx += 1

        if not found:
            unions.append(rect)

    return [tuple(u) for u in unions]


def get_diff_between_images(i1, i2):
    diff = cv2.absdiff(i1, i2)
    _, tres = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
    tres = cv2.dilate(tres, np.ones((5, 5), np.uint8), iterations=1)

    im2, contours, hierarchy = cv2.findContours(tres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if not (120 <= w * h <= 20000):
            continue

        rects.append((x, y, x + w, y + h))

    return diff, tres, rects


def main(clean=True, diff=False, reference=False, threshold=False, only_matches=False, match_file=None, reference_file=None):
    if clean:
        print('Cleaning output.')
        subprocess.call(['rm', '-rf', 'output'])

    if not match_file and not reference_file:
        filenames = sorted(glob.glob('2018-03-26_*/*.jpeg'))

    else:
        filenames = [reference_file, match_file]

    i1 = cv2.imread(filenames[0])
    i1_gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    reference_filename = filenames[0]
    os.makedirs('output', exist_ok=True)
    os.makedirs('matches', exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX

    model1 = createModel()
    model1.load_weights('weights.hdf5')

    last_images = deque(maxlen=2)
    counter = 0
    # filenames = ['2018-03-24_09/2018-03-24_09-03-1521882208.jpeg']
    for filename in filenames[1:]:
        orig_i2 = cv2.imread(filename)
        filename = os.path.basename(filename)
        i2_gray = cv2.cvtColor(orig_i2, cv2.COLOR_BGR2GRAY)

        i2_with_rects = orig_i2.copy()
        cv2.putText(i2_with_rects, f'{filename}', (10, 400), font, .7, (0, 255, 255), 2, cv2.LINE_AA)
        d, tres, rects = get_diff_between_images(i1_gray, i2_gray)

        # rects = merge_rects(rects)
        filtered_rects = []

        for rect in rects:
            matched_img = orig_i2[rect[1]:rect[3], rect[0]:rect[2]]
            img = cv2.resize(matched_img, (32, 32))
            predicted = model1.predict(np.array([img]))[0][0]
            print(predicted)
            if round(predicted):
                filtered_rects.append(rect)
                cv2.rectangle(i2_with_rects, rect[:2], rect[2:], (0, 255, 0), 1)

            else:
                cv2.rectangle(i2_with_rects, rect[:2], rect[2:], (255, 0, 0), 1)
                cv2.putText(i2_with_rects, str(round(predicted, 6)), (rect[0], rect[1] - 10), font, .7, (0, 255, 255), 2, cv2.LINE_AA)

        rects = filtered_rects

        output_filename = f'output/{filename}'
        if diff or threshold or reference:
            output_filename = f'output/{filename}-00.jpeg'

        # if len(rects):
        print('[Ref:', reference_filename, ']', filename, sorted([((x2 - x1) * (y2 - y1), (x2 - x1) / (y2 - y1)) for x1, y1, x2, y2 in rects]))
        cv2.imwrite(output_filename, i2_with_rects)
            # for idx, r in enumerate(rects):
            #     cv2.imwrite(f'matches/{filename}-{idx}.jpeg', orig_i2[r[1]:r[3], r[0]:r[2]])

        if len(rects) or not only_matches:
            if threshold:
                cv2.imwrite(f'output/{filename}-0-threshold.jpeg', tres)
                canny = cv2.Canny(orig_i2, 100, 200)
                cv2.imwrite(f'output/{filename}-0-contours.jpeg', canny)

            if diff:
                cv2.imwrite(f'output/{filename}-0-diff.jpeg', d)

            if reference:
                cv2.imwrite(f'output/{filename}-0-reference.jpeg', i1)

        diffs = 1
        # if len(rects):
        #     if len(last_images) == 2:
        #         diffs = sum(
        #             len(get_diff_between_images(i, i2_gray)[-1])
        #             for i in last_images
        #         )

        if not diffs or not len(rects):
            reference_filename = filename
            i1 = orig_i2
            i1_gray = i2_gray

        last_images.append(i2_gray)

# 2018-03-24_12-48-1521895739.jpeg-00

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dont-clean', default=False, action='store_true')
    parser.add_argument('--with-threshold', default=False, action='store_true')
    parser.add_argument('--with-reference', default=False, action='store_true')
    parser.add_argument('--with-diff', default=False, action='store_true')
    parser.add_argument('--only-matches', default=False, action='store_true')
    parser.add_argument('--match-file', default=None)
    parser.add_argument('--reference-file', default=None)

    args = parser.parse_args()
    main(
        clean=not args.dont_clean,
        threshold=args.with_threshold,
        reference=args.with_reference,
        diff=args.with_diff,
        only_matches=args.only_matches,
        match_file=args.match_file,
        reference_file=args.reference_file,
    )
