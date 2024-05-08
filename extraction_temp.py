import cv2
import numpy as np

bays = [
    [[107, 64], [138, 59], [120, 95], [82, 104]],
    [[138, 59], [161, 53], [139, 92], [118, 97]],
    [[161, 53], [183, 49], [164, 85], [141, 92]],
    [[588, 3], [620, 4], [627, 44], [589, 43]],
    [[160, 204], [194, 198], [172, 265], [144, 265]],
    [[839, 355], [878, 354], [886, 443], [847, 444]]
]

def resize_and_pad(img, size=(150, 150), pad_color=255):
    h, w = img.shape[:2]
    sh, sw = size

    # Interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # Aspect ratio of image
    aspect = w / h

    # Computing scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # Resize image
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[pad_color, pad_color, pad_color])

    return img_padded

img = cv2.imread('/home/chaman99/Project/Evening_data_4to6/frame_0051.jpg')
i = 1
for bay in bays:
    pts = np.array(bay, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 255))

    # Start Extraction
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)

    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(cropped, cropped, mask=mask)

    # Resize and pad
    dst_resized_padded = resize_and_pad(dst)

    cv2.imwrite(f"crop{i}.png", dst_resized_padded)
    i += 1

print("Reached end")
