import cv2

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"x: {x}, y: {y}")

img = cv2.imread('/home/chaman99/Project/Evening_data_4to6/frame_0054.jpg')

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cv2.imshow("RGB", img)
cv2.waitKey()
