import cv2
import numpy as np

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)


img = cv2.imread('/home/chaman99/Project/Evening_data_4to6/frame_0050.jpg')
print(img.shape)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cv2.imshow("RGB", img)
cv2.waitKey()

# pts = np.array([[544,360],[598,360],[597, 457],[543, 457]], np.int32)
# pts = pts.reshape((-1,1,2))
# cv2.polylines(img,[pts],True,(0,255,255))


# # ==================== Start Extraction =========================
# rect = cv2.boundingRect(pts)
# x,y,w,h = rect
# croped = img[y:y+h, x:x+w].copy()
# ## (2) make mask
# pts = pts - pts.min(axis=0)
# mask = np.zeros(croped.shape[:2], np.uint8)
# cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
# ## (3) do bit-op
# dst = cv2.bitwise_and(croped, croped, mask=mask)
# ## (4) add the white background
# bg = np.ones_like(croped, np.uint8)*255
# cv2.bitwise_not(bg,bg, mask=mask)
# dst2 = bg + dst
# cv2.imwrite("croped3.png", croped)
# cv2.imwrite("mask3.png", mask)
# cv2.imwrite("dst3.png", dst)
# cv2.imwrite("dst333.png", dst2)
# # ==================== End Extraction =========================

# height = img.shape[0]
# width = img.shape[1]

# cv2.imshow("cropped3", cropped3)



# print("reached end")