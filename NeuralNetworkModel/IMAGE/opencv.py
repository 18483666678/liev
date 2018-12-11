import cv2

img = cv2.imread("2.jpg")
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#画线
green = (0,255,0)
cv2.line(img,(0,0),(300,400),green,3)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#画圆
green = (0,255,0)
cv2.circle(img,(150,150),100,green,2)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#画矩形
green = (0,255,0)
cv2.rectangle(img,(50,100),(150,200),green,2)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()