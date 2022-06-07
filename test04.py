import cv2
import numpy as np


def canny():
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_img = cv2.Canny(blur, 50, 150)
    return canny_img


def region_of_interest(r_image):
    h = r_image.shape[0]
    w = r_image.shape[1]
    # 這個區域不穩定，需要根據圖片更換(左下, 右下, 右上, 左上)
    #[(0, h), (790, h), (400, 300), (200, 300)] #02.jpg
    poly = np.array([
                    [(0, h), (470, h), (280, 80), (170, 80)]
                    ])
    mask = np.zeros_like(r_image)
    cv2.fillPoly(mask, poly, 255)
    masked_image = cv2.bitwise_and(r_image, mask)
    return masked_image


def get_lines(img_lines):
    if img_lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
            # 分左右車道
                k = (y2 - y1) / (x2 - x1)
                if k < 0:
                    lefts.append(line)
                else:
                    rights.append(line)



def choose_lines(after_lines, slo_th): # 過濾斜率差別較大的點
    slope = [(y2 - y1) / (x2 - x1) for line in after_lines for x1, x2, y1, y2 in line] # 獲得斜率數組
    while len(after_lines) > 0:
        mean = np.mean(slope) # 計算平均斜率
        diff = [abs(s - mean) for s in slope] # 每條線斜率與平均斜率的差距
        idx = np.argmax(diff) # 找到最大斜率的索引
        if diff[idx] > slo_th: # 大於預設的閾值選取
            slope.pop(idx)
            after_lines.pop(idx)
        else:break

    return after_lines


def clac_edgepoints(points, y_min, y_max):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    k = np.polyfit(y, x, 1) # 曲線擬合的函數，找到xy的擬合關系斜率
    func = np.poly1d(k) # 斜率代入可以得到一個y=kx的函數
    x_min = int(func(y_min)) # y_min = 325其實是近似找瞭一個
    x_max = int(func(y_max))
    return [(x_min, y_min), (x_max, y_max)]


if __name__ == '__main__':
    image = cv2.imread('a01.jpg')
    lane_image = np.copy(image)
    canny_img = canny()
    cropped_image = region_of_interest(canny_img)
    lefts = []
    rights = []
    #HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None) 
    #lines = cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 15, np.array([]), minLineLength=40, maxLineGap=20)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 60, np.array([]), minLineLength=180, maxLineGap=60)
    get_lines(lines) # 分別得到左右車道線的圖片

    good_leftlines = choose_lines(lefts, 0.1) # 處理後的點
    good_rightlines = choose_lines(rights, 0.1)
    print(good_leftlines)
    leftpoints = [(x1, y1) for left in good_leftlines for x1, y1, x2, y2 in left]
    #print(leftpoints)
    leftpoints = leftpoints + [(x2, y2) for left in good_leftlines for x1, y1, x2, y2 in left]
    #print(leftpoints, '2')
    rightpoints = [(x1, y1) for right in good_rightlines for x1, y1, x2, y2 in right]
    rightpoints = rightpoints + [(x2, y2) for right in good_rightlines for x1, y1, x2, y2 in right]

    lefttop = clac_edgepoints(leftpoints, 200, image.shape[0]) # 要畫左右車道線的端點
    righttop = clac_edgepoints(rightpoints, 200, image.shape[0])

    src = np.zeros_like(image)

    cv2.line(src, lefttop[0], lefttop[1], (0, 255, 0), 7)
    cv2.line(src, righttop[0], righttop[1], (0, 255, 0), 7)

    cv2.imshow('line Image', src)
    src_2 = cv2.addWeighted(image, 0.8, src, 1, 0)
    cv2.imshow('Finally Image', src_2)

    cv2.waitKey(0)