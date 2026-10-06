import numpy as np
import cv2

from scipy import interpolate

def spline_line(img, points, color, thickness_line, is_closed = True):
    work_points = points.copy()
    if is_closed:
        work_points.append(work_points[0])

    x, y = zip(*work_points)
    tck,u = interpolate.splprep([x, y], s=0, per = is_closed)
    unew = np.linspace(0, 1, 1000)
    out = interpolate.splev(unew, tck)

    out_list = np.array([[int(round(out[0][i])), int(round(out[1][i]))] for i in range(len(out[0]))])

    img = cv2.polylines(img, [out_list], isClosed = is_closed, color = color, thickness = thickness_line, lineType = cv2.LINE_AA)

def small_spline_line(img, points, color, thickness_line):
    work_points = points.copy()
    #print(work_points)
    K = 3
    if len(work_points) == 3:
        K = 2
    if len(work_points) == 2:
        K = 1
    x, y = zip(*work_points)
    tck,u = interpolate.splprep([x, y], s = 0, k = K)
    
    unew = np.linspace(0, 1, 100)
    out = interpolate.splev(unew, tck)

    out_list = np.array([[int(round(out[0][i])), int(round(out[1][i]))] for i in range(len(out[0]))])

    img = cv2.polylines(img, [out_list], isClosed = False, color = color, thickness = thickness_line, lineType = cv2.LINE_AA)


def fill_full_spline(img, points, color, is_closed = True):
    work_points = points.copy()
    if is_closed:
        work_points.append(work_points[0])

    x, y = zip(*work_points)
    tck,u = interpolate.splprep([x, y], s=0, per = is_closed)
    unew = np.linspace(0, 1, 1000)
    out = interpolate.splev(unew, tck)

    out_list = np.array([[int(round(out[0][i])), int(round(out[1][i]))] for i in range(len(out[0]))])

    #img = cv2.polylines(img, [out_list], isClosed = is_closed, color = color, thickness = thickness_line, lineType = cv2.LINE_AA)
    img = cv2.drawContours(img, [out_list], -1, color, -1)

def fill_texture_spline(img, points, texture, is_closed = True):
    work_points = points.copy()
    if is_closed:
        work_points.append(work_points[0])

    x, y = zip(*work_points)
    tck,u = interpolate.splprep([x, y], s=0, per = is_closed)
    unew = np.linspace(0, 1, 1000)
    out = interpolate.splev(unew, tck)

    out_list = np.array([[int(round(out[0][i])), int(round(out[1][i]))] for i in range(len(out[0]))])

    #img = cv2.polylines(img, [out_list], isClosed = is_closed, color = color, thickness = thickness_line, lineType = cv2.LINE_AA)
    mask = np.zeros(img.shape[0:2], np.uint8)
    mask = cv2.drawContours(mask, [out_list], -1, 255, -1)

    img[mask[:,:] == 255] = texture[mask[:,:] == 255]

def fill_full_ellipse(img, center, radius, color, angle = 0, startAngle = 0, endAngle = 360):
    cv2.ellipse(img = img,
                center = center,
                axes = radius,
                color = color,
                thickness = -1,
                angle = angle,
                startAngle = startAngle,
                endAngle = endAngle)

def fill_texture_ellipse(img, center, radius, texture, angle = 0, startAngle = 0, endAngle = 360):
    #img = cv2.polylines(img, [out_list], isClosed = is_closed, color = color, thickness = thickness_line, lineType = cv2.LINE_AA)
    mask = np.zeros(img.shape[0:2], np.uint8)
    cv2.ellipse(img = mask,
               center = center,
               axes = radius,
               color = 255,
               thickness = -1,
               angle = angle,
               startAngle = startAngle,
               endAngle = endAngle)
    img[mask[:,:] == 255] = texture[mask[:,:] == 255]

def fill_texture_2_poligons(img, poligon1, poligon2, color, thickness_line):
    work_points = poligon1.copy()
    x, y = zip(*work_points)
    tck,u = interpolate.splprep([x, y], s=0, k=2, per = False)
    unew = np.linspace(0, 1, 100)
    out1 = interpolate.splev(unew, tck)
    out_list1 = [[int(round(out1[0][i])), int(round(out1[1][i]))] for i in range(len(out1[0]))]

    work_points = reversed(poligon2.copy())
    x, y = zip(*work_points)
    tck,u = interpolate.splprep([x, y], s=0, k=2, per = False)
    unew = np.linspace(0, 1, 100)
    out2 = interpolate.splev(unew, tck)

    out_list2 = [[int(round(out2[0][i])), int(round(out2[1][i]))] for i in range(len(out2[0]))]

    out_list = np.array(out_list1 + out_list2)
    #img = cv2.polylines(img, [out_list], isClosed = is_closed, color = color, thickness = thickness_line, lineType = cv2.LINE_AA)
    img = cv2.drawContours(img, [out_list], -1, color, -1)
    img = cv2.polylines(img, [out_list], isClosed = True, color = color, thickness = thickness_line, lineType = cv2.LINE_AA)

if __name__ == "__main__":
    img = np.zeros((512,512,3), np.uint8)

    points = [[100, 100], [380, 100], [380, 380], [100, 380], [255,255]]

    #print(points)

    spline_line(img, points, (0, 255, 255), 3)

    for i in range(len(points)):
        (x,y) = points[i]
        img = cv2.circle(img, (x,y), 4, (0,0,255), 3)

    cv2.imshow("img", img)

    img2 = np.zeros((512,512,3), np.uint8)

    fill_full_spline(img2, points, (255, 0, 255))

    for i in range(len(points)):
        (x,y) = points[i]
        img2 = cv2.circle(img2, (x,y), 4, (0,0,255), 3)

    cv2.imshow("img2", img2)


    img3 = np.zeros((512,512,3), np.uint8)

    texture = np.zeros((512,512,3), np.uint8)
    for y in range(512):
        colorsda = np.random.randint(0, 256, 3)
        for x in range(512):
            texture[y,x] = colorsda

    fill_texture_spline(img3, points, texture)

    for i in range(len(points)):
        (x,y) = points[i]
        img3 = cv2.circle(img3, (x,y), 4, (0,0,255), 3)

    cv2.imshow("img3", img3)


    img4 = np.zeros((512,512,3), np.uint8)

    fill_full_ellipse(img4, (255,255), (128,128) , (0,0,255))

    cv2.imshow("img4", img4)


    img5 = np.zeros((512,512,3), np.uint8)

    fill_texture_ellipse(img5, (255,255), (128,64), texture)

    cv2.imshow("img5", img5)
    cv2.waitKey()
