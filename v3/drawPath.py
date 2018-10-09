import cv2
from createBlank import create_blank

def drawPath(frameName, ps, scale):
        pts = []
        for i in range(len(ps)):
                pt = ps[i]
                pts += [[int(pt[0] * scale), int(pt[1] * scale)]]

        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for pt in pts:
                if pt[0] < min_x:
                        min_x = pt[0]
                if pt[1] < min_y:
                        min_y = pt[1]
                if pt[0] > max_x:
                        max_x = pt[0]
                if pt[1] > max_y:       
                        max_y = pt[1]

        for index in range(len(pts)):
                pts[index][0] = pts[index][0] - min_x
                pts[index][1] = pts[index][1] - min_y
                pts[index] = tuple(pts[index])

        max_x = max_x - min_x
        max_y = max_y - min_y

        #print("max_x: {0}, max_y {1}".format(max_x, max_y))
        if max_x <= 1 or max_y <= 1:
                return

        frame = create_blank(max_x, max_y)
        # print(pts)

        #pts = [(int(pt[0] * 3.0) - 1500, int(pt[1] * 3.0) - 1500) for pt in ps]
        for index in range(len(pts) - 1):
                cv2.line(frame, pts[index], pts[index + 1], (255, 255, 255))

        cv2.imshow(frameName, frame)
