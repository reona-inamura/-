import csv
import cv2
import math
import numpy as np
import time

CONTOUR_AREA_THRESHOLD = 400

def prepare_camera(camera_id, fps, width, height):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Camera {camera_id} could not be opened.")
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == width, "Width setting failed."
    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == height, "Height setting failed."

    return cap

class Buffer():
    def __init__(self, keys=['u', 'v', 'r']):
        self.data = list()
        self.keys = keys
        self.start_time = time.time()

    def append(self, **kwargs):
        # 時刻も保存
        t = 1000 * (time.time() - self.start_time)
        item = [t] + [kwargs[key] for key in self.keys]
        self.data.append(item)

    def dump(self, csv_path='logCDhome.csv'):
        header = ['t'] + self.keys
        with open(csv_path, 'w') as f:
            writer = csv.writer(f, lineterminator = "\n")
            writer.writerow(header)
            writer.writerows(self.data)


def detect_marker(contours):
    # 画像モーメントをもとに、マーカーの座標を計算
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        # 入力画像のモーメント
        mu = cv2.moments(largest_contour)
        # 面積
        s = cv2.contourArea(largest_contour)
        if mu["m00"] > 0 and s > CONTOUR_AREA_THRESHOLD:
            # モーメントからu,v座標を計算
            u, v = int(mu["m10"] / mu["m00"]) , int(mu["m01"] / mu["m00"])
            r_dot = round(math.sqrt(s/math.pi), 2) #半径
            return u, v, s, r_dot
    return 0, 0, 0.0, 0.0

def main_loop(cap, threshold, params):
    IMAGE_NAMES = ['Original image', 'Gray image', 'Binary image']
    image_mode = 0
    CAMERA_WIDTH, CAMERA_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    recording = False
    linefrag = 0
    linespace = 30
    marker_frag = 0
    while True:
        # キーボード入力を受付
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 終了
            break
        elif key == ord('t'):
            # 表示形式変更 (カラー/グレースケール/2値)
            cv2.destroyWindow(IMAGE_NAMES[image_mode])
            image_mode = (image_mode + 1) % len(IMAGE_NAMES)
        elif key == ord('s'):
            if not recording:
                print('start recording')
                # バッファーを初期化
                buffer = Buffer(['u', 'v', 'r', 'X', 'Y', 'Z'])
                recording = True
            else:
                print('saving finished')
                # バッファーの内容を保存
                buffer.dump()
                recording = False
        elif key == ord('u'):
            if threshold is None:
                threshold = 128
            #thresholdを増やす
            else:
                threshold += 1
                print(threshold)
        elif key == ord('d'):
            if threshold is None:
                threshold = 128
            #thresholdを減らす
            else:
            
                threshold -= 1
                print(threshold)
        elif key ==ord('o'):
            if threshold is not None:
                threshold = None
        elif key ==ord('l'):
            if linefrag == 0:
                linefrag = 1
            else:
                linefrag = 0
        elif key ==ord('w'):
            linespace +=1
        elif key ==ord('n'):
            linespace -=1
            if linespace < 1:
                linespace = 1
        elif key ==ord('m'):
            marker_x = []
            marker_y = []
            if marker_frag == 0:
                marker_frag = 1
            else:
                marker_frag = 0
        
        # TODO (他のキー操作)
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if threshold is not None:
            # thresholdにもとづき二値化(手動)
            _, im_th = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    
        else:
            # 大津の二値化を使う場合(自動)
            th, im_th = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_OTSU)
            print(th)
        
        binary_frame = cv2.bitwise_not(im_th)
        # 輪郭検出
        contours, hierarchy = cv2.findContours(binary_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 座標計算
        u, v, s, r_dot = detect_marker(contours)
            
        if marker_frag == 1:    
            marker_x.append(u)
            marker_y.append(v)

        # TODO (x, y, zの計算）
        L_kyori = 222
        D_x = 293
        D_y = 227
        #theta = 
        #phi = 
        f_x = 320*(L_kyori/(D_x/2))
        f_y = 240*(L_kyori/(D_y/2))
        #x=(u-320)*((D_x/2)/320)#課題C
        #y=(-v+240)*((D_y/2)/240)#課題C
        #z=L_kyori#課題ｃ
        r_dot0 = 35.78
        r_0hankei = 10
        L0 = 130
        f_ = r_dot0 * L0/ r_0hankei
        #z = f_ * r_0hankei/r_dot#
        
        z=(f_x+f_y)/2*r_0hankei/r_dot#D-7
        x=(z/f_x)*(u-320)
        y=(z/f_y)*(-v+240)
        print(len(contours))
        # 分母の０で，プログラムがフリーズになることに注意
        L, fx, fy, f = params['L'], params['fx'], params['fy'], params['f'],
        #
        
            
        if recording:
            # 時刻や座標をバッファに保存
            buffer.append(u=u, v=v, r = r_dot, X = x, Y = y, Z = z)#rはピクセル
            pass
        
        if image_mode == 0:
            show_img = frame
            point_color = (0, 200, 0)
        elif image_mode == 1:
            show_img = gray_frame
            point_color = 255
        elif image_mode == 2:
            show_img = binary_frame
            point_color = 128 
        
        # draw line
        #cv2.line(show_img,(0,20),(640,20),(255,0,0),1)
        #cv2.line(show_img,(60,0),(60,320),(255,0,0),1)
        if (linefrag == 1):
            tate = 0
            while(linespace * tate < 640):
                cv2.line(show_img,(linespace*tate,0),(linespace*tate,480),(255,0,0),1)
                tate+=1
            yoko = 0
            while(linespace * yoko < 480):
                cv2.line(show_img,(0,linespace*yoko),(640,linespace*yoko),(255,0,0),1)
                yoko+=1


        # show detection info
        cv2.circle(show_img, (u, v), 6, point_color, -1)#marker
        if marker_frag ==1:
            for i in range (0,len(marker_x)-1):
                #cv2.circle(show_img, (marker_x[i], marker_y[i]), 6, point_color, -1)
                cv2.line(show_img,(marker_x[i], marker_y[i]),(marker_x[i+1], marker_y[i+1]),(255,0,0),1)
        #ここで表示文字変えれそう
        if threshold is None:
            text = f'({u}, {v}), r: {r_dot}, num_contours: {len(contours)}, th: {th}'
        else:
            text = f'({u}, {v}), r: {r_dot}, num_contours: {len(contours)}, threshold: {threshold}'
        #text = f'({u}, {v}), r: {r_dot}, num_contours: {len(contours)}'
        
        text = f'({int(x)},{int(y)},{int(z)})'#Cの課題で使う
        cv2.putText(show_img, text, (u + 20, v+ 20),
                   cv2.FONT_HERSHEY_PLAIN, 1.0,
                   (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(IMAGE_NAMES[image_mode], show_img)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Marker detection using a camera.")
    parser.add_argument('--camera_id', type=int, default=0, help='Camera index')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--camera_width', type=int, default=640, help='Frame width')
    parser.add_argument('--camera_height', type=int, default=480, help='Frame height')
    parser.add_argument('--threshold', type=int, help='Thresohld for binarization')
    
    parser.add_argument('--L', type=float, help='L')
    parser.add_argument('--fx', type=float, help='fx')
    parser.add_argument('--fy', type=float, help='fy')
    parser.add_argument('--f', type=float, help='f')
    args = parser.parse_args()

    params = {
        'L': args.L,
        'fx': args.fx,
        'fy': args.fy,
        'f': args.f,
    }
    cap = prepare_camera(camera_id=args.camera_id, fps=args.fps, width=args.camera_width, height=args.camera_height)
    main_loop(cap, args.threshold, params)
    cap.release()
