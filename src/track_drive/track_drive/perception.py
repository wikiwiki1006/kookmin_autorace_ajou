import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32MultiArray, Int32 # Int32 추가
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.bridge = CvBridge()
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 1. 공통 카메라 구독
        self.image_sub = self.create_subscription(Image, '/usb_cam/image_raw/front', self.image_callback, qos_profile)
        
        # 2. 차선 인식 관련 퍼블리셔 유지
        self.marker_pub = self.create_publisher(MarkerArray, '/lane_markers', qos_profile)
        self.bev_pub = self.create_publisher(Image, '/lane_bev_debug', qos_profile)
        self.lane_left_pub = self.create_publisher(Float32MultiArray, '/lane_left', 10)
        self.lane_center_pub = self.create_publisher(Float32MultiArray, '/lane_center', 10)
        self.lane_right_pub = self.create_publisher(Float32MultiArray, '/lane_right', 10)
        self.intersection_pub = self.create_publisher(Bool, '/is_intersection', 10)
        self.stopline_pub = self.create_publisher(Bool, '/is_stopline', 10)
        self.safetyzone_pub = self.create_publisher(Bool, '/is_safetyzone', 10)
        
        # 3. 신호등 인식 관련 퍼블리셔 추가
        self.tl_image_pub = self.create_publisher(Image, '/traffic_light/annotated_image', 10)
        self.traffic_pub = self.create_publisher(Int32, '/traffic', 10) # 신호등 상태 퍼블리셔
        
        # 공통 해상도 설정
        self.img_width = 640
        self.img_height = 480
        
        # [신호등 설정] ROI 세로 한계점 (480 * 0.35 = 168)
        self.tl_roi_bottom = int(self.img_height * 0.35)
        
        # [차선 설정] BEV 변환 매트릭스
        self.src_pts = np.float32([[230, 260], [410, 260], [0, 480], [640, 480]]) 
        self.dst_pts = np.float32([[140, 30], [self.img_width-140, 30], [210, self.img_height], [self.img_width-210, self.img_height]])
        self.matrix = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        
        self.ym_per_pix = 0.0094
        self.xm_per_pix = 0.0094
        
        self.is_special_mode = False
        self.intersection_count = 0

        self.get_logger().info("통합 인지 노드 시작")

    def get_real_coords(self, pts_x, pts_y):
        real_x = (self.img_height - pts_y) * self.ym_per_pix
        real_y = (self.img_width // 2 - pts_x) * self.xm_per_pix
        return real_x, real_y

    def create_line_marker(self, m_id, plot_x, plot_y, r, g, b, a=1.0, stamp=None):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = stamp if stamp else self.get_clock().now().to_msg()
        marker.ns = "lane_lines"
        marker.id = m_id
        
        if plot_x is None or len(plot_x) == 0:
            marker.action = Marker.DELETE
            return marker
            
        marker.action = Marker.ADD
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 0.15 
        marker.color.r = float(r); marker.color.g = float(g); marker.color.b = float(b); marker.color.a = float(a)
        
        for x, y in zip(plot_x, plot_y):
            p = Point()
            p.x = float(x); p.y = float(y); p.z = 0.0
            marker.points.append(p)
            
        return marker

    def draw_real_pts_on_bev(self, img, real_x, real_y, color):
        if real_x is None or len(real_x) == 0: return
        pts_y = self.img_height - (np.array(real_x) / self.ym_per_pix)
        pts_x = (self.img_width // 2) - (np.array(real_y) / self.xm_per_pix)
        pts = np.array([np.transpose(np.vstack([pts_x, pts_y]))], np.int32)
        cv2.polylines(img, pts, isClosed=False, color=color, thickness=4)

    def extract_valid_pixels(self, mask, min_area=100):
        clean_mask = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
        return np.nonzero(clean_mask)

    def image_callback(self, msg):
        img_stamp = msg.header.stamp
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # =========================================================
        # 🚦 1. 신호등 인식 로직 (이미지 상단 35%)
        # =========================================================
        tl_annotated_img = cv_img.copy()
        roi_img = tl_annotated_img[0:self.tl_roi_bottom, 0:self.img_width]
        
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 70])
        black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
        
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        state_str = "UNKNOWN"
        traffic_state_num = -1 # 미인식 시 -1 (기본값)
        box_color = (255, 255, 255)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # 크기 상한선/하한선 필터 적용 (예: 500 ~ 10000)
            if 500 <= area:
                x, y, w, h = cv2.boundingRect(largest_contour)
                traffic_light_roi = hsv_roi[y:y+h, x:x+w]
                
                # 색상 추출
                lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
                lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
                mask_red = cv2.bitwise_or(cv2.inRange(traffic_light_roi, lower_red1, upper_red1), 
                                          cv2.inRange(traffic_light_roi, lower_red2, upper_red2))
                
                mask_green = cv2.inRange(traffic_light_roi, np.array([40, 100, 100]), np.array([90, 255, 255]))
                mask_yellow = cv2.inRange(traffic_light_roi, np.array([15, 100, 100]), np.array([35, 255, 255]))
                
                r_pix = cv2.countNonZero(mask_red)
                g_pix = cv2.countNonZero(mask_green)
                y_pix = cv2.countNonZero(mask_yellow)
                threshold_pixels = 500 # 픽셀 임계치 조정
                
                # 상태 판별 및 Int32 매핑
                if r_pix > threshold_pixels and g_pix > threshold_pixels:
                    state_str, traffic_state_num, box_color = "LEFT", 3, (255, 0, 255)
                elif g_pix > threshold_pixels:
                    state_str, traffic_state_num, box_color = "GREEN", 1, (0, 255, 0)
                elif r_pix > threshold_pixels:
                    state_str, traffic_state_num, box_color = "RED", 0, (0, 0, 255)
                elif y_pix > threshold_pixels:
                    state_str, traffic_state_num, box_color = "YELLOW", 2, (0, 255, 255)
                    
                # 신호등 시각화 박스 그리기
                cv2.rectangle(tl_annotated_img, (x, y), (x+w, y+h), box_color, 3)
                cv2.putText(tl_annotated_img, state_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # 신호등 메시지 퍼블리시
        tl_msg = Int32()
        tl_msg.data = traffic_state_num
        self.traffic_pub.publish(tl_msg)
        
        annotated_msg = self.bridge.cv2_to_imgmsg(tl_annotated_img, encoding="bgr8")
        self.tl_image_pub.publish(annotated_msg)

        # =========================================================
        # 🛣️ 2. 차선 인식 로직 (BEV 변환)
        # =========================================================
        bev_img = cv2.warpPerspective(cv_img, self.matrix, (self.img_width, self.img_height))
        debug_img = bev_img.copy()
        hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
        
        # 안전 도로 영역 추출
        black_mask = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([360, 30, 80]))
        road_mask = cv2.dilate(black_mask, np.ones((21, 21), np.uint8), iterations=2)
        dist_transform = cv2.distanceTransform(road_mask, cv2.DIST_L2, 5)
        _, safe_road_mask = cv2.threshold(dist_transform, 0.5 / self.xm_per_pix, 255, cv2.THRESH_BINARY)
        safe_road_mask = np.uint8(safe_road_mask)
        _, road_center_mask = cv2.threshold(dist_transform, np.max(dist_transform) * 0.5, 255, cv2.THRESH_BINARY)
        road_center_mask = np.uint8(road_center_mask)

        # 차선 마스크 추출
        yellow_mask_all = cv2.bitwise_and(cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255])), safe_road_mask)
        raw_white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        white_mask_all = cv2.bitwise_and(raw_white_mask, safe_road_mask)
        
        y_y_all, y_x_all = self.extract_valid_pixels(yellow_mask_all, min_area=50)
        w_y_all, w_x_all = self.extract_valid_pixels(white_mask_all, min_area=50)

        # 교차로 인식
        cnts_yellow, _ = cv2.findContours(yellow_mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_horizontal_dash_detected = False
        
        for cnt in cnts_yellow:
            area = cv2.contourArea(cnt)
            if 200 < area < 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / max(h, 1)
                if aspect_ratio > 2.5:
                    is_horizontal_dash_detected = True
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 255), 3)
                    break 

        if is_horizontal_dash_detected:
            self.intersection_count += 1
        else:
            self.intersection_count = 0 
            
        is_intersection_detected = self.intersection_count >= 5
        self.intersection_count = min(self.intersection_count, 5)

        # 정지선 인식
        bottom_30_start_y = int(self.img_height * 0.7)
        margin_x = 180
        roi_white_mask = raw_white_mask[bottom_30_start_y:self.img_height, margin_x:self.img_width-margin_x]
        
        is_stop_line_detected = False
        if is_intersection_detected and np.count_nonzero(roi_white_mask) >= 10000:
            is_stop_line_detected = True
            cv2.rectangle(debug_img, (margin_x, bottom_30_start_y), (self.img_width-margin_x, self.img_height), (0, 0, 255), -1)

        # 상태 머신 처리
        num_white, num_yellow = len(w_y_all), len(y_y_all)
        if num_white > 5000 and num_yellow < 10000:
            self.is_special_mode = False
        elif num_white < 1000 and num_yellow > 20000:
            if np.count_nonzero(cv2.bitwise_and(yellow_mask_all, cv2.bitwise_not(road_center_mask))) > 100:
                self.is_special_mode = True

        if not self.is_special_mode:
            real_c_x, real_c_y = self.get_real_coords(y_x_all, y_y_all)
            white_y_img, white_x_img = w_y_all, w_x_all
            real_w_x, real_w_y = self.get_real_coords(white_x_img, white_y_img)
            debug_img[y_y_all, y_x_all] = [0, 255, 255] 
        else:
            c_y, c_x = self.extract_valid_pixels(cv2.bitwise_and(yellow_mask_all, road_center_mask), min_area=50)
            real_c_x, real_c_y = self.get_real_coords(c_x, c_y)
            white_y_img, white_x_img = self.extract_valid_pixels(cv2.bitwise_and(yellow_mask_all, cv2.bitwise_not(road_center_mask)), min_area=50)
            real_w_x, real_w_y = self.get_real_coords(white_x_img, white_y_img)
            debug_img[c_y, c_x] = [0, 255, 255]

        valid_lines = {} 
        if len(real_c_x) > 100: valid_lines['center'] = np.polyfit(real_c_x, real_c_y, 2)

        if len(real_w_x) > 0:
            if 'center' in valid_lines:
                c_coeffs = valid_lines['center']
                expected_y = c_coeffs[0]*(real_w_x**2) + c_coeffs[1]*real_w_x + c_coeffs[2]
                left_idx = real_w_y > expected_y   
                right_idx = real_w_y < expected_y  
            else:
                left_idx, right_idx = real_w_y > 0, real_w_y < 0
            
            real_l_x, real_l_y = real_w_x[left_idx], real_w_y[left_idx]
            real_r_x, real_r_y = real_w_x[right_idx], real_w_y[right_idx]

            if len(real_l_x) > 100: valid_lines['left'] = np.polyfit(real_l_x, real_l_y, 2)
            if len(real_r_x) > 100: valid_lines['right'] = np.polyfit(real_r_x, real_r_y, 2)

            debug_img[white_y_img[left_idx], white_x_img[left_idx]] = [0, 0, 255]
            debug_img[white_y_img[right_idx], white_x_img[right_idx]] = [0, 255, 0]

        # 궤적 및 메시지 생성
        marker_array = MarkerArray()
        msg_left, msg_center, msg_right = Float32MultiArray(), Float32MultiArray(), Float32MultiArray()
        
        if valid_lines:
            best_line_type = min(valid_lines.keys(), key=lambda k: abs(valid_lines[k][0]))
            best_coeffs = valid_lines[best_line_type]
            
            if best_line_type == 'center': off_L, off_C, off_R = 1.5, 0.0, -1.5
            elif best_line_type == 'left': off_L, off_C, off_R = 0.0, -1.5, -3.0
            elif best_line_type == 'right': off_L, off_C, off_R = 3.0, 1.5, 0.0
            
            plot_real_x = np.linspace(0.0, 15.0, 30)
            plot_y_base = best_coeffs[0]*(plot_real_x**2) + best_coeffs[1]*plot_real_x + best_coeffs[2]
            pts_L_x, pts_L_y, pts_C_x, pts_C_y, pts_R_x, pts_R_y = [], [], [], [], [], []
            data_L, data_C, data_R = [], [], []
            
            for x, y in zip(plot_real_x, plot_y_base):
                dy_dx = 2*best_coeffs[0]*x + best_coeffs[1]
                N = np.array([-dy_dx, 1.0])
                norm = np.linalg.norm(N); Unit_N = N / norm if norm != 0 else np.array([0.0, 1.0])
                
                pts_L_x.append(x + off_L * Unit_N[0]); pts_L_y.append(y + off_L * Unit_N[1])
                pts_C_x.append(x + off_C * Unit_N[0]); pts_C_y.append(y + off_C * Unit_N[1])
                pts_R_x.append(x + off_R * Unit_N[0]); pts_R_y.append(y + off_R * Unit_N[1])

            marker_array.markers.append(self.create_line_marker(11, pts_C_x, pts_C_y, 1.0, 1.0, 0.0, stamp=img_stamp))
            marker_array.markers.append(self.create_line_marker(21, pts_L_x, pts_L_y, 1.0, 0.0, 0.0, stamp=img_stamp))
            marker_array.markers.append(self.create_line_marker(31, pts_R_x, pts_R_y, 0.0, 1.0, 0.0, stamp=img_stamp))
            
            for x, y in zip(pts_L_x, pts_L_y): data_L.extend([float(x), float(y)])
            for x, y in zip(pts_C_x, pts_C_y): data_C.extend([float(x), float(y)])
            for x, y in zip(pts_R_x, pts_R_y): data_R.extend([float(x), float(y)])
            
            msg_left.data, msg_center.data, msg_right.data = data_L, data_C, data_R
            self.draw_real_pts_on_bev(debug_img, pts_C_x, pts_C_y, (0, 255, 255))
            self.draw_real_pts_on_bev(debug_img, pts_L_x, pts_L_y, (0, 0, 255))
            self.draw_real_pts_on_bev(debug_img, pts_R_x, pts_R_y, (0, 255, 0))

        else:
            marker_array.markers.extend([
                self.create_line_marker(11, None, None, 0,0,0, stamp=img_stamp),
                self.create_line_marker(21, None, None, 0,0,0, stamp=img_stamp),
                self.create_line_marker(31, None, None, 0,0,0, stamp=img_stamp)
            ])
            msg_left.data, msg_center.data, msg_right.data = [], [], []

        # 차선 퍼블리시
        self.marker_pub.publish(marker_array)
        bev_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        bev_msg.header.stamp = img_stamp; bev_msg.header.frame_id = "base_link"
        self.bev_pub.publish(bev_msg)

        intersect_msg, stopline_msg, safetyzone_msg = Bool(), Bool(), Bool()
        intersect_msg.data, stopline_msg.data, safetyzone_msg.data = bool(is_intersection_detected), bool(is_stop_line_detected), bool(self.is_special_mode)
        
        self.intersection_pub.publish(intersect_msg)
        self.stopline_pub.publish(stopline_msg)
        self.safetyzone_pub.publish(safetyzone_msg)

        self.lane_left_pub.publish(msg_left)
        self.lane_center_pub.publish(msg_center)
        self.lane_right_pub.publish(msg_right)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()