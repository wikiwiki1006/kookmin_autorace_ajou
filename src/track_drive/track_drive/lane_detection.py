#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector_node')
        self.bridge = CvBridge()
        
        # 1. 구독(Subscribe) 및 발행(Publish) 설정
        self.image_sub = self.create_subscription(Image, '/usb_cam/image_raw/front', self.image_callback, 10)
        self.marker_pub = self.create_publisher(Marker, '/lane_markers', 10)
        
        # 2. BEV 해상도 및 변환 행렬 설정
        self.bev_width = 640
        self.bev_height = 480
        
        # 이전 분석을 통해 도출된 BEV 4점 좌표 
        self.src_pts = np.float32([[200, 300], [440, 300], [0, 480], [640, 480]])
        self.dst_pts = np.float32([[160, 0], [480, 0], [160, self.bev_height], [480, self.bev_height]])
        self.matrix = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        
        # 3. 픽셀 -> 미터 변환 비율 (임의의 예시값, 캘리브레이션 툴로 구한 값으로 수정하세요)
        self.ym_per_pix = 0.015  # 전방 y축 비율 (예: 1픽셀 = 1.5cm)
        self.xm_per_pix = 0.01   # 측면 x축 비율 (예: 1픽셀 = 1.0cm)

        self.get_logger().info("🚀 차선 인식 및 RViz 시각화 노드가 시작되었습니다.")

    def image_callback(self, msg):
        # [1단계] 이미지 변환 및 BEV 처리
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        bev_img = cv2.warpPerspective(cv_img, self.matrix, (self.bev_width, self.bev_height))
        
        # [2단계] 흰색 실선 추출 (명암 기반 이진화)
        gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
        # 200 이상 밝기(흰색)만 255로, 나머지는 0(검은색)으로 변환
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # [3단계] 좌우 차선 분리 및 픽셀 좌표 추출
        midpoint = self.bev_width // 2
        
        # 왼쪽 차선 (화면 왼쪽 절반)
        left_half = thresh.copy()
        left_half[:, midpoint:] = 0
        left_y, left_x = np.nonzero(left_half)
        
        # 오른쪽 차선 (화면 오른쪽 절반)
        right_half = thresh.copy()
        right_half[:, :midpoint] = 0
        right_y, right_x = np.nonzero(right_half)
        
        # [4단계] 다항식 피팅 및 RViz 마커 발행
        self.process_and_publish_lane(left_x, left_y, is_left=True)
        self.process_and_publish_lane(right_x, right_y, is_left=False)
        
        # (선택) 디버깅용 화면 출력
        cv2.imshow("BEV Threshold", thresh)
        cv2.waitKey(1)

    def process_and_publish_lane(self, pts_x, pts_y, is_left):
        """추출된 픽셀 좌표를 미터로 변환하고 2차 다항식 피팅 후 RViz로 보냅니다."""
        if len(pts_x) < 100:  # 인식된 차선 픽셀이 너무 적으면 무시 (노이즈 방지)
            return

        # 픽셀 좌표계를 차량 중심 물리적 좌표계(m)로 변환
        # 차량 기준 X축: 전방 (이미지상에서는 height에서 빼서 구함)
        # 차량 기준 Y축: 좌측 (이미지상에서는 width/2에서 빼서 구함)
        real_x = (self.bev_height - pts_y) * self.ym_per_pix
        real_y = (self.bev_width // 2 - pts_x) * self.xm_per_pix
        
        # 물리 좌표 기반 2차 다항식 피팅: y = a*x^2 + b*x + c
        poly_coeffs = np.polyfit(real_x, real_y, 2)
        
        # 다항식으로부터 RViz에 그릴 점 생성 (전방 0m ~ 10m 구간)
        plot_x = np.linspace(0.0, 10.0, 20)
        plot_y = poly_coeffs[0]*(plot_x**2) + poly_coeffs[1]*plot_x + poly_coeffs[2]
        
        # 마커 메시지 생성
        self.publish_marker(plot_x, plot_y, is_left)

    def publish_marker(self, plot_x, plot_y, is_left):
        marker = Marker()
        # RViz에서 기준이 될 TF 프레임 설정 (일반적으로 base_link 사용)
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lane_lines"
        marker.id = 0 if is_left else 1 # 좌우 차선을 ID로 구분
        
        marker.type = Marker.LINE_STRIP # 선 형태로 그리기
        marker.action = Marker.ADD
        
        # 선 굵기
        marker.scale.x = 0.1 
        
        # 선 색상 (왼쪽은 노란색, 오른쪽은 파란색으로 지정해 보았습니다)
        marker.color.a = 1.0 # 투명도
        if is_left:
            marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0 
        else:
            marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0
            
        # 다항식으로 계산된 x, y 좌표들을 Marker.points 배열에 추가
        for x, y in zip(plot_x, plot_y):
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0 # 평면 차선이므로 z는 0
            marker.points.append(p)
            
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()