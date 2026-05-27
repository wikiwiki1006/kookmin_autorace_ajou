cmake_minimum_required(VERSION 3.8)
project(track_drive)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ========================================================
# 1. 의존성 패키지 찾기
# ========================================================
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED) # 파이썬 설치를 위해 필수


# ========================================================
# 3. 설치 (Install) 규칙 정의
# ========================================================


# ② 파이썬 라이브러리 폴더(track_drive/) 전체 인식
# 이 명령어가 폴더 내의 __init__.py 및 import용 파일들을 통째로 묶어줍니다.
ament_python_install_package(${PROJECT_NAME})

# ③ ros2 run으로 직접 실행할 메인 파이썬 스크립트들 지정
install(PROGRAMS
  track_drive/track_drive.py
  track_drive/lane_detection.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()