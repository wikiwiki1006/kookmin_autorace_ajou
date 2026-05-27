cmake_minimum_required(VERSION 3.8)
project(track_drive)

# ========================================================
# 1. 의존성 패키지 찾기 (C++ 관련 패키지 전부 제거)
# ========================================================
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED) # 파이썬 설치를 위해 필수

# ========================================================
# 2. 설치 (Install) 규칙 정의
# ========================================================

# ① 파이썬 라이브러리 폴더(track_drive/) 전체 인식
ament_python_install_package(${PROJECT_NAME})

# ② ros2 run으로 직접 실행할 메인 파이썬 스크립트들 지정
install(PROGRAMS
  track_drive/track_drive.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========================================================
# 3. 테스트 및 패키징 마무리
# ========================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()