import cv2
import numpy as np
import os
from tqdm import tqdm


if __name__ == '__main__':
    # 이미지 파일들이 있는 디렉토리 경로 설정
    image_dir = "data/train_target_image"  # 이미지 파일들이 있는 디렉토리 경로를 설정하세요.
    output_dir = "data/"  # 결과 이미지를 저장할 디렉토리 경로를 설정하세요.

    # 이미지 파일 목록 가져오기
    image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(".png")]
    np.random.shuffle(image_files)
    print(len(image_files))

    # 초기값 설정
    total_mean = None
    total_variance = None
    n = 0

    # 이미지 파일들을 하나씩 처리
    for idx, image_file in tqdm(enumerate(image_files), desc="이미지 처리 진행", total=len(image_files)):

        # 이미지 로드
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.GaussianBlur(image, (0, 0), 3)
        image = cv2.fastNlMeansDenoising(image, None, 15, 7, 21)
        image = image / 255.0
        # 이미지의 높이와 너비 가져오기
        height, width = image.shape[:2]
        
        # 픽셀 값 배열로 변환
        pixel_values = image.reshape((height, width, 1))
        
        # 현재 이미지의 평균 계산
        mean_image = np.mean(pixel_values, axis=-1)
        
        # 현재 이미지의 분산 계산 (Welford's online algorithm 사용)
        n += 1
        if total_mean is None:
            total_mean = mean_image
            total_variance = np.zeros_like(mean_image)
        else:
            delta = mean_image - total_mean
            total_mean += delta / n
            delta2 = mean_image - total_mean
            total_variance += delta * delta2

        # 100장 랜덤 샘플링
        # if idx == (100 - 1):
        #     break
    
    # 분산을 표준편차로 변환
    total_stddev = np.sqrt(total_variance / n)

    total_mean = total_mean * 255.0
    total_stddev = total_stddev * 255.0

    stddev_image = total_stddev.astype(np.uint8)
    stddev_image = cv2.fastNlMeansDenoising(stddev_image, None, 15, 7, 21)

    # 표준편차 임계값 설정 (임계값 이하의 std를 가진 픽셀은 0으로 만듦)
    threshold_stds = [5, 10, 15, 18, 20]  # 원하는 임계값을 설정하세요.
    for threshold_std in threshold_stds:

        stddev_image_new = stddev_image.copy()
        # 임계값 이하의 std를 가진 픽셀을 0으로 만듦
        stddev_image_new[stddev_image_new > threshold_std] = 255
        stddev_image_new[stddev_image_new <= threshold_std] = 0
        height, width = stddev_image_new.shape[:2]

        # Bottom Masking
        print(int(height * 0.85))
        print(int(width*0.2), int(width*0.8))
        left=int(width*0.2)
        right = int(width*0.8)
        stddev_image_new[int(height * 0.90):, :left] = 0
        stddev_image_new[int(height * 0.90):, right:] = 0

        cv2.imwrite(f"{output_dir}filtered_stddev_image_{threshold_std}.png", stddev_image_new.astype(np.uint8))

    print("DONE !!")