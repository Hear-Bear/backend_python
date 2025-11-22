import os
import librosa
import soundfile as sf
from tqdm import tqdm

# 1. 설정
TARGET_SR = 16000  # 모델의 샘플링 속도
#
# ‼️ config.py에 설정한 실제 경로를 여기에 입력하세요
INPUT_NOISE_DIR = r"D:\dataset\augmentation\background_noise"
INPUT_IR_DIR = r"D:\dataset\augmentation\impulse_response"

# 2. 변환 후 저장할 새 폴더 (이름 뒤에 _16k를 붙임)
OUTPUT_NOISE_DIR = INPUT_NOISE_DIR + "_16k"
OUTPUT_IR_DIR = INPUT_IR_DIR + "_16k"


def convert_folder(input_dir, output_dir, target_sr):
    """
    폴더 내의 모든 .wav 파일을 읽어 target_sr로 리샘플링한 후
    새 폴더에 동일한 파일명으로 저장합니다.
    """
    if not os.path.exists(input_dir):
        print(f"경고: 원본 폴더를 찾을 수 없습니다: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]

    print(f"--- '{os.path.basename(input_dir)}' 폴더 변환 시작 ---")

    for filename in tqdm(file_list, desc=f"Converting {os.path.basename(output_dir)}"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            # 1. 파일 로드 및 리샘플링
            wav, sr = librosa.load(input_path, sr=target_sr, mono=True)

            # 2. 새 폴더에 저장
            sf.write(output_path, wav, target_sr)

        except Exception as e:
            print(f"오류: {filename} 처리 중 문제 발생: {e}")

    print(f"--- 변환 완료: {output_dir} ---")


# --- 스크립트 실행 ---
if __name__ == "__main__":
    convert_folder(INPUT_NOISE_DIR, OUTPUT_NOISE_DIR, TARGET_SR)
    convert_folder(INPUT_IR_DIR, OUTPUT_IR_DIR, TARGET_SR)