import kss
import re

# 예약된 명령어 정의
COMMAND_KEYWORDS = [
    '안녕',
    '놀아줘',
    '앞으로 가',
    '뒤로 가',
    '너 별로야',
    '잘자',
    '돌아봐',
    '춤춰줘',
    '최고야'
] # etc.


def text_preprocessing(text):
    # 문장 단위 분리
    sentences = kss.split_sentences(text)

    # 결과 추출
    reservation_commands = [s for s in sentences if is_reservation_command(s)]

    print("예약 관련 문장: ", reservation_commands)


# 예약된 명령어 추출
def is_reservation_command(sentence):
    """
        문장에서 구두점을 제거한 뒤, COMMAND_KEYWORDS 목록에 정확히 일치하는지 검사.
        """
    # 문장 중간의 마침표, 쉼표, 느낌표, 물음표 등 제거
    cleaned = clean_text(sentence).strip()

    # 예약어 목록에 정확히 일치하면 True 반환
    return cleaned in COMMAND_KEYWORDS


# 문장 종결 부호 제거
# 더 깔끔한 단어 검색을 위해서
def clean_text(text):
    return re.sub(r'[,.!?]', '', text)


def has_command(text):
    sentences = kss.split_sentences(text)
    for s in sentences:
        if is_reservation_command(s):
            # 감지된 예약어 문장(s)도 같이 반환
            return True, s
    return False, None

