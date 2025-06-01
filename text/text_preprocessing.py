import kss
from konlpy.tag import Okt
import re

# 형태소 분석기 초기화
okt = Okt()

# 예약된 명령어의 어근 형태(기본형)로 정의
COMMAND_KEYWORDS = ['앉다', '일어나다', '오다'] # etc.
# 앉아, 일어나, 이리와

# 이리와 같은 복합어 등, 분석기가 제대로 검색하지 못하는 예외 단어들을 정의
SPECIALS_CASES = {
    "이리와": "오다"
}


def text_preprocessing(text):
    # 문장 단위 분리
    sentences = kss.split_sentences(text)

    # 결과 추출
    reservation_commands = [s for s in sentences if is_reservation_command(clean_text(s))]

    print("예약 관련 문장: ", reservation_commands)


# 예약된 명령어 추출
def is_reservation_command(sentence):
    if any(word in sentence for word in SPECIALS_CASES.keys()):
        return True

    # 형태소 + 품사 분석, 어근 추출을 위해 stem=True
    pos_tags = okt.pos(sentence, stem=True)
    # print(pos_tags)

    # 동사/형용사만 추출하여 기본형 체크
    base_verbs = [word for word, tag in pos_tags if tag in ['Verb', 'Adjective']]

    # print(f"[DEBUG] 문장: {sentence}")
    # print(f"[DEBUG] 어근 추출: {base_verbs}")

    # 예: '오다'가 들어 있으면 '이리 와', '와줘', '와라' 등 감지 가능
    return any(keyword in base_verbs for keyword in COMMAND_KEYWORDS)


# 문장 종결 부호 제거
# 더 깔끔한 단어 검색을 위해서
def clean_text(text):
    return re.sub(r'[,.!?]', '', text)


# 예제
text_preprocessing("어디 있어? 이리와!")
text_preprocessing("오늘 날씨가 참 좋네. 오늘은 밖에 나가야 겠어.")
text_preprocessing("히어베어, 앉아! 손 줘. 여기서 같이 놀자.")