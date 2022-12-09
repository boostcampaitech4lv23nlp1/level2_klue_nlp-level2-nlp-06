# KLUE-RE
NLP 6조 HAPPY팀이 만든 부스트 캠프에서 진행한 KLUE-RE 대회 코드입니다.
## 📝 Table of Contents

- [프로젝트 개요](#about)
- [팀 구성 및 역할](#team_member)
- [프로젝트 진행](#progress)
- [프로젝트 결과](#result)
- [코드 사용 방법](#use)

## 🧐 프로젝트 개요 <a name = "about"></a>

문장 속에서 단어 간 관계성을 파악하는 것은 문장의 의미나 의도를 해석함에 있어 많은 도움을 제공해준다. 문장 속 단어에 대한 속성과 관계를 파악하는 문제를 관계 추출(Relation Extraction, 이하 RE)이라고 부른다. 본 대회에서 사용한 dataset은 KLUE Benchmark의 RE로 sentence, subject entity, object entity가 주어졌을 때, subject entity와 object entity의 관계를 추출하도록 설계되어 있으므로 이 task는 classification의 일종이라고 볼 수 있다.
아래 예시는 한 문장에서 object entity와 subject entity의 관계를 분류하는 문제이다. 

```
문장 : 디비시옹 엘리트는 1926년 창설되어 FFBS가 주관하는 프랑스의 프로 야구 리그이다.

object entity : {’word’:”디비시옹 엘리트”, ‘type’: “PER”, ‘start_idx’: 0, ‘end_idx’: 7}

subject entity : {’word’: ”1926년”, ‘type’: ”DAT”, ‘start_idx’: 10, ‘end_idx’: 15}

label : “ORG:founded”
```

  Subject entity와 object entity에는 entity word, index, type 정보가 포함된다. word는 entity 단어, index는 문장 내에서 관계를 보고자 하는 entity word의 위치, type은 해당 entity가 어떤 종류(인물, 단체 등)의 것인지 설명한다.
  
  데이터는 총 30개의 라벨(관계를 표현하는 29개 라벨과 no_relation)로 이루어져 있다. 라벨의 경우 분포가 매우 불균형하다.
  
  본 대회에서는 32470개의 train set과 7765개의 test set을 사용한다. 별도의 dev set을 제공하지 않고, 본래 KLUE RE dataset의 dev set을 test set으로 사용한다.
## 🏁 팀 구성 및 역할 <a name = "team_member"></a>

- Project Manager: 박수현
- Code Reviewer: 류재환, 박승현
- Researcher: 김준휘, 설유민

## 🔧 프로젝트 진행 <a name = "progress"></a>

![ryu drawio](https://user-images.githubusercontent.com/99873921/205595706-628ebac6-bc11-48c9-978c-e8f1f23ca5c4.png)


## ✍️ 프로젝트 결과 <a name = "result"></a>

- 27개의 PR

![화면 캡처 2022-12-05 180217](https://user-images.githubusercontent.com/99873921/205596813-7a568fc7-b2cf-47fa-bc10-62bdd11f4642.png)
- 126개의 Commit

![화면 캡처 2022-12-05 180230](https://user-images.githubusercontent.com/99873921/205596895-e8ee1928-7f79-4534-825b-6781f2ccf1c5.png)
- 노션과 피어세션에서의 활발한 토론
- **값진 프로젝트 경험** 

## 🎉 코드 사용 방법 <a name = "use"></a>

run.py 파일에서 원하는 parameter를 설정한 이후 아래와 같이 실행 가능합니다.<br>
parameter에 대한 설명은 run.py에 담겨 있습니다.
```python3
python3 run.py
```

## 외부 리소스 
https://huggingface.co/klue/roberta-large 의 tokenizer에 스페셜 토큰을 추가한 tokenizer가 main/data_processing/newtokenzier에 있습니다.

## Wrap Up Report
[KLUE_NLP_팀 리포트(06조).pdf](https://github.com/boostcampaitech4lv23nlp1/level2_klue_nlp-level2-nlp-06/files/10193340/Relation.Extraction.Task.Wrap.pdf)
