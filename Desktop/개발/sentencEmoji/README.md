# sentencEmoji: 문장 단위 이모지 선정 모델 🤗 

<p float="left" align="center">
    <img width="700" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FRn0zJ%2FbtrsnofWf6j%2FU2klZmLt6KQWsZkVdZ8caK%2Fimg.png" />  
</p>


## I. 개요 🍀
- 기존 아이폰/안드로이드폰 등에서 제공하는 단어 단위 이모지 추천 기능을 넘어, 한 문장 전체에 대해 자연스러운 이모지를 추천해주는 모델.
- 문장을 x, 이모지를 label로 하여 주어진 문장에 어울리는 이모지 하나를 출력하는 multi-class classification 모델

## II. 연구 배경 🙋
- 기존 아이폰/안드로이드폰 등에서는 단어 단위로 일치하는 이모지를 추천해준다. 하지만 우리는 단어 단위로 이모지를 문장 내 삽입하기 보다는 문장이 끝난 이후에 해당 문장 전체에 어울리는 이모지를 붙이는 것을 더 많이 볼 수 있다. 

## III. Data 🌈
- Twitter API로 크롤링한 한국어 문장 (2200여개 문장)
- Kaggle 영어 데이터 (https://www.kaggle.com/rexhaif/emojifydata-en/version/3?select=emojitweets-01-04-2018.txt) (18883592개 문장)

## IV. 모델 설명 🙏
- 문장을 x, 이모지를 label로 하여 주어진 문장에 어울리는 이모지 하나를 출력하는 multi-class classification 모델
- KoBERT 모델 활용

## V. 활용될 수 있는 비즈니스 영역 💯

- 인간이 아닌 기계가 새로운 문장을 생성하는 경우 적용 (eg. 챗봇) 
- 이모지를 추가하는 것에 많은 manual 리소스가 소요될 때 적용 (eg. 웹/모바일 애플리케이션으로 사용자에게 친근한 이미지로 접근하고자 하는 기업. 많은 양의 텍스트에 대해 이모지를 추가하는 것이 어려울 때)
- 이모지를 사용하기 어려운 기기(eg. 모바일 기기가 아닌 컴퓨터 등)에서 사용
