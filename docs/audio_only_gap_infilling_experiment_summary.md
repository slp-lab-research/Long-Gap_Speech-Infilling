# Audio-Only Speech Gap Infilling 실험 정리

## 1. 원래 연구 주제

원래 목표는 **text-conditioned speech editing**이 아니라, transcript 없이 앞뒤 음성만 보고 중간 gap을 채우는 **audio-only speech gap infilling**이다.

핵심 문제 설정은 다음과 같다.

```text
입력:
prefix speech token + gap marker + suffix speech token

목표:
middle gap speech token 생성

조건:
transcript/text condition 사용하지 않음
```

즉 모델이 해야 하는 일은 단순 speech editing이 아니라, 주변 acoustic context를 보고 사라진 speech token 구간을 infill 하는 것이다.

## 2. 초안에서 말하는 문제 정의

논문 초안의 방향은 다음 주장과 맞닿아 있다.

```text
Speech gap infilling은 packet loss, microphone dropout, corrupted recording 등에서 발생한다.
짧은 gap은 acoustic continuity로 어느 정도 복원 가능하지만,
긴 gap은 linguistic/semantic context reasoning이 필요하다.
LLM-based speech foundation model이 이러한 long-gap infilling에 도움이 될 수 있는지 본다.
```

따라서 초안 기준의 핵심은 다음이다.

```text
audio-only
surrounding speech context
gap duration별 성능 분석
semantic/contextual reasoning 가능성 평가
```

주의할 점:

```text
word-aligned gap은 초안의 필수 조건이 아니다.
real-world gap은 단어 경계에 맞게 비는 것이 아니라 random time/frame gap에 가깝다.
```

## 3. 지금까지의 큰 실험 흐름

### 3.1 초기 VoiceCraft-X 학습 코드 구성

처음에는 VoiceCraft-X 공개 repo에 training forward가 충분히 없어서, 다음 우선순위로 학습 코드를 구성했다.

```text
1순위: VoiceCraft trainer.py
2순위: VoiceCraft voicecraft.py masking/loss
3순위: VoiceCraft dataset/collate
4순위: VoiceCraft-X voicecraftx.py
5순위: VoiceCraft-X helper.py
```

핵심 판단:

```text
VoiceCraft의 학습 루프는 참고 가능하지만,
VoiceCraft-X는 Qwen 기반 LLM backbone과 speech token 구조가 달라서
입력/forward/loss는 새로 구성해야 했다.
```

### 3.2 tokenized WebDataset 구성

LibriTTS-R 데이터를 VoiceCraft-X codec으로 미리 tokenizing하여 WDS로 만들었다.

목적:

```text
학습 중 wav를 매번 codec encoder에 넣지 않기 위해
speech token을 사전 계산해 I/O와 preprocessing 병목을 줄인다.
```

WDS에 포함된 주요 정보:

```text
speech codec tokens
text
metadata
speaker embedding
word alignment metadata가 있는 경우 alignment 정보
```

이후 online masking은 학습 시점에 collator에서 동적으로 적용했다.

### 3.3 기본 text-conditioned / FIM 계열 실험

초기에는 VoiceCraft-X의 원래 editing 흐름을 참고하여 text/FIM prompt를 사용하거나 변형했다.

관찰:

```text
eval_loss 약 3.5~3.7
masked_token_accuracy 약 0.20 전후
codebook별 accuracy도 약 0.20 근처
```

해석:

```text
수치가 계속 비슷하게 정체됨.
text-conditioned 구조는 원래 speech editing에는 가깝지만,
원래 목표인 transcript-free audio-only gap infilling과는 다름.
```

## 4. Audio-Only 방향으로 전환

원래 주제를 명확히 하면서 text/transcript condition을 제거하는 방향으로 전환했다.

목표:

```text
text/transcript 없이
speech token prefix/suffix만 보고
middle gap speech token을 예측
```

이론적 근거:

```text
VoiceCraft-X는 speech token embedding과 speech decoder head를 가진다.
LLM backbone은 discrete sequence modeling이 가능하다.
따라서 speech token도 acoustic token LM처럼 모델링할 수 있다.
```

하지만 어려운 점:

```text
일반 next-token prediction은 prefix-only continuation이다.
gap infilling은 prefix와 suffix를 모두 봐야 한다.
따라서 suffix-aware infilling 학습이 필요하다.
```

## 5. Audio-Only 실험 1: roles-fim 구조

### 설정

처음 audio-only 전용 구조는 다음과 같았다.

```text
speaker embedding
+ PREFIX_ROLE
+ prefix speech tokens
+ SUFFIX_ROLE
+ suffix speech tokens
+ MIDDLE_ROLE
→ middle speech tokens
```

특징:

```text
text/transcript condition 제거
text-speech interleaving 제거
middle gap token CE만 사용
learnable role embedding 추가
speech head + 마지막 LLM 일부 layer fine-tuning
```

### 문제점

이 구조는 모델 입장에서 원래 VoiceCraft-X prompt가 아니다.

```text
PREFIX_ROLE / SUFFIX_ROLE / MIDDLE_ROLE은 새로 추가된 구조적 신호다.
모델은 이 prompt protocol을 처음부터 배워야 한다.
```

### 결과 해석

일부 조건에서는 token accuracy가 올라가는 듯한 신호가 있었지만, 실제 generation sample 품질은 좋지 않았다.

핵심 문제:

```text
학습 loss/token accuracy가 좋아져도
실제 wav에서 gap이 자연스럽게 채워지는지는 별도 문제다.
```

## 6. Audio-Only 실험 2: word-aligned mask

### 설정

random frame mask가 단어/음소 중간을 자르는 문제가 있다고 보고, word alignment 기반 mask도 실험했다.

```text
mask_mode: word-aligned
mask_len: 10~30 codec frames
word_mask_max_words: 4
```

### 의도

단어 token을 예측하려는 것이 아니라, mask boundary를 더 자연스럽게 만들려는 목적이었다.

```text
random frame mask:
음소/단어 중간을 자를 가능성이 높음

word-aligned mask:
단어 또는 단어 span 단위로 더 깔끔한 gap 구성
```

### 팀원 지적

팀원 의견:

```text
모델은 speech codec token을 예측하지,
이 token이 단어인지 음소인지는 모른다.
word-aligned gap이 타당한가?
```

정리된 답:

```text
예측 대상은 항상 speech codec token이므로 팀원 말은 맞다.
하지만 mask boundary가 어디에 걸리느냐는 task difficulty와 ambiguity를 바꾼다.
word-aligned는 main task가 아니라 masking strategy/보조 분석으로 봐야 한다.
```

### 초안과의 관계

초안의 real-world gap 관점에서는 word-aligned가 기본 설정이 되면 안 된다.

```text
packet loss / dropout / corrupted recording은 random time gap에 가깝다.
```

따라서 최종 주제 기준 메인 실험은 random frame/time gap이 더 적합하다.

## 7. Audio-Only 실험 3: original-mask-roles 구조

### 배경

지금까지의 role FIM 구조가 너무 새로운 prompt protocol이라는 문제가 있었다.

그래서 원래 VoiceCraft-X의 speech mask 흐름에 더 가깝게 만들고, role만 최소로 붙이는 구조를 시도했다.

### 설정

```text
speaker
+ PREFIX_ROLE + prefix speech
+ GAP_ROLE + speech mask marker
+ SUFFIX_ROLE + suffix speech
+ MIDDLE_ROLE
→ middle speech tokens
```

옵션:

```text
--prompt-format original-mask-roles
```

### 해석

이 구조도 완전히 원래 VoiceCraft-X prompt는 아니다.

```text
원래 VoiceCraft-X:
text/transcript condition + speaker + prefix + speech mask + suffix + speech mask

현재 original-mask-roles:
text 없음 + role embedding 추가 + speech mask 구조 일부 유지
```

즉 “원래 prompt와 가까운 변형”이지 “원래 prompt 그대로”는 아니다.

## 8. 동료 의견: prefix/suffix에도 loss를 줘야 하는가

동료 의견:

```text
VoiceCraft-X editing 결과는 replacement 단어만이 아니라 대체된 전체 음성이다.
따라서 모델도 전체 시퀀스를 생성하게 하고,
prefix/suffix에도 loss를 줘야 하는 것 아닌가?
```

정리된 판단:

```text
최종 wav가 prefix + generated middle + suffix 전체라고 해서
반드시 prefix/suffix에 loss를 줘야 하는 것은 아니다.

gap infilling objective에서는 prefix/suffix는 조건이고,
middle이 생성 대상이다.
```

하지만 보조 loss로는 실험 가치가 있다고 봤다.

```text
prefix/suffix CE는 gap infilling 자체라기보다
audio-only speech token LM adaptation을 돕는 auxiliary objective다.
```

## 9. Audio-Only 실험 4: context auxiliary CE

### 설정

새 인자를 추가했다.

```text
--context-loss-weight
```

loss 구성:

```text
total_loss =
  middle_ce
  + energy_loss_weight * energy_loss
  + context_loss_weight * prefix_suffix_ce
```

실험 명령의 핵심:

```text
main loss: middle gap token CE
aux loss: prefix/suffix speech token CE
context_loss_weight = 0.1
mask_mode = frame
prompt_format = original-mask-roles
unfreeze_llm_layers = 8
```

### CE가 걸린 위치

메인 CE:

```text
masked middle gap token
```

보조 CE:

```text
prefix + suffix token
middle 구간은 -100으로 제외
```

### 결과

후반 로그 예시:

```text
loss: 약 7.1~7.6
ce_loss: 약 7.0~7.4
context_loss: 약 4.6~5.6
masked_token_accuracy: 대부분 0~2%
가끔 11% 정도 튐
```

### 해석

이 설정은 학습이 안 됐다고 판단했다.

근거:

```text
middle gap CE가 7 근처에서 정체
masked token accuracy가 거의 0
3000 step 후반까지 개선 없음
```

또한 context aux loss는 VRAM을 크게 증가시켰다.

```text
prefix/suffix 전체 sequence에 CE를 걸면 attention memory가 길이 제곱으로 증가한다.
추가 forward/backward 때문에 24GB GPU에서 OOM 발생 가능.
```

결론:

```text
전체 prefix/suffix auxiliary CE는 현재 방식으로는 효과 없음.
메모리만 크게 쓰고 middle CE 개선에 직접적으로 도움을 주지 못했다.
```

## 10. 지금까지 실패 원인에 대한 현재 판단

### 10.1 문제는 단순 파라미터 튜닝 부족만은 아님

지금까지 여러 설정을 바꿨지만 반복적으로 정체가 나타났다.

```text
loss/accuracy가 일정 수준에서 정체
실제 generated wav 품질도 좋지 않음
```

따라서 단순히 lr, scheduler, step 수만의 문제는 아닐 가능성이 크다.

### 10.2 원본 codec token exact reconstruction이 너무 어려움

audio-only gap infilling은 가능한 답이 여러 개다.

```text
앞뒤 context에 자연스럽게 이어지는 speech는 여러 개일 수 있음.
하지만 CE는 원본 codec token 하나만 정답으로 둔다.
```

즉 모델이 자연스러운 gap을 만들 수 있어도, 원본 token과 다르면 모두 오답 처리된다.

### 10.3 random frame gap은 더 어렵다

random frame gap은 다음을 자주 포함한다.

```text
음소 중간
단어 중간
coarticulation 구간
호흡/무음/전이 구간
```

이 경우 사람도 원본 token을 정확히 복원하기 어렵다.

### 10.4 VoiceCraft-X 원래 task와 다르다

VoiceCraft-X는 원래 text/transcript condition을 받는 editing/generation 구조에 가깝다.

현재 실험은 다음을 제거했다.

```text
text condition
middle content guidance
text-speech interleaving
```

따라서 모델 입장에서는 원래 사용 분포에서 꽤 벗어난 task다.

### 10.5 지금까지 대부분 새 prompt 구조를 학습시켰다

지금까지 실험은 다음과 같은 새 구조를 계속 추가했다.

```text
PREFIX_ROLE
SUFFIX_ROLE
MIDDLE_ROLE
GAP_ROLE
audio-only FIM protocol
original-mask-roles protocol
```

즉 모델이 gap infilling을 배우는 동시에 새 prompt protocol도 배워야 했다.

이 점이 실패 원인 중 하나일 수 있다.

## 11. role embedding 없이 하면 원래 prompt인가?

아니다.

role embedding을 빼면 원래에 더 가까워지지만, 원래 VoiceCraft-X prompt 그대로는 아니다.

```text
원래 VoiceCraft-X:
text/transcript condition
+ speaker embedding
+ prefix speech
+ speech mask
+ suffix speech
+ speech mask
→ middle speech

role 없는 audio-only:
speaker embedding
+ prefix speech
+ speech mask
+ suffix speech
+ speech mask
→ middle speech
```

차이:

```text
text condition이 없음.
```

따라서 가장 원래에 가까운 audio-only ablation은 다음이다.

```text
role embedding 없음
새 special token 없음
speaker embedding 그대로 유지
speech mask 구조 그대로 유지
text slot은 empty 또는 dummy로 유지
middle CE
```

## 12. 현재 중요한 결론

현재까지의 가장 중요한 정리는 다음이다.

```text
지금까지는 대부분 새로운 prompt 구조를 모델에게 학습시키는 실험이었다.
따라서 성능이 안 나온 이유가 audio-only infilling 자체의 한계인지,
새 prompt protocol 적응 실패인지 분리되지 않았다.
```

따라서 다음 실험은 새 구조를 최대한 줄여야 한다.

## 13. 다음 실험 방향 제안

### 13.1 원래 VoiceCraft-X 구조에 가장 가까운 audio-only ablation

우선순위가 가장 높다.

```text
role embedding 제거
GAP_ROLE/PREFIX_ROLE/SUFFIX_ROLE/MIDDLE_ROLE 제거
speaker embedding 유지
speech prompt는 원래처럼:
prefix + speech_mask + suffix + speech_mask
text 입력 slot은 empty 또는 고정 dummy
loss는 middle CE
mask는 random frame
```

실험 질문:

```text
새 prompt protocol 없이,
VoiceCraft-X의 원래 speech mask 흐름에서 text condition만 비웠을 때
audio-only gap infilling이 되는가?
```

이 실험이 중요하다.

### 13.2 16/32 sample overfit sanity check

먼저 작은 subset에서 확실히 외워지는지 확인해야 한다.

```text
limit 16 또는 32
mask_len 1~5 frames
eval_on_train
충분한 step
```

해석:

```text
작은 gap/작은 subset도 못 외우면:
forward/loss/prompt 구조 문제 가능성 큼

작은 것은 외우는데 10~30 frame에서 실패하면:
task 난이도 또는 CE objective mismatch 가능성 큼
```

### 13.3 CE 외 평가 강화

최종 목표는 원본 token exact match가 아니라 자연스러운 gap infilling이다.

따라서 다음 평가가 필요하다.

```text
generated wav listening test
boundary continuity
RMS/envelope continuity
log-mel/spectral distance
speaker similarity
PESQ/STOI
ASR WER은 보조 지표
```

### 13.4 context auxiliary loss는 제한적으로만 재시도

전체 prefix/suffix CE는 비추천이다.

다시 한다면 다음처럼 제한해야 한다.

```text
gap 주변 prefix tail N frames
+ suffix head N frames
에만 auxiliary CE
```

전체 utterance prefix/suffix에 CE를 걸면 VRAM이 커지고, gap infilling에 직접적인 도움도 약하다.

## 14. 현재까지 실험별 요약

| 실험 | 설정 | 결과 | 해석 |
|---|---|---|---|
| 기본 inpainting 학습 | text/FIM 또는 generation context 기반 | eval acc 약 0.20 근처 정체 | text-conditioned 계열로 원래 audio-only 주제와 다소 다름 |
| audio-only roles-fim | text 제거, PREFIX/SUFFIX/MIDDLE role 추가 | 일부 token acc 신호 있으나 생성 품질 낮음 | 새 prompt protocol 학습 부담 큼 |
| word-aligned mask | 단어 경계 기반 gap 구성 | 일부 실험에서 수치 개선 신호 | main task가 아니라 mask strategy/보조 분석으로 봐야 함 |
| original-mask-roles | 원래 mask 흐름 + role 추가 | 여전히 새 구조이며 안정적 개선 불확실 | 원래 구조에 가까운 변형이지만 원본은 아님 |
| context auxiliary CE | middle CE + prefix/suffix CE 0.1 | ce_loss 약 7, acc 거의 0, OOM 위험 | 실패. 전체 prefix/suffix aux CE는 비효율적 |

## 15. 현재 공유용 한 문장 결론

```text
지금까지 실험은 audio-only speech gap infilling 방향 자체를 확인하는 과정이었지만,
대부분 VoiceCraft-X가 원래 보던 prompt 구조가 아닌 새 audio-only FIM/role 구조를 학습시키는 형태였다.
현재까지 random frame gap에서 middle token CE는 잘 수렴하지 않았고,
prefix/suffix auxiliary CE도 개선을 주지 못했다.
따라서 다음 단계는 role/special 구조를 제거하고,
원래 VoiceCraft-X의 speech mask prompt에 최대한 가까운 audio-only ablation으로 돌아가
작은 subset/짧은 gap overfit부터 검증하는 것이다.
```
