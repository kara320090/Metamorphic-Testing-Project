# Metamorphic Testing | MNIST 모델의 입력 변형 검증

손글씨 이미지에 변형을 가했을 때 분류 모델의 예측이 어떻게 달라지는지 비교하는 소프트웨어 테스트 실습 프로젝트입니다.

**Python · TensorFlow 1.x API · NumPy · SciPy · Matplotlib**

## 검증 아이디어

정답을 매번 새로 작성하는 대신, 원본 입력과 변형 입력 사이에 유지되어야 할 관계를 정의합니다. 이 프로젝트의 `E(source_y, follow_y)`는 두 입력의 예측 클래스가 같은지 비교합니다.

```text
MNIST 이미지 → 원본 예측
      ↓ T(입력 변형)
변형 이미지 → 변형 후 예측 → E(클래스 일치 비교) → T/F 기록·이미지 저장
```

회전, 밝기 조절, 흐림, 점 추가, 이진화와 반전 등의 변형 예시가 [`metamorphic_relation.py`](metamorphic_testing/lib/metamorphic_relation.py)에 있습니다. 큰 변형은 숫자 자체의 의미를 바꿀 수 있으므로 불일치가 곧바로 모델 결함을 뜻하지는 않습니다.

## 구현 구성

| 경로 | 내용 |
|---|---|
| [metamorphic_testing/lib](metamorphic_testing/lib/) | 변형 함수와 관계 검증, 반복 실행, 결과 저장 |
| [MNIST 실행 예제](metamorphic_testing/example/mnist/run_mnist.py) | 모델 로드, 샘플 선택, 테스트 실행 |
| [config.json](metamorphic_testing/example/mnist/config.json) | `NumTransformation` 반복 횟수 |
| [model/mnist/tensorflow](model/mnist/tensorflow/) | 모델 학습·복원·예측과 그래프 정보 |
| [결과 캡처](<결과물 캡쳐 이미지/>) | 회전·밝기·흐림 실험의 터미널 기록 |

## 재현 전 준비

1. `tf.Session`, `tf.logging`, `tensorflow.examples.tutorials.mnist`를 제공하는 기존 TensorFlow 1.x 호환 환경을 준비합니다. 현재 저장소에는 의존성 버전을 고정한 파일이 없습니다.
2. `metamorphic_relation.py`에서 사용할 **`T` 함수 하나를 활성화**합니다. 현재 기본 브랜치는 모든 `T` 예시가 주석 처리되어 있어 그대로는 import 단계에서 실행되지 않습니다.
3. `model/mnist/tensorflow/`의 체크포인트, 그래프 이름 JSON과 MNIST 자료를 확인합니다.
4. 필요하면 `config.json`에서 반복 횟수를 조절한 후 실행합니다.

```powershell
python metamorphic_testing/example/mnist/run_mnist.py
```

예제는 고정 난수 시드로 테스트 이미지 10개를 선택합니다. 원본·변형 이미지와 예측 결과가 실행 시각별 폴더에 저장됩니다.

## 학습·검토 포인트

- 입력 변형 `T`와 기대 관계 `E`를 분리하는 테스트 설계
- 정답 라벨 기반 정확도와 변형 전후 예측 일관성의 차이
- 원본·변형 이미지, 예측 클래스와 실행 조건을 함께 남기는 재현 가능한 기록

기존 실험 코드와 결과 기록을 보관하는 학습 저장소입니다. 비교할 때는 선택한 변형 함수와 반복 횟수를 함께 확인합니다.
