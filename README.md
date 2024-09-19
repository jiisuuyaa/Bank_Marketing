# ML041

# 은행 마케팅 예측 프로젝트 README

streamlit 

## 주요 결과

- 최고 **AUC 점수**: 0.898422 (Gradient Boosting)
- 최고 **Recall 점수**: 0.831303 (Optuna Tuned)

### 성능 비교

| Model | AUC | Recall |
| --- | --- | --- |
| Gradient Boosting | 0.898422 | 0.828491 |
| Optuna Tuned | 0.887757 | 0.831303 |
| SVM | 0.893348 | 0.824742 |
| Random Forest | 0.886189 | 0.805998 |
| PyCaret | 0.384209 | 0.833177 |

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/29f4beeb-5a3b-4bdf-be4f-054ca428347f/8e10f184-39a8-4165-9d62-a9963579ddaf/image.png)

## 프로젝트 개요

이 프로젝트의 목표는 은행 마케팅 데이터를 바탕으로, 고객이 캠페인에 응답하여 **deposit**을 여는지 여부를 예측하는 머신러닝 모델을 개발하고 최적화하는 것입니다.

모델 개발을 위해 특성 선택(feature selection)을 수행하였고, **Gradient Boosting** 알고리즘을 사용했습니다. 하이퍼파라미터 최적화는 **Optuna** 프레임워크를 통해 이루어졌습니다. 최종 목표는 **AUC** 성능 지표와 **Recall**을 기준으로 최적의 예측 모델을 만드는 것입니다.

## 최적화 과정

하이퍼파라미터 최적화는 **Optuna**를 사용하여 다음과 같은 하이퍼파라미터를 조정했습니다:

- **트리 개수**: 108
- **최대 깊이**: 10
- **최소 분할 샘플 수**: 2

최적화된 모델은 AUC 0.8761을 기록했으며, **Optuna**의 대시보드를 통해 최적화 과정을 확인할 수 있습니다.

## 특성 선택 방법

특성 선택에는 Sequential Feature Selection(SFS)을 사용하여 성능을 개선하였습니다. 점진 선택법을 적용한 후, 모델의 성능이 향상되었습니다.

## 다양한 시도들

1. **점진 선택법 없이 feature selection + Gradient Boosting**: AUC 0.78
2. **PyCaret 사용 결과**: AUC 0.78(GBM), 0.81(LGBM+optuna)

## 트러블슈팅

### 문제점 및 해결책

- **PyCaret 사용 이슈**: TPU 런타임 환경에서 PyCaret이 실행되지 않음 → 런타임 환경을 변경하여 해결.
- **Feature selection hitmap 문제 →** Sequential Feature Selection(SFS) 사용으로 해결.
- 적절한 SQLight 설정이 어려웠음

## 결론

이 프로젝트에서는 **Optuna**를 통한 하이퍼파라미터 최적화와 **특성 선택**을 활용하여 은행 마케팅 데이터에 대한 예측 모델을 성공적으로 구축하였습니다. 최적화 결과, AUC 0.8761을 달성하였습니다.

- **Best Trial (number=31)**: AUC 0.8761
- **Params**: 트리 개수 108, 최대 깊이 10, 최소 분할 샘플 수 2
