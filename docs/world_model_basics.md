# 월드모델(World Model) 기초부터 실전까지: 수식 중심 설명

## 0. 문서 목적
이 문서는 월드모델을 처음 접하는 독자를 대상으로, "환경의 작동 원리를 내부적으로 학습해 미래를 시뮬레이션하는 모델"이라는 핵심 아이디어를 수학적으로 정리한다.  
강화학습(RL), 제어(control), 시계열 예측에 공통으로 적용되는 형태를 기준으로 설명한다.

---

## 1. 월드모델이란 무엇인가?
월드모델은 에이전트가 관측한 데이터로부터 환경의 동역학을 학습하는 모델이다.

- 입력: 과거 관측 \(o_{1:t}\), 행동 \(a_{1:t}\)
- 출력: 다음 관측/보상의 분포 또는 잠재상태의 전이

핵심은 "한 스텝 예측"이 아니라 "다단계 롤아웃(rollout)"이다.  
즉, \(t+1\)뿐 아니라 \(t+H\)까지의 미래를 내부 시뮬레이션해서 정책을 개선한다.

---

## 2. 문제 설정: POMDP 관점
현실 문제는 부분관측 환경(POMDP)으로 보는 것이 일반적이다.

\[
\mathcal{M}=(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, \Omega, R, \gamma)
\]

- \(\mathcal{S}\): 실제 상태(직접 관측 불가 가능)
- \(\mathcal{A}\): 행동 공간
- \(\mathcal{O}\): 관측 공간
- \(T(s_{t+1}\mid s_t,a_t)\): 상태 전이
- \(\Omega(o_t\mid s_t)\): 관측 생성
- \(R(r_t\mid s_t,a_t)\): 보상 생성
- \(\gamma\in(0,1)\): 할인율

월드모델은 보통 \(s_t\) 대신 잠재상태 \(z_t\)를 도입해 학습한다.

---

## 3. 확률적 잠재동역학 모델
대표 구조는 다음 세 가지 분포로 구성된다.

1) 전이(동역학) 모델
\[
p_\theta(z_t \mid z_{t-1}, a_{t-1})
\]

2) 관측(디코더) 모델
\[
p_\theta(o_t \mid z_t)
\]

3) 보상 모델
\[
p_\theta(r_t \mid z_t)
\]

실제 학습에서는 추론 모델(인코더)도 필요하다.

\[
q_\phi(z_t \mid z_{t-1}, a_{t-1}, o_t)
\]

여기서 \(q_\phi\)는 posterior 근사, \(p_\theta\)는 generative 모델 역할이다.

---

## 4. 학습 목표: ELBO(증거 하한)
관측 시퀀스 \(o_{1:T}\)의 로그우도 \(\log p_\theta(o_{1:T}, r_{1:T})\)를 직접 최적화하기 어려우므로 ELBO를 최대화한다.

\[
\log p_\theta(o_{1:T},r_{1:T}) \ge
\sum_{t=1}^{T}
\mathbb{E}_{q_\phi(z_t)}[\log p_\theta(o_t\mid z_t)+\log p_\theta(r_t\mid z_t)]
- \mathrm{KL}\!\left(q_\phi(z_t\mid \cdot)\,\|\,p_\theta(z_t\mid z_{t-1},a_{t-1})\right)
\]

실무 손실함수는 보통 부호를 바꾼 최소화 형태:

\[
\mathcal{L}_{\text{wm}}
=
\sum_{t=1}^{T}
\Big(
\lambda_o \mathcal{L}^{(t)}_{\text{recon}}
+\lambda_r \mathcal{L}^{(t)}_{\text{reward}}
+\lambda_{\text{kl}} \mathcal{L}^{(t)}_{\text{KL}}
\Big)
\]

- \(\mathcal{L}_{\text{recon}}\): 관측 재구성 오차 (MSE 또는 NLL)
- \(\mathcal{L}_{\text{reward}}\): 보상 예측 오차
- \(\mathcal{L}_{\text{KL}}\): posterior-prior 정합

---

## 5. 결정론 + 확률론 결합(RSSM 계열)
많이 쓰이는 형태는 결정론적 은닉상태 \(h_t\)와 확률 잠재 \(z_t\)를 결합한다.

\[
h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})
\]
\[
z_t \sim p_\theta(z_t\mid h_t)
\]
\[
z_t \sim q_\phi(z_t\mid h_t, o_t)
\]

장점:
- \(h_t\): 긴 시간 의존성 유지
- \(z_t\): 불확실성 표현

즉, 장기 메모리와 확률적 미래를 동시에 다룬다.

---

## 6. 월드모델 위에서의 계획(Planning)
월드모델의 핵심 활용은 "행동을 실제로 하기 전에 잠재공간에서 평가"하는 것이다.

길이 \(H\) 행동열 \(a_{t:t+H-1}\)의 예측 누적보상:

\[
J(a_{t:t+H-1})=
\mathbb{E}_{p_\theta}
\left[\sum_{k=0}^{H-1}\gamma^k \hat r_{t+k}\right]
\]

최적 계획:
\[
a_{t:t+H-1}^*=\arg\max_{a_{t:t+H-1}} J(a_{t:t+H-1})
\]

실전에서는 CEM(Cross-Entropy Method)으로 근사:
1. 행동열 샘플 \(N\)개 생성  
2. 상위 \(K\)개(elite) 선택  
3. elite 분포로 샘플링 분포 갱신  
4. 반복 후 첫 행동만 실행(MPC)

---

## 7. 정책학습과 결합
월드모델만으로 끝나지 않고 actor-critic과 결합한다.

정책 \(\pi_\psi(a_t\mid z_t)\), 가치함수 \(V_\eta(z_t)\)를 두고  
상상 궤적(imagined trajectories)에서 정책을 업데이트한다.

정책 목적함수(예시):
\[
\max_\psi\;
\mathbb{E}_{z_t\sim \mathcal{D},\,a_t\sim \pi_\psi}
\left[
\sum_{k=0}^{H-1}\gamma^k \hat r_{t+k}
\right]
\]

가치함수는 TD 타깃으로 학습:
\[
y_t = \hat r_t + \gamma V_\eta(\hat z_{t+1})
\]
\[
\mathcal{L}_V = \mathbb{E}\left[(V_\eta(z_t)-y_t)^2\right]
\]

---

## 8. 불확실성과 모델 바이어스
월드모델은 다단계 롤아웃에서 오차가 누적된다.

\[
\epsilon_H \lesssim \sum_{k=1}^{H} \alpha_k \epsilon_{\text{1-step}}
\]

따라서 다음 기법이 중요하다.

- 짧은 horizon 학습 + 점진적 확장
- ensemble로 epistemic uncertainty 추정
- latent overshooting(멀티스텝 정합 항 추가)
- 실제 환경 데이터로 주기적 재학습(Dyna 루프)

Ensemble \(M\)개일 때 불확실성 예시:
\[
\mathrm{Var}[\hat r_t] \approx
\frac{1}{M}\sum_{m=1}^{M}\left(\hat r_t^{(m)}-\bar r_t\right)^2
\]

---

## 9. 시계열/의사결정 문제에의 적용 관점
가격, 수요, 시스템 부하 같은 시계열에서도 월드모델 관점이 유효하다.

- \(o_t\): 피처 벡터(가격, 환율, 뉴스 임베딩 등)
- \(a_t\): 행동(매수/대기/매도 혹은 리소스 할당)
- \(r_t\): 목적함수(수익, 비용 절감, SLA 만족도)

단순 예측 모델과 차이:
- 예측 모델: \(o_{t+1}\) 정확도 중심
- 월드모델: 행동에 따른 "미래 분기"를 모델링

즉, "맞추는 모델"이 아니라 "의사결정용 시뮬레이터"다.

---

## 10. 구현 시 최소 체크리스트
1. 데이터 시퀀스 정의  
- \((o_t,a_t,r_t,o_{t+1},done_t)\) 형태로 저장

2. 학습 손실 분해  
- 재구성, 보상, KL 비중 \(\lambda\) 튜닝

3. 롤아웃 안정성 검증  
- 1-step/5-step/20-step 예측오차 분리 측정

4. 정책 평가 분리  
- 모델 내부 리턴과 실제 환경 리턴을 함께 기록

5. 드리프트 대응  
- 최신 데이터 반영 주기와 재학습 트리거 정의

---

## 11. 핵심 요약
- 월드모델은 \(p(z_t\mid z_{t-1},a_{t-1})\), \(p(o_t\mid z_t)\), \(p(r_t\mid z_t)\)를 학습해 미래를 내부 시뮬레이션한다.
- 학습은 ELBO(재구성 + 보상 + KL) 기반이 표준적이다.
- 활용은 planning(CEM/MPC) 또는 imagination 기반 policy learning이 핵심이다.
- 성능 병목은 장기 롤아웃 오차와 모델 바이어스이며, ensemble/재학습/overshooting으로 완화한다.

---

## 부록 A. 자주 쓰는 손실 항 예시
가우시안 가정 시 관측 NLL(상수항 제외):
\[
\mathcal{L}_{\text{recon}}
=
\frac{1}{2\sigma_o^2}\|o_t-\hat o_t\|_2^2
\]

가우시안 posterior/prior KL:
\[
\mathrm{KL}(\mathcal{N}(\mu_q,\sigma_q^2)\|\mathcal{N}(\mu_p,\sigma_p^2))
=
\log\frac{\sigma_p}{\sigma_q}
+\frac{\sigma_q^2+(\mu_q-\mu_p)^2}{2\sigma_p^2}
-\frac{1}{2}
\]

---

## 부록 B. 용어 정리
- World Model: 환경 동역학 학습 모델
- Latent State \(z_t\): 압축된 내부 상태표현
- Posterior \(q_\phi\): 관측을 본 뒤의 잠재분포
- Prior \(p_\theta\): 관측 없이 예측한 잠재분포
- Rollout: 모델로 미래를 전개하는 과정
- MPC: 매 스텝 재계획하며 첫 행동만 실행하는 제어 방식
