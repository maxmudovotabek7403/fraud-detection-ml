# Fraud Detection — IEEE-CIS Dataset

## Project Description

Ushbu loyiha [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) Kaggle competition'i asosida qurilgan. Maqsad — online tranzaksiyalarni **fraudulent** yoki **legitimate** ekanligini aniqlaydigan machine learning model yaratish. Bu binary classification vazifasi bo'lib, asosiy baholash metrikasi sifatida **ROC-AUC** ishlatilgan.

## Dataset

- **Manba:** IEEE-CIS Fraud Detection (Kaggle)
- **Fayllar:** `train_transaction.csv`, `test_transaction.csv`
- **Hajmi:** train — 590,540 satr / 394 ustun; test — 506,691 satr / 393 ustun (test'da `isFraud` yo'q)
- **Target:** `isFraud` (0 = legitimate, 1 = fraudulent)
- **Class imbalance:** ~3.50% fraud, ~96.50% legitimate — kuchli nomutanosib (imbalanced) dataset

## Problem Statement

Bank uchun har bir tranzaksiyani real vaqtda fraud yoki legitimate deb tasniflaydigan model kerak. Muammoning murakkabligi:
- Kuchli class imbalance (fraud kam uchraydi)
- Juda ko'p feature (300+ ustun, ko'plab anonim `V`-ustunlar)
- Yuqori missing value darajasi ko'plab ustunlarda
- False Positive (haqiqiy mijozni bloklash) va False Negative (fraud'ni o'tkazib yuborish) o'rtasidagi biznes trade-off

## EDA Summary

- Dataset kuchli imbalanced — shu sababli accuracy yagona baholash metrikasi sifatida yetarli emas, ROC-AUC/Precision/Recall/F1 ishlatildi
- `TransactionAmt` kuchli right-skewed (skewness ≈ 14.4); outlier tranzaksiyalar orasida fraud foizi (~5.07%) umumiy foizdan (~3.50%) yuqori — katta summalar fraud signali bo'lishi mumkin
- `ProductCD` bo'yicha fraud foizi sezilarli farqlanadi: "C" turida ~11.7%, "W" turida ~2.0%
- `card6` (karta turi) bo'yicha credit kartalarda fraud (~6.7%) debit kartalarga (~2.4%) nisbatan taxminan 3 baravar yuqori
- Ko'plab `V`-ustunlar orasida kuchli multicollinearity aniqlandi (masalan `V95`↔`V322`, r>0.999) — bu feature selection bosqichida hisobga olindi

## Preprocessing

- Missing value'lar 3 guruhga ajratildi: deyarli yo'q (0-5%), o'rtacha (5-50%), juda yuqori (50%+)
- Boosting modellar (XGBoost/LightGBM/CatBoost) uchun missing value'lar model ichida boshqarildi; Logistic Regression baseline uchun median/`"Unknown"` bilan to'ldirildi
- Categorical ustunlar Label Encoding orqali raqamlashtirildi (CatBoost uchun native categorical support ishlatildi)
- Almost-constant ustunlar (99%+ bitta qiymat) va sof ID ustun (`TransactionID`) feature to'plamidan chiqarib tashlandi
- Train/Validation/Test — 70% / 15% / 15%, `stratify` bilan class taqsimoti saqlangan holda bo'lindi

## Feature Engineering

Yaratilgan yangi feature'lar:
- **`TransactionAmt_log`** — kuchli skewness'ni kamaytirish uchun log-transformation
- **`Transaction_hour`, `Transaction_day`** — `TransactionDT`dan olingan vaqtga oid pattern'lar
- **`card1_count`** — kartaning tranzaksiyalar orasida qanchalik tez-tez uchrashi (frequency encoding); LightGBM va CatBoost modellarida eng muhim feature'lardan biri bo'lib chiqdi
- **`Amt_to_card1_mean_ratio`** — tranzaksiya summasining shu karta uchun o'rtacha summadan chetlanishi (anomaliya signali)

## Models

1. **Logistic Regression** — baseline model
2. **XGBoost** — gradient boosting
3. **LightGBM** — histogram-based, leaf-wise gradient boosting
4. **CatBoost** — native categorical feature support bilan gradient boosting

## Hyperparameter Tuning

Har bir boosting model uchun tasodifiy qidiruv (RandomizedSearchCV / qo'lda parameter sampling) orqali quyidagi parametrlar tuning qilindi: `n_estimators`/`iterations`, `max_depth`/`depth`, `learning_rate`, `subsample`, `colsample_bytree`, `num_leaves`, `min_child_samples`, `l2_leaf_reg`.

| Model | Before tuning (ROC-AUC) | After tuning (ROC-AUC) |
|---|---|---|
| XGBoost | 0.942 | **0.957** |
| LightGBM | 0.936 | 0.946 |
| CatBoost | 0.921 | 0.936 |

## Results

Validation to'plamidagi yakuniy natijalar (tuning'dan keyin):

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.862 | 0.133 | 0.735 | 0.225 |
| XGBoost (tuned) | **0.957** | **0.383** | 0.809 | **0.520** |
| LightGBM (tuned) | 0.946 | 0.285 | 0.811 | 0.422 |
| CatBoost (tuned) | 0.936 | 0.242 | 0.811 | 0.373 |

**Tanlangan model: XGBoost (tuned)** — eng yuqori ROC-AUC, Precision va F1-score ko'rsatdi, bu esa production muhitida false positive'larni minimallashtirish bilan fraud'ni ushlash o'rtasida eng yaxshi muvozanatni ta'minlaydi.

### Overfitting Analysis

| Model | Train ROC-AUC | Validation ROC-AUC | Gap |
|---|---|---|---|
| XGBoost | 0.991 | 0.957 | 0.035 |
| LightGBM | 0.973 | 0.946 | 0.027 |
| CatBoost | 0.957 | 0.936 | 0.021 |

Gap qiymatlari nisbatan kichik (<0.05), bu jiddiy overfitting emasligini ko'rsatadi, ammo production'ga chiqarishdan oldin regularizatsiya kuchaytirilishi yoki early stopping qo'shilishi tavsiya etiladi.

## Feature Importance

- Barcha uchta boosting modelda umumiy muhim feature: **`C14`**
- Yaratilgan `card1_count` feature'i LightGBM va CatBoost'da top qatorda chiqdi — feature engineering samarali bo'lganini tasdiqlaydi
- XGBoost asosan anonim `V`-ustunlarga, LightGBM/CatBoost esa `card`, `C`, `D`-ustunlariga ko'proq tayanadi — bu modellarning turli split-tanlash algoritmlari va V-ustunlar orasidagi multicollinearity bilan izohlanadi

## Final Conclusion

Loyiha davomida IEEE-CIS Fraud Detection dataseti chuqur EDA'dan o'tkazildi, missing value va outlier strategiyalari asoslandi, 4 ta yangi feature yaratildi va 4 ta model (1 baseline + 3 boosting) solishtirildi. Hyperparameter tuning barcha boosting modellarda sezilarli yaxshilanish berdi. Yakuniy tavsiya etilgan model — **tuned XGBoost** (ROC-AUC 0.957, F1 0.520) — bu production fraud detection tizimi uchun eng muvozanatli tanlov hisoblanadi, garchi joriy holatda ham ma'lum darajada false positive muammosi (Precision ≈ 0.38) saqlanib qolmoqda, bu keyingi iteratsiyalarda threshold tuning va qo'shimcha feature engineering orqali yaxshilanishi mumkin.
