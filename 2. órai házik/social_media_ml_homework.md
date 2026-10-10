# Social Media Impact - gépi tanulási házi

## Kutatási kérdés

1. Megjósolható-e a hallgatók `Academic_Performance_GPA` értéke a közösségimédia-használat, az alvás, a stressz és a mentális egészség mutatói alapján?
2. Megjósolható-e az `Overall_Impact` kategória ugyanezekből a változókból?

## Változók

- **Magyarázóváltozók:** `Age`, `Gender`, `Academic_Level`, `Primary_Platform`,
  `Daily_Usage_Hours`, `Weekend_Extra_Hours`, `Device_Type`,
  `Sleep_Duration_Hours`, `Sleep_Quality_Score`, `Late_Night_Usage`,
  `Social_Comparison_Frequency`, `Perceived_Stress_Score`,
  `Mental_Health_Index`.
- **Numerikus eredményváltozó:** `Academic_Performance_GPA`.
- **Kategorikus eredményváltozó:** `Overall_Impact` (`Beneficial`, `Neutral`,
  `Negative`).
- `Student_ID` azonosító, ezért nem használjuk modellváltozóként.
- A GPA-t az `Overall_Impact` klasszifikációjánál szándékosan nem használjuk,
  mert az összetett címke feltehetően az akadémiai teljesítményt is magában
  foglalhatja. Ezzel elkerüljük a target leakage problémát.

## Módszerek

### Regresszió a GPA előrejelzésére

- baseline: átlagos GPA
- lineáris regresszió
- döntési fa
- random forest
- gradient boosting
- neurális háló

Mérőszámok: MAE, RMSE és R². A kisebb MAE/RMSE, illetve a nagyobb R² a jobb.

### Klasszifikáció az `Overall_Impact` előrejelzésére

- baseline: leggyakoribb osztály
- multinomiális logisztikus regresszió
- döntési fa
- random forest
- gradient boosting
- neurális háló

Mérőszámok: accuracy és macro F1. A macro F1 azért fontos, mert minden
osztály teljesítményét azonos súllyal veszi figyelembe.

## Futtatás

```powershell
pip install -r requirements.txt
python social_media_ml_homework.py
```

A program a `results/` mappába menti az EDA ábrákat, modell-összehasonlító
táblázatokat, confusion matrixot és a permutation importance eredményeket.

## Értelmezési korlátok

Az adatok kérdőíves, megfigyeléses adatok, ezért az eredmények kapcsolatokat
mutatnak, nem bizonyítanak ok-okozati hatást. A modell teljesítményét külön
tesztkészleten mérjük; a végső beszámolóban a baseline-hoz viszonyított
javulást és a legjobb modell mérőszámait kell kiemelni.

## Lefuttatott eredmények

A program 80/20 arányú tanító- és tesztfelosztással futott, `random_state=42`
beállítással.

### GPA-regresszió

| Modell | MAE | RMSE | R² |
|---|---:|---:|---:|
| Lineáris regresszió | 0,218 | 0,269 | 0,567 |
| Gradient boosting | 0,222 | 0,274 | 0,552 |
| Random forest | 0,222 | 0,276 | 0,546 |
| Döntési fa | 0,232 | 0,282 | 0,524 |
| Neurális háló | 0,229 | 0,283 | 0,522 |
| Baseline (átlag) | 0,332 | 0,409 | -0,001 |

A lineáris regresszió átlagosan körülbelül 0,218 GPA-ponttal tér el a tesztadatok
valós értékeitől, és a GPA varianciájának mintegy 56,7%-át magyarázza. A
baseline-hoz képest jelentős javulást ad.

### `Overall_Impact` klasszifikáció

| Modell | Accuracy | Macro F1 |
|---|---:|---:|
| Döntési fa | 0,954 | 0,884 |
| Gradient boosting | 0,951 | 0,885 |
| Random forest | 0,950 | 0,881 |
| Logisztikus regresszió | 0,947 | 0,885 |
| Neurális háló | 0,946 | 0,869 |
| Baseline (leggyakoribb osztály) | 0,818 | 0,300 |

A gradient boosting macro F1-ben a legjobb, ezért kiegyensúlyozottabb
klasszifikációs teljesítményt ad a ritkább `Negative` és `Neutral` osztályokon
is. A tesztkészleten a macro F1 értéke 0,885, az accuracy 95,11%.

## Következtetés

A vizsgált szociális média-, alvás-, stressz- és mentális egészségváltozók
együtt használva jól előrejelzik az `Overall_Impact` kategóriát. A GPA
előrejelzése nehezebb, de a modellek a pusztán átlagot használó baseline-hoz
képest érdemben jobb eredményt adnak. A permutation importance alapján a GPA
modellben a `Daily_Usage_Hours`, a klasszifikációban pedig a
`Perceived_Stress_Score` és a `Mental_Health_Index` voltak a legfontosabb
változók.
