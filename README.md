# Kepfeldolgozas_Project

Agyi MRI képek osztályozása négy osztályba (glioma, meningioma, pituitary, nincs tumor)
képfeldolgozásos előfeldolgozással és tradicionális gépi tanulással.

## Lépések

1. **Adattisztítás** – duplikált képek kiszűrése (a teszthalmaz 16%-a a tanítóban is szerepelt)
2. **Előfeldolgozás** – szürkeárnyalat → medián szűrés → agy kivágása (Otsu + morfológia + legnagyobb kontúr) → 128×128 → CLAHE
3. **Jellemzők** – HOG + LBP + GLCM + intenzitás-hisztogram (1838 dimenzió)
4. **Modellek** – k-NN, logisztikus regresszió, Random Forest, SVM (RBF); rácskeresés 5-szörös keresztvalidációval

## Eredmények (teszthalmaz, 1092 kép)

| Modell | CV accuracy | Teszt accuracy | F1 (makro) |
|---|---|---|---|
| k-NN (k=1) | 0,901 | 0,944 | 0,938 |
| Logisztikus regresszió | 0,864 | 0,863 | 0,848 |
| Random Forest | 0,871 | 0,855 | 0,845 |
| SVM (RBF) | 0,913 | 0,933 | 0,926 |

Ábrák és részletes eredmények: [`eredmenyek/`](eredmenyek/).

## Futtatás

Az adathalmaz nincs a repóban: az `agyikepek_4_osztaly/` mappát (`Training/` és `Testing/`,
osztályonként almappákkal) a projekt gyökerébe kell tenni.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python projekt.py
```
