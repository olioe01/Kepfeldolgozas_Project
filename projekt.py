"""
Agyi MRI tumor osztályozás – képfeldolgozás + tradicionális gépi tanulás.

Adathalmaz: agyikepek_4_osztaly (glioma, meningioma, notumor, pituitary),
            a meglévő Training / Testing felosztással.

Lépések:
  0. Adattisztítás: duplikált képek kiszűrése (a teszthalmaz ~16%-a a tanítóban is szerepel)
  1. Előfeldolgozás: szürkeárnyalat -> zajszűrés (medián) -> agy kivágása
     (Otsu küszöb + legnagyobb kontúr) -> átméretezés 128x128 -> CLAHE kontrasztjavítás
  2. Jellemzőkinyerés: HOG (alak/él), LBP (textúra), GLCM (textúra), intenzitás-hisztogram
  3. Modellek: k-NN, Logisztikus regresszió, Random Forest, SVM (RBF)
  4. Kiértékelés: accuracy, precision, recall, F1, konfúziós mátrix

Futtatás:  .venv/bin/python projekt.py
"""
import glob
import hashlib
import os
import time

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ADAT = "agyikepek_4_osztaly"
OSZTALYOK = ["glioma", "meningioma", "notumor", "pituitary"]
MERET = 128
KI = "eredmenyek"
os.makedirs(KI, exist_ok=True)


# ---------------------------------------------------------------- 1. Előfeldolgozás
def agy_kivagas(img):
    """Otsu küszöböléssel és morfológiával megkeresi az agyat, és körbevágja."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, maszk = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    maszk = cv2.morphologyEx(maszk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    kontururok, _ = cv2.findContours(maszk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not kontururok:
        return img
    x, y, w, h = cv2.boundingRect(max(kontururok, key=cv2.contourArea))
    return img[y:y + h, x:x + w]


def elofeldolgoz(ut, lepesek=False):
    eredeti = cv2.imread(ut, cv2.IMREAD_GRAYSCALE)          # szürkeárnyalat
    szurt = cv2.medianBlur(eredeti, 3)                      # só-bors zaj szűrése
    vagott = agy_kivagas(szurt)                             # háttér eltávolítása
    meretezett = cv2.resize(vagott, (MERET, MERET), interpolation=cv2.INTER_AREA)
    kesz = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(meretezett)  # kontrasztjavítás
    if lepesek:
        return [eredeti, szurt, vagott, meretezett, kesz]
    return kesz


# ---------------------------------------------------------------- 2. Jellemzők
def jellemzok(img):
    f_hog = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    f_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    glcm = graycomatrix(img // 8, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    f_glcm = np.hstack([graycoprops(glcm, p).ravel()
                        for p in ("contrast", "homogeneity", "energy", "correlation")])
    f_hist, _ = np.histogram(img, bins=32, range=(0, 256), density=True)
    return np.hstack([f_hog, f_lbp, f_glcm, f_hist]).astype(np.float32)


def feldolgoz(ut):
    return jellemzok(elofeldolgoz(ut))


def kep_hash(ut):
    """Ujjlenyomat a duplikált képek kiszűréséhez (32x32-es kicsinyített kép MD5-je)."""
    img = cv2.resize(cv2.imread(ut, cv2.IMREAD_GRAYSCALE), (32, 32), interpolation=cv2.INTER_AREA)
    return hashlib.md5(img.tobytes()).hexdigest()


def fajlok(resz):
    utak, cimkek = [], []
    for i, o in enumerate(OSZTALYOK):
        f = sorted(glob.glob(os.path.join(ADAT, resz, o, "*.jpg")))
        utak += f
        cimkek += [i] * len(f)
    return utak, cimkek


def duplikatum_szures():
    """Eltávolítja a halmazon belüli duplikátumokat, és azokat a tesztképeket,
    amelyek a tanítóhalmazban is szerepelnek (adatszivárgás elkerülése)."""
    lattuk, ki, stat = set(), {}, {}
    for resz in ("Training", "Testing"):
        utak, cimkek = fajlok(resz)
        hashek = Parallel(n_jobs=-1)(delayed(kep_hash)(u) for u in utak)
        tart = [(u, c) for u, c, h in zip(utak, cimkek, hashek) if not (h in lattuk or lattuk.add(h))]
        ki[resz] = ([u for u, _ in tart], np.array([c for _, c in tart]))
        stat[resz] = (len(utak), len(tart))
    return ki, stat


def betolt(utak):
    return np.array(Parallel(n_jobs=-1)(delayed(feldolgoz)(u) for u in utak))


# ---------------------------------------------------------------- Ábrák az adatokról
def abra_eloszlas(adat):
    db = {r: np.bincount(adat[r][1], minlength=len(OSZTALYOK)) for r in adat}
    x = np.arange(len(OSZTALYOK))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, db["Training"], 0.4, label="Tanító")
    plt.bar(x + 0.2, db["Testing"], 0.4, label="Teszt")
    plt.xticks(x, OSZTALYOK)
    plt.ylabel("Képek száma")
    plt.title("Osztályeloszlás (duplikátumszűrés után)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{KI}/osztalyeloszlas.png", dpi=150)
    plt.close()
    return db


def abra_elofeldolgozas():
    nevek = ["Eredeti", "Medián szűrés", "Agy kivágása", "128x128", "CLAHE"]
    fig, ax = plt.subplots(len(OSZTALYOK), 5, figsize=(12, 10))
    for i, o in enumerate(OSZTALYOK):
        ut = sorted(glob.glob(os.path.join(ADAT, "Training", o, "*.jpg")))[10]
        for j, kep in enumerate(elofeldolgoz(ut, lepesek=True)):
            ax[i, j].imshow(kep, cmap="gray")
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(nevek[j])
        ax[i, 0].axis("on"); ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
        ax[i, 0].set_ylabel(o, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{KI}/elofeldolgozas.png", dpi=120)
    plt.close()


def abra_hog():
    ut = sorted(glob.glob(os.path.join(ADAT, "Training", "pituitary", "*.jpg")))[10]
    img = elofeldolgoz(ut)
    _, hog_kep = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                     visualize=True)
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for a, k, c in zip(ax, [img, hog_kep, lbp], ["Előfeldolgozott", "HOG", "LBP"]):
        a.imshow(k, cmap="gray"); a.set_title(c); a.axis("off")
    plt.tight_layout()
    plt.savefig(f"{KI}/jellemzok.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------- 3–4. Tanítás és kiértékelés
def main():
    adat, stat = duplikatum_szures()
    for r, (eredeti, marad) in stat.items():
        print(f"{r}: {eredeti} kép -> {marad} a duplikátumszűrés után")
    print("Osztályonként:", {r: abra_eloszlas(adat)[r].tolist() for r in adat})
    abra_elofeldolgozas()
    abra_hog()

    t = time.time()
    (u_tr, y_tr), (u_te, y_te) = adat["Training"], adat["Testing"]
    X_tr, X_te = betolt(u_tr), betolt(u_te)
    print(f"Jellemzők: {X_tr.shape[1]} dimenzió, {time.time() - t:.0f} s")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    modellek = {
        "k-NN": (make_pipeline(StandardScaler(), KNeighborsClassifier()),
                 {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7]}),
        "Logisztikus regresszió": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                                   {"logisticregression__C": [0.01, 0.1, 1]}),
        "Random Forest": (RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
                          {"max_features": ["sqrt", 0.05]}),
        "SVM (RBF)": (make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                      {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.0001]}),
    }

    sorok = []
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for k, (nev, (modell, racs)) in enumerate(modellek.items()):
        t = time.time()
        gs = GridSearchCV(modell, racs, cv=cv, scoring="accuracy", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        y_pred = gs.predict(X_te)
        p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro")
        sorok.append({"Modell": nev, "Legjobb paraméterek": str(gs.best_params_),
                      "CV accuracy": gs.best_score_, "Teszt accuracy": accuracy_score(y_te, y_pred),
                      "Precision": p, "Recall": r, "F1 (macro)": f1,
                      "Idő [s]": time.time() - t})
        print(f"\n=== {nev}  {gs.best_params_}  ({time.time() - t:.0f} s)")
        print(classification_report(y_te, y_pred, target_names=OSZTALYOK, digits=3))
        ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred), display_labels=OSZTALYOK).plot(
            ax=ax[k], cmap="Blues", colorbar=False)
        ax[k].set_title(f"{nev}\nacc = {accuracy_score(y_te, y_pred):.3f}")
        ax[k].set_xlabel("Becsült"); ax[k].set_ylabel("Valós")
        joblib.dump(gs.best_estimator_, f"{KI}/{nev.split()[0].lower()}_modell.joblib")
    plt.tight_layout()
    plt.savefig(f"{KI}/konfuzios_matrixok.png", dpi=130)
    plt.close()

    df = pd.DataFrame(sorok)
    df.to_csv(f"{KI}/eredmenyek.csv", index=False, float_format="%.4f")
    print("\n", df.drop(columns="Legjobb paraméterek").round(3).to_string(index=False))

    m = ["Teszt accuracy", "Precision", "Recall", "F1 (macro)"]
    x = np.arange(len(df))
    plt.figure(figsize=(9, 4.5))
    for i, c in enumerate(m):
        plt.bar(x + (i - 1.5) * 0.2, df[c], 0.2, label=c)
    plt.xticks(x, df["Modell"])
    plt.ylim(0.5, 1.0)
    plt.ylabel("Érték")
    plt.title("Modellek összehasonlítása a teszthalmazon")
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{KI}/modell_osszehasonlitas.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
