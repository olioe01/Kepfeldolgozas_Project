# Bemutató útmutató – lépésről lépésre

Diasor: https://claude.ai/artifact/MDj57sUGfpyajbKekefVUu (minden dia alatt előadói jegyzet, mit mondj)

---

## 1. Előtte (otthon, egyszer)

1. **Diasor letöltése:** nyisd meg a linket → Share › Export → **PPTX** vagy **PDF**. Tedd pendrive-ra is, ha nem a saját gépedről adsz elő.
2. **Címdia:** írd át a `[Név / Neptun-kód]` részt.
3. **Kód futtatása egyszer**, hogy biztosan meglegyenek az eredmények és a modellek:
   ```bash
   cd ~/suli/kepfeldolgozasproj
   .venv/bin/python projekt.py        # kb. 6 perc
   ```
4. **VS Code:** `code .` → Python: Select Interpreter → `.venv/bin/python`.
   Nyisd meg előre: `projekt.py`, `eredmenyek/elofeldolgozas.png`, egy terminál.
5. **Próbáld el** stopperrel: a jegyzetek szerint kb. 4:55. Ha túl hosszú, a 4. diánál (jellemzők) lehet rövidíteni.

---

## 2. A bemutató menete (5 perc)

| # | Dia | Idő | A lényeg egy mondatban |
|---|---|---|---|
| 1 | Címdia | 0:20 | MRI képek osztályozása 4 osztályba, képfeldolgozás + klasszikus ML. |
| 2 | Adathalmaz | 0:40 | A 4 osztályos halmazt választottuk (van benne „nincs tumor”); a duplikátumokat kiszűrtük. |
| 3 | Előfeldolgozás | 0:50 | Szűrés → agy kivágása → 128×128 → CLAHE (az ábrán soronként látszik). |
| 4 | Jellemzők | 0:40 | HOG (alak) + LBP, GLCM (textúra) + hisztogram = 1838 szám/kép. |
| 5 | Modellek | 0:30 | k-NN, logisztikus regresszió, Random Forest, SVM; keresztvalidációs hangolás. |
| 6 | Eredmények | 0:50 | SVM és k-NN 93–94%, a másik kettő ~86%; CV-ben az SVM a legjobb. |
| 7 | Konfúziós mátrixok | 0:35 | „Nincs tumor” ~99%; a fő hiba glioma ↔ meningioma. |
| 8 | Összegzés | 0:30 | SVM 93,3%; tanulság: az adatot ellenőrizni kell. |

**A két legerősebb pont, ezeket hangsúlyozd:**
- **Duplikátumszűrés:** a tesztképek 16%-a a tanítóban is benne volt. Ezt felfedeztük és kiszűrtük.
- **Előfeldolgozás:** ez a kötelező képfeldolgozási rész, a 3. dia ábrája mindent megmutat.

---

## 3. Ha a kódot is kérik – mi mit csinál (`projekt.py`)

| Függvény | Sor | Mit csinál |
|---|---|---|
| `agy_kivagas()` | 50 | Gauss-elmosás → Otsu-küszöb → morfológiai nyitás → legnagyobb kontúr → kivágás |
| `elofeldolgoz()` | 62 | A teljes előfeldolgozás: szürkeárnyalat → medián → kivágás → 128×128 → CLAHE |
| `jellemzok()` | 74 | HOG + LBP + GLCM + hisztogram egy vektorba fűzve |
| `kep_hash()`, `duplikatum_szures()` | 90, 105 | Minden képre ujjlenyomat; ami már szerepelt, kimarad |
| `main()` – `modellek` | 185 | A 4 modell és a kipróbált paraméterek |
| `GridSearchCV(...)` | 200 | Paraméterhangolás 5-szörös keresztvalidációval, csak a tanítóhalmazon |
| `abra_*()` | 123–156 | A diákon látható ábrák elkészítése |

Mit mutass meg, ha kevés az idő: **csak az `elofeldolgoz()` függvényt** (62. sor). Pár sorban látszik az egész képfeldolgozási lánc.

---

## 4. Élő demó (opcionális, ~30 mp)

A VS Code termináljában:

```bash
.venv/bin/python demo.py agyikepek_4_osztaly/Testing/glioma/Te-gl_0010.jpg agyikepek_4_osztaly/Testing/notumor/Te-no_0010.jpg agyikepek_4_osztaly/Testing/pituitary/Te-pi_0010.jpg
```

Kimenet:
```
.../glioma/Te-gl_0010.jpg     ->  glioma
.../notumor/Te-no_0010.jpg    ->  notumor
.../pituitary/Te-pi_0010.jpg  ->  pituitary
```

Mit mondj: „A betanított SVM modellt betöltjük, a képet ugyanúgy előfeldolgozzuk, kinyerjük a jellemzőket, és a modell 1-2 másodperc alatt megmondja az osztályt.”

> A demóhoz kellenek az `eredmenyek/*.joblib` fájlok (a `projekt.py` készíti el őket) és az adathalmaz. Ezek nincsenek a GitHubon, ezért a saját gépedről demózz.

---

## 5. Várható kérdések és válaszok

**Miért ezt az adathalmazt?**
Négy osztálya van, köztük „nincs tumor”, és van kész tanító/teszt felosztás.

**Miért nem a k-NN a legjobb, ha a teszten az nyert (0,944)?**
Keresztvalidációban az SVM jobb (0,913 vs. 0,901). A k=1-es k-NN valószínűleg abból profitál, hogy ugyanannak a betegnek a szomszédos MRI-szeletei a tanító- és tesztrészben is szerepelnek.

**Miért keveri a modell a gliomát és a meningiomát?**
A két tumor textúrája és megjelenése hasonló az MRI-n.

**Mi az a CLAHE?**
Kontrasztjavítás kis régiónként (nem az egész képre egyszerre). Így a szövetek közti különbségek jobban látszanak.

**Miért vágtátok ki az agyat?**
Eltűnik a fekete háttér, és az agy minden képen ugyanakkora részt foglal el, így a jellemzők összehasonlíthatók.

**Mi az a HOG / LBP / GLCM?**
- HOG: az élek irányainak hisztogramja → alakot ír le.
- LBP: minden pixelt a szomszédaihoz hasonlít → finom textúra.
- GLCM: milyen gyakran áll egymás mellett két szürkeárnyalat → textúra (kontraszt, homogenitás…).

**Miért nem deep learning?**
Opcionális volt; a feladat a képfeldolgozás + klasszikus ML volt, azzal is 93% feletti eredményt értünk el.

**Hogyan tanítottatok, nem csaltatok a teszthalmazzal?**
A paramétereket csak a tanítóhalmazon, 5-szörös keresztvalidációval hangoltuk; a teszthalmazt egyszer, a legvégén használtuk.
