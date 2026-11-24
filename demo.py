"""Élő demó: egy MRI kép osztályozása a betanított SVM modellel.

Használat:  .venv/bin/python demo.py agyikepek_4_osztaly/Testing/meningioma/Te-me_0010.jpg
(Előtte egyszer le kell futtatni a projekt.py-t, az menti el a modellt.)
"""
import sys

import joblib

from projekt import OSZTALYOK, elofeldolgoz, jellemzok

modell = joblib.load("eredmenyek/svm_modell.joblib")
for ut in sys.argv[1:]:
    X = jellemzok(elofeldolgoz(ut)).reshape(1, -1)
    print(f"{ut}  ->  {OSZTALYOK[modell.predict(X)[0]]}")
