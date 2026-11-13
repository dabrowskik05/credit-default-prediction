# Predykcja niewypłacalności klientów kart kredytowych

*English:* Course project for Introduction to Machine Learning at Warsaw University of Technology (winter 2026/27). The task is to predict the probability that a credit card client will default on the next payment, using a modified version of the UCI "Default of Credit Card Clients" dataset. So far: EDA and a tuned decision tree (ROC AUC 0.754 on the validation set). Pipeline, other model families and the final model will be added in the next milestones. The rest of the documentation, the notebooks and the reports are in Polish.

Projekt na Wstęp do Uczenia Maszynowego (PW, semestr zimowy 2026/27). Zadanie to klasyfikacja binarna: na podstawie danych klienta i historii spłat z ostatnich 6 miesięcy trzeba przewidzieć prawdopodobieństwo, że klient nie spłaci zadłużenia w kolejnym miesiącu (klasa 1). Wynikiem końcowym jest wektor prawdopodobieństw dla zbioru testowego.

## Dane

Dane dostaliśmy na zajęciach. Są to przerobione dane [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) z UCI.

| plik | opis |
|---|---|
| `data/X_train.csv` | 6400 obserwacji, 23 cechy |
| `data/y_train.csv` | etykiety (ok. 78% klasy 0, 22% klasy 1) |
| `data/X_test.csv` | zbiór testowy bez etykiet |

Cechy:

| zmienna | opis |
|---|---|
| `LIMIT_BAL` | limit kredytowy |
| `SEX` | płeć (1 = mężczyzna, 2 = kobieta) |
| `EDUCATION` | wykształcenie (1 = magister/doktor, 2 = licencjat, 3 = średnie, 0/4/5/6 = inne) |
| `MARRIAGE` | stan cywilny (1 = w związku małżeńskim, 2 = stan wolny, 0/3 = inne) |
| `AGE` | wiek |
| `PAY_0`, `PAY_2` ... `PAY_6` | status spłaty w kolejnych miesiącach (`PAY_0` najnowszy); im wyżej, tym większe opóźnienie |
| `BILL_AMT1` ... `BILL_AMT6` | kwota wyciągu w kolejnych miesiącach (`BILL_AMT1` najnowszy) |
| `PAY_AMT1` ... `PAY_AMT6` | kwota wpłaty w kolejnych miesiącach (`PAY_AMT1` najnowsza) |

## Etap 1: eksploracja danych i drzewo decyzyjne

Notebook: `01_eda_drzewo_decyzyjne.ipynb`, raport: `raporty/raport_km1.pdf`.

Co wyszło z EDA:

- Kwoty (`LIMIT_BAL`, `BILL_AMT*`, `PAY_AMT*`) są mocno prawoskośne, stąd prawie 10% outlierów według IQR. To raczej długi ogon niż błędy.
- Braki są tylko w `PAY_5`, `BILL_AMT6` i `PAY_AMT4` (w każdej ok. 10%, w `X_test` też). Odsetek klasy 1 w wierszach z brakami i bez jest podobny, więc braki wyglądają na losowe.
- Prawdopodobne błędy w danych: jeden ujemny `LIMIT_BAL` i jedna wartość -3 w `PAY_3`. Kategorie "inne" w `EDUCATION` i `MARRIAGE` są rzadkie i rozbite na kilka kodów.
- Najwięcej informacji o targecie niosą statusy spłat `PAY_*`, szczególnie `PAY_0`, i do tego `LIMIT_BAL` (sprawdzone korelacją i mutual information). Dla `PAY_0 <= 0` odsetek klasy 1 jest poniżej 0.2, dla `PAY_0 >= 2` wynosi 0.4-0.7. `BILL_AMT*` są silnie skorelowane między sobą i mało mówią o targecie.

Metryka i podział:

- Główna metryka to ROC AUC, bo model ma zwracać prawdopodobieństwa, a AUC nie zależy od progu. Pomocniczo używam Average Precision, bo skupia się na rzadszej klasie 1. Accuracy odpada: model zawsze zwracający 0 miałby ok. 0.78.
- `X_train` podzieliłem stratyfikowanie 80/20. Na 80% stroję hiperparametry 5-krotną stratyfikowaną CV, a 20% jest zbiorem walidacyjnym użytym raz na końcu. Seed wszędzie to 67.

Strojenie drzewa (`GridSearchCV`, 252 kombinacje):

- `max_depth`: 2-8, 10, 12, 15, 20, 25, 30, None
- `min_samples_leaf`: 1, 5, 10, 20, 50, 100, 200, 300, 500
- `class_weight`: None, balanced

Na tym etapie drzewo dostaje surowe dane, bez imputacji i bez usuwania outlierów (`DecisionTreeClassifier` w sklearn obsługuje braki).

Wyniki:

| | drzewo domyślne | wybrane drzewo |
|---|---|---|
| hiperparametry | głębokość 34, 709 liści | `max_depth=6`, `min_samples_leaf=100`, `class_weight=balanced`, 20 liści |
| AUC trening | 1.000 | 0.771 |
| AUC CV | 0.600 | 0.735 |
| AUC walidacja | 0.616 | 0.754 |
| AP walidacja | 0.283 | 0.495 |

Domyślne drzewo jest mocno przeuczone. Najwięcej dało ograniczenie `min_samples_leaf`: przy liściach 100-300 wynik jest stabilny niezależnie od głębokości. `class_weight` prawie nic nie zmienia (0.735 vs 0.733 w CV), bo AUC ocenia tylko kolejność klientów. Najlepsze kombinacje różnią się o ułamki tysięcznych, mniej niż odchylenie między foldami (ok. 0.02). Wybrane drzewo opiera się głównie na `PAY_0` (ważność ponad 0.6) i `PAY_3` (ok. 0.17).

Do poprawienia w kolejnym etapie: imputacja braków, połączenie rzadkich kodów w `EDUCATION` i `MARRIAGE`, obsługa błędnych wartości. `class_weight=balanced` zawyża prawdopodobieństwa, co nie przeszkadza AUC, ale może mieć znaczenie przy końcowej ocenie.

## Etap 2: pipeline i porównanie modeli

W trakcie. Pipeline poprawiający problemy znalezione w etapie 1, ponownie strojone drzewo decyzyjne i model z innej rodziny, porównane na tej samej metryce.

## Etap 3: model końcowy

Inżynieria cech, bardziej zaawansowane modele, ocena końcowego modelu (macierz pomyłek, krzywe ROC i PR, analiza błędów) i predykcje dla `X_test`.

## Uruchomienie

Python 3.13, wersje bibliotek są w `requirements.txt`.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook
```

Notebooki trzeba uruchamiać z głównego folderu repo, bo wczytują dane z `data/`.
