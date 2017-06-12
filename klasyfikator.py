from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import os
from pandas import DataFrame

#Ścieżka do katalogu z danymi
sciezka_katalogu = "C:/dane/"


def main():
    print("Wczytywanie Danych")
    dane = wczytaj_dane()

    print("Uczenie")
    dane_uczace, dane_testowe = train_test_split(dane, test_size=.2, random_state=0)
    vectorizer = TfidfVectorizer(norm='l1')
    #Linear SVC
    linear_svc = LinearSVC(C=1.0)
    #Obliczanie częstotliwości występowania słów wykorzystując narzędzie Tfidf
    zliczenia = vectorizer.fit_transform(dane_uczace['text'].values)
    targets = dane_uczace['klasa'].values
    linear_svc.fit(zliczenia, targets)

    print("Walidacja")
    zliczenia = vectorizer.transform(dane_testowe['text'].values)
    predykcje = linear_svc.predict(zliczenia)

    print('Maciez pomyłek:')
    print(confusion_matrix(dane_testowe['klasa'].values, predykcje))


def wczytaj_dane():
    dane = DataFrame({'text': [], 'klasa': []})
    paths = os.path.abspath(sciezka_katalogu)
    sciezki = []
    for name in os.listdir(paths):
        path_name = os.path.join(paths, name)
        if os.path.isdir(path_name):
            sciezki.append(name)
    for sciezka in sciezki:
        klasa = sciezka
        sciezka_folderu = os.path.join(sciezka_katalogu, sciezka)
        wiersze = []
        index = []
        for nazwa_pliku, text in wczytaj_pliki(sciezka_folderu):
            wiersze.append({'text': text, 'klasa': klasa})
            index.append(nazwa_pliku)
        dane = dane.append(DataFrame(wiersze, index=index))
    return dane


def wczytaj_pliki(sciezka):
    for scr, nazwy_katlagow, nazwy_plikow in os.walk(sciezka):
        for plik in nazwy_plikow:
            sciezka_pliku = os.path.join(scr, plik)
            if os.path.isfile(sciezka_pliku):
                with open(sciezka_pliku, "r", encoding="latin-1") as file:
                    text = file.read()
                yield sciezka_pliku, text


if __name__ == "__main__":
    main()




