# Dyktowanie opisów radiologicznych w języku polskim: zbiór testowy, porównanie lokalnych modeli ASR i architektura systemu wpisującego tekst w dowolnej aplikacji

**Autor:** [Imię Nazwisko]  ·  **Wersja robocza:** 0.1 (2026-11-16)  ·  **Status:** praca w toku. Wyniki oznaczone jako *wstępne* zostaną zastąpione pomiarami z pełnego benchmarku.

---

## Streszczenie

Polscy radiolodzy tworzą opisy badań w systemach RIS, często przez sesje zdalne (Citrix/RDP), a dostępne
narzędzia do rozpoznawania mowy są albo przestarzałe, albo działają we własnym oknie, albo przetwarzają
głos w chmurze. W pracy zaprojektowano system dyktowania, który wpisuje rozpoznany tekst w miejscu kursora
w dowolnej aplikacji i może działać w pełni lokalnie. Zbadano, czy do tego celu wystarcza gotowy model
rozpoznawania mowy (ASR), czy potrzebne jest dostrojenie (fine-tuning) albo własny model.

Wkład pracy:
1. **Odtwarzalny zbiór testowy polskich opisów radiologicznych.** Z publicznego korpusu PARROT wydzielono
   829 opisów i 156-elementowy split testowy (21,4 tys. słów, ok. 2,75 h dyktowania), losowany warstwowo.
   Wykryto i naprawiono wcześniej nieopisany artefakt korpusu: ciąg „dt” zamieniony na „RX” w 51 opisach.
2. **Werbalizator** przekształcający zapis opisu w formę mówioną (polskie liczebniki z odmianą przez
   przypadki, jednostki, wymiary, skróty, dyktowana interpunkcja). Błędy w ręcznej kontroli jakości:
   6,7% zdań ogółem i ok. 22% wśród zdań z liczbami lub skrótami.
3. **Harness ewaluacyjny z metrykami istotnymi klinicznie.** Oprócz WER/CER mierzy błędy na terminach
   medycznych, liczbach i wymiarach, stronie ciała (lewa/prawa) oraz zaprzeczeniach.
4. **Porównanie ośmiu konfiguracji lokalnych modeli ASR** (Whisper large-v3, large-v3-turbo, medium,
   medium dostrojony medycznie, NVIDIA Parakeet-TDT v3, Canary-1b-v2), z pomiarem jakości i opóźnienia na GPU i CPU.
5. **Architektura i prototyp aplikacji** (Rust, docelowo Tauri). Obejmuje wielopoziomową strategię
   wstawiania tekstu do cudzych aplikacji, w tym w środowiskach zdalnych, oraz obsługę mikrofonów
   dyktafonowych przez surowy HID.
6. **Analiza rynku, regulacji (MDR, AI Act, RODO, KSC) i strategii pozyskania danych** treningowych.

Wstępny wynik: model ogólny **Whisper large-v3-turbo** osiąga na testowych zdaniach radiologicznych
WER ok. 12–14% po normalizacji, bez błędów w liczbach, przy medianie opóźnienia ok. 1,1 s na GPU.
Model dostrojony na polskiej mowie medycznej (ADMEDVOICE) poprawia wyniki modelu bazowego medium
(WER 19,4 → 16,1%), ale nie przewyższa większego modelu ogólnego. Bez GPU Whisper jest za wolny
(p90 ok. 13 s na zdanie). Lżejszy Parakeet jest ok. 30× szybszy, ale wyraźnie gorszy na terminologii.

---

## 1. Wprowadzenie

### 1.1 Problem
Opis badania obrazowego (TK, MR, RTG, USG) to gęsty tekst specjalistyczny. Zawiera łacińską i polską
terminologię, wymiary („19 x 16 x 20 mm”), oznaczenia kręgów (L4/L5, Th12), skróty („śr.”, „wym.”, „i.v.”)
i stałe frazy szablonowe. Radiolog opisuje kilkadziesiąt badań dziennie, więc szybkość wprowadzania tekstu
bezpośrednio przekłada się na wydajność pracy. Dyktowanie wymaga jednak, aby:
- rozpoznawanie było dokładne na terminologii i liczbach, bo błąd w wymiarze, stronie ciała albo zaprzeczeniu
  jest błędem klinicznym;
- tekst trafiał tam, gdzie lekarz pracuje (pole opisu w RIS), bez kopiowania między oknami;
- dane pacjenta (głos, treść) nie opuszczały placówki, jeśli tego wymaga jej polityka bezpieczeństwa.

### 1.2 Pytania badawcze
- **P1.** Czy dostępne modele ASR (otwarte i lokalne) osiągają jakość wystarczającą dla polskich opisów radiologicznych?
- **P2.** Ile dają tanie metody adaptacji (podpowiedzi słownikowe, post-processing) w porównaniu z dostrojeniem modelu?
- **P3.** Jakie dane są potrzebne do dostrojenia i skąd je legalnie pozyskać?
- **P4.** Jak niezawodnie wpisywać tekst w dowolnej aplikacji, także w sesjach zdalnych?
- **P5.** Jakie wymagania regulacyjne determinują architekturę (lokalnie czy w chmurze, zakres użycia LLM)?

---

## 2. Przegląd istniejących rozwiązań i zasobów

### 2.1 Zasoby danych dla języka polskiego
| Zasób | Zawartość | Licencja | Uwagi |
|---|---|---|---|
| **PARROT v1.0** | Fikcyjne opisy radiologiczne pisane przez radiologów, wielojęzyczne; 837 polskich | CC BY-NC-SA 4.0 | Wszystkie polskie opisy pochodzą od jednego zespołu (Wrocław), więc reprezentują jeden styl. Wolno je testować, ale trenować model komercyjny tylko za zgodą autorów |
| **ADMEDVOICE** | Polska mowa medyczna: część naturalna (~15 h, 28 mówców), anonimizowana (~58 h), syntetyczna (~11 h); 52 989 nagrań, ~30 GB, kolumna `category` | CC BY 4.0 | Czytane krótkie frazy, a nie płynne dyktowanie. Dostęp na HF wymaga ręcznej akceptacji warunków |
| `whisper-medium-medical-pl` | Whisper medium dostrojony LoRA na ADMEDVOICE (WER 24,03 → 13,91 wg autorów) | Apache 2.0 | Wersja ggml dla whisper.cpp |
| Korpus TheLion.ai | 1200 opisów TK klatki z adnotacjami (BSNLP 2023) | niepubliczny | Wymaga kontaktu z autorami |
| AI4AR (OPI) | MR prostaty | — | Zweryfikowano, że zawiera tylko deskryptory PI-RADS, bez tekstowych opisów |
| BIGOS / PELCRA | Ogólny benchmark polskiego ASR | różne | Do kontroli, czy dostrojenie nie pogarsza ogólnej polszczyzny |

### 2.2 Rozwiązania rynkowe (szczegóły w `product/REPORT.md` §1)
Na rynku polskim działają trzy grupy rozwiązań:
- **systemy starszej generacji** (MagicSpeech/Radcomp, Newton Dictate): lokalne, sprzedawane instytucjom;
- **narzędzia przeglądarkowe z LLM** (Opisy.AI): wynik trzeba ręcznie przenieść do RIS;
- **silniki ASR bez aplikacji** (Medalion VoxMedic, deklarowany WER 7,64% bez opisanego zbioru testowego).

Światowi liderzy (Dragon Medical One, PowerScribe, Rad AI) nie publikują wsparcia dla polskiego.
**Luka:** brak rozwiązania, które łączy pracę lokalną, wpisywanie przy kursorze w każdej aplikacji
(także w Citrixie), nowoczesny ASR i zakup przez pojedynczego lekarza.

---

## 3. Dane

### 3.1 Konstrukcja zbioru testowego z PARROT
Skrypt `scripts/data/build_parrot.py` (deterministyczny, seed 42):
1. **Filtr i deduplikacja.** Filtr `language == "Polish"` daje 837 opisów. Dokładnych duplikatów jest 3,
   po `strip()` 4, a po normalizacji białych znaków 8. Usunięcie ostatnich daje **829** opisów.
2. **Mapowanie modalności** (RX→RTG, CT→TK, US→USG) i grupowanie okolic ciała w 6 grup:
   klatka, jama brzuszna i miednica, klatka z jamą brzuszną, głowa i szyja, kręgosłup, kończyny i inne.
   Po deduplikacji: TK 357, RTG 280, USG 139, MR 39, XA 11, endoskopia 2, PET-TK 1.
3. **Wykluczenia z ewaluacji:** 38 opisów echokardiografii (nie radiologia) oraz prawie-duplikaty
   (Jaccard ≥ 0,6 na trigramach słów: 23 pary). Z każdej grupy prawie-duplikatów do test/dev trafia
   najwyżej jeden opis, a 6 opisów z części treningowej oznaczono jako „przeciek”, którego nie wolno użyć do treningu.
4. **Split warstwowy** (modalność × grupa okolic, długość 15–400 słów, MR uzupełnione do 15):

| Split | Opisy | Słowa | Czas czytania (130 sł./min) |
|---|---|---|---|
| test | 156 (TK 67, RTG 54, USG 20, MR 15) | 21 413 | ~2,75 h |
| dev | 40 | 4 700 | ~0,6 h |
| rest | 633 | — | — |

### 3.2 Artefakt korpusu PARROT (nowe odkrycie)
W polskiej części PARROT ciąg „dt” został systematycznie zamieniony na „RX”, prawdopodobnie przez globalne
zastąpienie podczas ujednolicania kodów modalności. Przykłady: „ponaRXo” zamiast „ponadto”,
„poRXorebkowo” zamiast „podtorebkowo”. Artefakt występuje **60 razy w 51 opisach** i zaniża wyniki każdego
modelu ASR ewaluowanego na surowym tekście. Poprawiona wersja znajduje się w kolumnie `report_clean`.
*Wart zgłoszenia autorom korpusu.*

### 3.3 Słownik terminów medycznych
2 653 terminy wyznaczone statystycznie, a nie ręcznie. Częstość słowa w opisach porównano z ogólną
polszczyzną (`wordfreq`, skala Zipfa) i za termin uznano słowo o keyness ≥ 2,5 albo rzadkie w ogólnej
polszczyźnie (Zipf ≤ 2,5). Do metryki „błąd na terminach” używany jest tier A (2 520 pozycji).

### 3.4 Formy wymagające normalizacji
Ekstrakcja regexami (`data/normalization_forms.tsv`) objęła m.in.: wymiary (342 wystąpienia), wartości
z jednostkami (~2 400), oznaczenia kręgów (~280), cyfry rzymskie (~280), skróty z kropką (~1 370)
i skrótowce (TK, DWI, AP). Te formy decydują o różnicy między tym, co radiolog mówi, a tym, co ma się
pojawić w opisie.

### 3.5 Werbalizacja: forma pisana → forma mówiona
Ponieważ brak publicznych nagrań dyktowania całych opisów, audio testowe syntetyzowano z formy mówionej.
Werbalizator regułowy (`scripts/data/verbalize.py`, v1) obejmuje:
- liczebniki w przypadku wymaganym przez przyimek, w tym koordynację: „do 40 mm … i 25 mm” → „do czterdziestu … i dwudziestu pięciu milimetrów”;
- wymiary: „19 x 16 mm” → „dziewiętnaście na szesnaście milimetrów”;
- skróty z kropką i z łącznikiem (p-ż → przeponowo-żebrowych) oraz literowanie skrótowców (TK → „te ka”);
- dwa warianty interpunkcji: **dyktowana** („kropka”, „przecinek”, „otwieram nawias”, „nowa linia”) i **ciągła** (bez interpunkcji).

**Kontrola jakości** na 30 losowych zdaniach (seed 42, ocena ręczna): **2/30 zdań błędnych (6,7%;
95% CI ≈ 1–22%)**. Wśród zdań z liczbami lub skrótami błędne były 2/9 (~22%). Szum dotyczy wyłącznie
audio syntetycznego, bo referencją do WER jest forma pisana. Literówki w test/dev poprawiono
(57 wystąpień), a oryginał zachowano.

---

## 4. Metodyka ewaluacji ASR

### 4.1 Metryki (`asr/eval/metrics.py`)
- **WER RAW i WER NORM.** Normalizacja jest jawna (`asr/eval/normalize.py`): wielkość liter,
  interpunkcja, a liczebniki słowne są sprowadzane do cyfr przez parser polskich liczebników
  we wszystkich przypadkach (`plnum.py`). Bez tego wyniki modeli zapisujących liczby słownie
  i cyframi nie są porównywalne.
- **CER.**
- **Błąd na terminach medycznych:** odsetek terminów z tieru A, których brak w hipotezie.
- **Błędy krytyczne:** liczby, wymiary, strona ciała (lew*/praw*/obustronn*), zaprzeczenia (nie/bez/brak),
  odsetek wypowiedzi z co najmniej jednym błędem krytycznym.
- **Wydajność:** RTF (czas przetwarzania / długość audio) oraz opóźnienie p50 i p90 na wypowiedź.

### 4.2 Audio testowe
Lokalne TTS: macOS „Zosia” i 4 polskie głosy Piper. **Ograniczenie:** syntetyczna mowa jest czystsza
niż mowa lekarza z mikrofonem w gabinecie, więc wyniki zawyżają jakość bezwzględną. Wnioski dotyczą
**porównania modeli między sobą**, a nie oczekiwanej dokładności produktu.

### 4.3 Środowisko
MacBook M1 Pro 16 GB; whisper.cpp 1.9.5 (Metal) przez `whisper-server`, beam 5, temperatura 0;
sherpa-onnx (Parakeet, int8, CPU); onnx-asr (Canary, int8, CPU).

---

## 5. Wyniki

### 5.1 Eksperyment wstępny (test harnessu)
15 zdań radiologicznych (1,7 min audio), wariant z dyktowaną interpunkcją. *Próba jest za mała na wnioski
statystyczne. W próbce są tylko 4 zaprzeczenia i 4 oznaczenia strony.*

| Model | WER RAW | WER NORM | CER | Terminy | Liczby | Zaprzecz. | Wypow. z bł. kryt. | RTF | p50 [s] | p90 [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| Whisper large-v3 q5_0 (GPU) | 37,6 | 16,1 | 7,7 | 18,8 | 4,8 | 25 | 20% | 0,30 | 2,12 | 2,36 |
| **Whisper large-v3-turbo q5_0 (GPU)** | 38,9 | 13,7 | 5,0 | 18,8 | 0 | 0 | 6,7% | 0,17 | 1,19 | 1,26 |
| **Whisper large-v3-turbo q8_0 (GPU)** | 37,1 | **11,9** | **3,8** | 18,8 | 0 | 0 | **0%** | 0,15 | 1,05 | 1,13 |
| Whisper large-v3-turbo q5_0 (CPU, 8 wątków) | 38,9 | 13,7 | 5,0 | 18,8 | 0 | 0 | 6,7% | 1,75 | 11,28 | 13,43 |
| Whisper medium q5_0 (GPU) | 38,9 | 19,4 | 7,3 | 34,4 | 4,8 | 25 | 20% | 0,17 | 1,22 | 1,38 |
| whisper-medium-medical-pl q5_0 (GPU) | 36,7 | 16,1 | 7,1 | 21,9 | 0 | 25 | 13% | 0,19 | 1,30 | 1,42 |
| NVIDIA Parakeet-TDT-0.6B-v3 int8 (CPU) | 40,7 | 26,1 | 10,4 | 40,6 | 9,5 | 25 | 13% | **0,04** | **0,26** | **0,34** |
| NVIDIA Canary-1b-v2 int8 (CPU) | 34,4 | 25,1 | 12,2 | 43,8 | 0 | 25 | 20% | 0,09 | 0,59 | 0,85 |

*Wartości w %, poza RTF i czasami.*

**Obserwacje:**
1. Duża różnica między WER RAW a NORM (np. 37 → 12%) pokazuje, że większość „błędów” to forma zapisu
   (interpunkcja, liczby słownie lub cyframi), a nie złe rozpoznanie. Uzasadnia to osobny moduł
   normalizacji odwrotnej (ITN), który zamienia zapis mówiony na pisany.
2. Model ogólny large-v3-turbo jest najlepszy jakościowo i szybszy od large-v3.
3. Dostrojenie medyczne poprawia model, na którym je wykonano (medium: terminy 34 → 22% błędów),
   ale nie dorównuje większemu modelowi ogólnemu. **Wniosek roboczy:** dostrajanie ma sens na mocniejszej
   bazie (turbo), a nie jako zamiennik większego modelu.
4. Opóźnienie na CPU: architektura Whisper zawsze koduje okno 30 s, więc nawet krótkie zdanie kosztuje
   ok. 11 s na CPU. To kluczowe ograniczenie dla stacji bez GPU.

### 5.2 Benchmark na zbiorze testowym — *w toku*
60 opisów ze splitu testowego (752 wypowiedzi, TTS z dyktowaną interpunkcją). Modele: turbo, large-v3,
medium, medium-medical-pl oraz turbo i medical-pl z podpowiedzią słownikową (initial prompt z 60 najczęstszych
terminów). Wyniki: `asr/results/v0/table.md` (zostaną wstawione tutaj).

### 5.3 Obserwacje z prototypu aplikacji
- Komendy głosowe są osobnym problemem: turbo zamienia „dwukropek” na „2.”, a medical-pl zniekształca
  komendy („wstew normy”). Podpowiedź kontekstowa tego nie naprawia. **Komendy muszą być częścią zbioru testowego.**
- Dla wypowiedzi 1–2 s dominuje stały narzut (RTF ≈ 1). Dzielenie dyktowania na porcje po pauzie (VAD)
  musi to uwzględniać.

---

## 6. Architektura systemu

### 6.1 Potok przetwarzania
```
mikrofon / SpeechMike (HID) → VAD (porcje po pauzie) → ASR lokalny (whisper.cpp)
   → ITN + komendy głosowe + makra (reguły) → wstawianie przy kursorze (z fallbackami)
```

### 6.2 Wstawianie tekstu do dowolnej aplikacji (`app/REPORT.md`)
Analiza kodu otwartych narzędzi (Handy, VoiceInk, OpenWhispr, enigo, nerd-dictation, whisper-writer)
i dokumentacji Dragon Medical One doprowadziła do strategii z kolejnymi fallbackami:
1. **API dostępności** (UI Automation / macOS AX): wstawienie w zaznaczenie z weryfikacją wartości pola;
2. **klawiatura Unicode** (SendInput `KEYEVENTF_UNICODE` / CGEvent), w porcjach ≤ 20 znaków;
3. **schowek + Ctrl/Cmd+V** z przywróceniem schowka po potwierdzeniu odczytu (delayed rendering),
   a nie po stałym opóźnieniu;
4. **„remote typing”**: fizyczne kody klawiszy według układu Polski programisty (AltGr dla ą, ę, ł…),
   przeznaczony dla Citrix i RDP. Istniejące narzędzia w tym trybie zamieniają „ł” na „l”, co dla polskiego
   jest niedopuszczalne.

Zasada bezpieczeństwa: gdy fokus zmienił się w trakcie dyktowania, tekst nie jest wpisywany, tylko trafia do schowka.

**Citrix/RDP to największe ryzyko.** Najpewniejsze rozwiązanie produkcyjne to uruchomienie aplikacji
wewnątrz sesji zdalnej z przekazaniem audio przez kanał HDX (tak działa Dragon Medical One).
Wpisywanie z klienta jest trybem awaryjnym.

### 6.3 Mikrofony dyktafonowe
Philips SpeechMike i Nuance PowerMic obsłużymy przez surowy HID, bez SDK i licencji. Protokół
odczytano z biblioteki Google `dictation_support` (Apache 2.0): VID 0x0911 i 0x0554, usage page 0xFFA0,
bajt 0 = 0x80, maska przycisków uint16 LE od bajtu 7.

### 6.4 Stos technologiczny
Rdzeń w Ruście (`cpal`, `whisper-rs`, `enigo`, `global-hotkey`, `tray-icon`), docelowo Tauri 2.
Zalety: jeden kod na Windows i macOS, ASR w tym samym procesie, instalator ok. 10 MB plus model,
instalacja MSI per-user bez praw administratora. Prototyp (`app/prototype/`) przetwarza audio przez
whisper.cpp (Metal) i post-processing; ma 9 testów jednostkowych.

---

## 7. Regulacje i dane (`product/REPORT.md` §3–4)
- **MDR.** Sama transkrypcja z przeznaczeniem „narzędzie do wprowadzania tekstu” najpewniej nie jest
  wyrobem medycznym. Generowanie wniosków lub sugerowanie rozpoznań przez LLM przekracza granicę
  (reguła 11: co najmniej klasa IIa). **Decyzja projektowa: w MVP żaden LLM nie zmienia treści klinicznej.**
- **AI Act.** Transkrypcja to system o minimalnym ryzyku.
- **RODO.** Wariant w pełni lokalny, bez telemetrii treści, nie przetwarza danych pacjentów po stronie
  dostawcy. Wariant chmurowy wymaga umowy powierzenia, DPIA, hostingu w EOG i spełnienia wymagań KSC.
- **Strategia danych:**
  - testy na PARROT (niekomercyjnie) i ADMEDVOICE;
  - dostrojenie na własnym, opłaconym korpusie dyktowania (radiolodzy dyktują fikcyjne opisy;
    30–50 h, ok. 40–60 tys. zł, pełne prawa);
  - lokalna pętla poprawek lekarza (słownik, reguły);
  - docelowo partnerstwo z firmą teleradiologiczną i anonimizacja po stronie administratora danych.

*Analiza nie stanowi opinii prawnej.*

---

## 8. Dyskusja i wnioski (stan na wersję 0.1)
- **P1:** Lokalny model ogólny (Whisper large-v3-turbo) daje obiecującą jakość na GPU, ale ocenę
  bezwzględną trzeba potwierdzić na nagraniach naturalnych.
- **P2:** Większość różnic w WER wynika z formy zapisu, więc **ITN i komendy głosowe są tańszą i pilniejszą
  dźwignią niż dostrojenie**. Efekt podpowiedzi słownikowej jest mierzony w benchmarku (§5.2).
- **P3:** Trenowanie własnego modelu od zera jest nieuzasadnione przy dostępnych ~15 h naturalnej mowy
  medycznej. Rekomendowana ścieżka to dostrojenie (LoRA) mocnej bazy na własnym korpusie dyktowania.
- **P4:** Wstawianie lokalne jest rozwiązywalne standardowymi API. Sesje zdalne wymagają osobnej
  architektury (aplikacja wewnątrz sesji).
- **P5:** Ograniczenie roli LLM do formy zapisu, a nie treści, utrzymuje produkt poza MDR i upraszcza wdrożenie.

## 9. Ograniczenia
- Audio testowe jest syntetyczne (TTS). Brak nagrań prawdziwego dyktowania.
- Teksty pochodzą od jednego zespołu radiologów (PARROT PL), więc nie reprezentują różnorodności stylów w kraju.
- Eksperyment wstępny jest mały (15 zdań). Pełny benchmark obejmuje 60 opisów.
- Pomiary CPU wykonano na Apple M1 Pro, a nie na typowej stacji z Windows.
- Silniki chmurowe nie zostały jeszcze zmierzone (wymagają kluczy API).

## 10. Dalsze prace
1. Wyniki pełnego benchmarku i wpływ podpowiedzi słownikowej.
2. Moduł ITN i jego ewaluacja na splicie dev, w tym ryzyko zmiany treści klinicznej.
3. Ewaluacja na naturalnej części ADMEDVOICE i na nagraniach radiologów (protokół w przygotowaniu: `product/recording/`).
4. Pomiar opóźnienia na Windows bez GPU (greedy decoding, skrócony kontekst audio, faster-whisper int8).
5. Dostrojenie LoRA turbo na korpusie dyktowania, z kontrolą degradacji ogólnej polszczyzny (BIGOS).
6. Test wstawiania tekstu w Citrix/RDP na prawdziwej infrastrukturze.

## 11. Metoda pracy
Projekt prowadzono jako zespół czterech równoległych agentów AI (Claude Code) o rozdzielonych rolach:
dane, rozpoznawanie mowy, aplikacja, produkt i regulacje. Pracą kierował autor jako nadzorca: podział
zadań (`briefs/`), koordynacja zależności (`status/`), kontrola jakości wyników i decyzje projektowe.
Wszystkie liczby w tej pracy pochodzą ze skryptów zapisanych w repozytorium i są odtwarzalne.

## 12. Odtwarzalność i struktura repozytorium
| Ścieżka | Zawartość |
|---|---|
| `scripts/data/` | budowa zbioru PARROT, słownik, formy normalizacji, werbalizator, QC |
| `data/` | zbiory wynikowe (`parrot_pl.*`, `test_spoken.jsonl`, `medical_terms.*`) i raport |
| `asr/eval/` | harness: normalizacja, metryki, adaptery silników, TTS, `bench_v0.sh` |
| `asr/results/` | wyniki pomiarów (CSV, hipotezy) |
| `app/prototype/` | prototyp aplikacji (Rust) |
| `product/` | raport rynkowo-regulacyjny, szkice korespondencji |
| `briefs/`, `status/`, `HANDOFF.md` | organizacja pracy |

## Bibliografia
1. PARROT v1.0 — https://github.com/PARROT-reports/PARROT_v1.0
2. ADMEDVOICE: polski korpus mowy medycznej. *Scientific Data* (2025). https://www.nature.com/articles/s41597-025-05776-1
3. Radford A. i in., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper), 2022.
4. whisper.cpp — https://github.com/ggml-org/whisper.cpp
5. NVIDIA Parakeet-TDT-0.6B-v3, Canary-1b-v2 — karty modeli na Hugging Face.
6. BIGOS — benchmark polskiego ASR, HF `amu-cai/pl-asr-bigos-v2`.
7. Google `dictation_support` — https://github.com/GoogleChromeLabs/dictation_support
8. Rozporządzenie (UE) 2017/745 (MDR); MDCG 2019-11; Rozporządzenie (UE) 2024/1689 (AI Act).
