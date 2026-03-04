<div align="center">
  <h3>Departamentul Automatică şi Tehnologia Informației</h3>
  <h1>Proiect PS</h1>
  <h2>Sistem de Detecție și Numărare a Persoanelor folosind Procesare de Semnal (DSP) și Inteligență Artificială cu Tracking</h2>
</div>

<br>

**Student:** Moroșanu Răzvan (Gr. 4LF342A, TI)  
**Profesor coordonator:** Dr. ing. Bogdan Sibișan Trăsnea  
**Locație / An:** BRAȘOV, 2025-2026  

---

## Cuprins
- [Capitolul 1. Introducere](#capitolul-1-introducere)
- [Capitolul 2. Fundamente Teoretice](#capitolul-2-fundamente-teoretice)
- [Capitolul 3. Arhitectura Sistemului Propus](#capitolul-3-arhitectura-sistemului-propus)
- [Capitolul 4. Implementare Software](#capitolul-4-implementare-software)
- [Capitolul 5. Rezultate Experimentale](#capitolul-5-rezultate-experimentale)
- [Capitolul 6. Concluzii](#capitolul-6-concluzii)
- [Bibliografie](#bibliografie)

---

## CAPITOLUL 1. INTRODUCERE

### 1.1 Contextul General
În contextul actual al dezvoltării sistemelor de supraveghere inteligentă și analiză biometrică, detecția precisă a persoanelor în fluxuri video reprezintă o provocare fundamentală. Orașele inteligente și industria de retail se bazează tot mai mult pe automatizarea monitorizării fluxurilor de oameni pentru optimizarea spațiilor și asigurarea securității. Deși operatorii umani sunt capabili să distingă persoane în condiții dificile, atenția acestora scade rapid în timp, făcând necesară implementarea unor soluții automate.

Prezenta lucrare propune proiectarea și implementarea unui **Sistem Hibrid de Monitorizare**, care integrează tehnici fundamentale de Procesare Digitală a Semnalelor (DSP) cu arhitecturi avansate de rețele neurale convoluționale (CNN).

### 1.2 Definirea Problemei
Deși algoritmii moderni de Deep Learning au atins performanțe notabile, aceștia depind în mod critic de calitatea semnalului de intrare. Imaginile captate de camere web standard suferă adesea de zgomot, gamă dinamică redusă și artefacte de compresie. Factori precum iluminarea slabă, contralumina sau prezența obiectelor cu forme antropomorfe (ex: scaune de birou cu haine pe ele) pot duce la clasificări eronate:
* **Fals Pozitiv:** Identificarea unui obiect neînsuflețit ca fiind o persoană.
* **Fals Negativ:** Omiterea unei persoane aflate într-o zonă umbrită.

### 1.3 Obiectivele Proiectului
Obiectivele specifice ale proiectului sunt:
1. **Optimizarea semnalului video:** Implementarea unui etaj de pre-procesare (DSP) care să compenseze neliniaritățile senzorului optic și să îmbunătățească contrastul local înainte de inferența AI.
2. **Reducerea ratei de Fals-Pozitiv:** Eliminarea detecțiilor eronate prin utilizarea unui model YOLOv8 Medium și a pragurilor de încredere adaptive.
3. **Eficiență computațională:** Realizarea unui sistem capabil să ruleze în timp real pe hardware de uz general (CPU), folosind tabele de căutare (LUT) pentru optimizarea calculelor matematice.
4. **Numărare automată și Vizualizare:** Implementarea logicii de cuantificare a subiecților și afișarea datelor printr-o interfață grafică informativă.

### 1.4 Structura Lucrării
Lucrarea este structurată în șase capitole. După introducere, Capitolul 2 prezintă fundamentele teoretice ale imaginii digitale și rețelelor neurale. Capitolul 3 detaliază arhitectura sistemului, iar Capitolul 4 explică implementarea software specifică în Python. Capitolul 5 analizează rezultatele obținute în diverse scenarii, iar lucrarea se încheie cu concluziile și direcțiile viitoare de dezvoltare.

---

## CAPITOLUL 2. FUNDAMENTE TEORETICE

### 2.1 Procesarea Digitală a Semnalelor (DSP) în Imagini

#### 2.1.1 Natura Discretă a Imaginii: Pixelul și Rezoluția
În procesarea digitală, o imagine nu este un mediu continuu, ci o reprezentare discretă a realității vizuale, obținută prin două procese fundamentale: **eșantionare** (discretizare spațială) și **cuantizare** (discretizare valorică a intensității).

**a) Pixelul (Picture Element)** Termenul "pixel" provine din contracția expresiei englezești *Picture Element* și reprezintă cea mai mică unitate indivizibilă a unei imagini digitale raster. Din punct de vedere matematic, într-o imagine definită ca o funcție $f(x,y)$, pixelul este valoarea funcției într-un punct de coordonate întregi $(x,y)$.

În formatul BGR utilizat de OpenCV, un pixel este un vector tridimensional:
$$P(x, y) = [B(x, y), G(x, y), R(x, y)]$$
Unde fiecare componentă $B, G, R \in [0, 255]$.  
Numărul total de stări posibile pentru un singur pixel este de aproximativ 16.7 milioane de culori ($256^3$).

**b) Rezoluția Spațială** Rezoluția se referă la densitatea pixelilor care compun imaginea, definită ca produsul $M \times N$. În Computer Vision, rezoluția introduce un compromis critic:
* Rezoluțiile mari (1080p) oferă detalii dar cresc latența, având nevoie de putere de procesare pentru a rula la un frame rate decent.
* Rezoluțiile medii (640x480), permit procesarea în timp real la un frame rate bun pe un CPU performant. 

#### 2.1.2 Histograma Imaginii și Egalizarea Adaptivă (CLAHE)
Histograma unei imagini digitale cu nivele de gri în intervalul $[0, L-1]$ este o funcție discretă care descrie probabilitatea apariției unui nivel de intensitate:
$$p(r_k) = \frac{n_k}{MN}, \quad k = 0, 1, \dots, L-1$$
Unde $r_k$ este al $k$-lea nivel de intensitate, $n_k$ este numărul de pixeli cu această intensitate, iar produsul $M \times N$ reprezintă numărul total de pixeli.

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** este o rafinare a egalizării standard. În loc să proceseze întreaga imagine global, CLAHE împarte imaginea în regiuni mici ("tiles", de exemplu $8 \times 8$) și egalizează histograma local. Parametrul "Clip Limit" este crucial deoarece limitează panta funcției de transformare pentru a preveni accentuarea zgomotului.

#### 2.1.3 Corecția Gamma și Percepția Neliniară
Senzorii camerelor au un răspuns liniar la lumină, dar ochiul uman percepe luminozitatea logaritmic. Corecția Gamma este o operație neliniară definită de relația teoretică:
$$V_{out} = A V_{in}^\gamma$$



> *Fig 1. Graficul Curbei Gamma*

În procesarea noastră (din codul sursă), dorim să deschidem zonele întunecate. De aceea, aplicăm o transformare inversă pentru corecție, calculând noua valoare a pixelului pe baza celei vechi:
$$P_{nou} = 255 \left( \frac{P_{vechi}}{255} \right)^{\frac{1}{\gamma}}$$
Unde o valoare $\gamma > 1$ (în cazul nostru 1.2) va expanda zona tonurilor întunecate, făcând fețele umane mai vizibile în condiții de lumină slabă.

#### 2.1.4 Spațiul de Culoare CIE LAB
Spre deosebire de RGB, spațiul **CIE Lab** este proiectat să aproximeze vederea umană. Acesta separă complet componenta de **Luminanță (L)** de informația cromatică **(a, b)**.

Formula standard pentru obținerea luminanței din RGB (simplificată) este:
$$L \approx 0.299 R + 0.587 G + 0.114 B$$



[Image of CIE LAB color space vs RGB]


> *Fig 2. Spațiul LAB vs RGB*

Această separare permite aplicarea filtrelor de contrast (CLAHE) doar pe canalul $L$, lăsând canalele $a$ și $b$ nealterate. Dacă am aplica egalizarea pe RGB, culorile s-ar distorsiona semnificativ.

### 2.2 Inteligența Artificială și Computer Vision

#### 2.2.1 Machine Learning vs. Deep Learning
* **Machine Learning (Clasic):** Se bazează pe extragerea manuală a trăsăturilor (feature engineering). Algoritmi precum HOG (Histogram of Oriented Gradients) necesită ca programatorul să definească ce înseamnă o "margine".
* **Deep Learning:** Utilizează rețele neurale cu multe straturi care învață automat trăsăturile direct din datele brute. Nu necesită definirea manuală a formelor, fiind mult mai robuste în scenarii complexe. Proiectul de față utilizează această paradigmă prin YOLOv8.

#### 2.2.2 Rețele Neurale Convoluționale (CNN)
CNN-urile folosesc operația matematică de convoluție. Pentru o imagine de intrare $I$ și un filtru (kernel) $K$, convoluția discretă bidimensională este definită ca:
$$(I * K)(i,j) = \sum_m \sum_n I(m,n) K(i-m, j-n)$$



> *Fig 3. Procesul de convoluție*

Această operație permite rețelei să detecteze margini, texturi și forme complexe, fundamentale pentru identificarea persoanelor.

#### 2.2.3 Arhitectura YOLOv8
YOLO (You Only Look Once) este un detector de tip "single-stage". Spre deosebire de alte arhitecturi (R-CNN) care scanează imaginea de mai multe ori, YOLO împarte imaginea într-o grilă și prezice coordonatele (bounding boxes) și probabilitățile de clasă simultan.

Metrica principală de evaluare este **IoU (Intersection over Union)**:
$$IoU = \frac{Aria(B_{pred} \cap B_{gt})}{Aria(B_{pred} \cup B_{gt})}$$



> *Fig 4. Arhitectura YOLO*

### 2.3 Constrângeri de Sistem - Latența
**Latența sistemului** este suma timpilor de execuție pentru captură, pre-procesare DSP, inferență AI și afișare:
$$L_{total} = t_{captură} + t_{DSP} + t_{AI} + t_{display}$$

Pentru funcționarea "Real-Time", este necesar ca frecvența cadrelor (FPS) să respecte condiția minimă (ex. 15 FPS):
$$FPS = \frac{1}{L_{total}} \ge 15$$

---

## CAPITOLUL 3. ARHITECTURA SISTEMULUI PROPUS

### 3.1 Schema Bloc a Sistemului
Sistemul este structurat pe un pipeline de procesare secvențială a datelor. Fluxul de informație este unidirecțional, de la senzorul optic către interfața cu utilizatorul.



### 3.2 Modulul de Achiziție (Input)
Sistemul preia fluxul video folosind biblioteca OpenCV (`cv2.VideoCapture`). Imaginile sunt extrase în format BGR. Rezoluția de intrare este dinamică, fiind adaptată la capacitățile camerei conectate (implicit 640x480 sau 1080p).

### 3.3 Modulul de Pre-procesare Hibrid (DSP)
Acesta este elementul inovator al proiectului. Rolul său este de a "curăța" imaginea înainte de a ajunge la inteligența artificială.
* **Corecția Gamma:** Expandarea zonelor întunecate folosind LUT.
* **Conversie LAB:** Separarea informației de luminozitate.
* **Egalizare Locală:** Aplicarea CLAHE cu parametri conservatori (`clipLimit=1.1`) pentru a nu introduce zgomot.

### 3.4 Modulul de Detecție (AI Core)
Semnalul procesat este introdus în rețeaua **YOLOv8 Medium**. Această variantă a modelului a fost aleasă pentru echilibrul dintre precizie (mAP) și viteza de inferență pe CPU. Se aplică un filtru de clasă (`class_id=0` pentru Persoane) și un prag de încredere minim de `0.75`.

---

## CAPITOLUL 4. IMPLEMENTARE SOFTWARE

### 4.1 Mediul de Dezvoltare
Aplicația a fost dezvoltată în limbajul **Python 3.13** (PyCharm), folosind bibliotecile **OpenCV** pentru DSP și **Ultralytics** pentru componenta de Deep Learning.

### 4.2 Algoritmul de Corecție Gamma (LUT)
O componentă esențială a proiectului este funcția `adjust_gamma`. Deoarece operația de ridicare la putere este costisitoare computațional dacă este aplicată individual pentru fiecare pixel, am implementat o optimizare prin **Look-Up Table (LUT)**. Această tehnică pre-calculează valorile transformate pentru toate cele 256 de nivele de intensitate posibile.

**Secvență din codul sursă:**
```python
# Inserează aici codul pentru funcția adjust_gamma
# Exemplu:
# def adjust_gamma(image, gamma=1.0):
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(image, table)
