
CAPITOLUL 1. INTRODUCERE
1.1 Contextul General
În contextul actual al dezvoltării sistemelor de supraveghere inteligentă și analiză biometrică, detecția precisă a persoanelor în fluxuri video reprezintă o provocare fundamentală. Orașele inteligente  și industria de retail se bazează tot mai mult pe automatizarea monitorizării fluxurilor de oameni pentru optimizarea spațiilor și asigurarea securității. Deși operatorii umani sunt capabili să distingă persoane în condiții dificile, atenția acestora scade rapid în timp, făcând necesară implementarea unor soluții automate.
Prezenta lucrare propune proiectarea și implementarea unui Sistem Hibrid de Monitorizare, care integrează tehnici fundamentale de Procesare Digitală a Semnalelor (DSP) cu arhitecturi avansate de rețele neurale convoluționale (CNN).
1.2 Definirea Problemei
Deși algoritmii moderni de Deep Learning au atins performanțe notabile, aceștia depind în mod critic de calitatea semnalului de intrare. Imaginile captate de camere web standard suferă adesea de zgomot, gamă dinamică redusă și artefacte de compresie. Factori precum iluminarea slabă, contralumina sau prezența obiectelor cu forme antropomorfe (ex: scaune de birou cu haine pe ele) pot duce la clasificări eronate:
Fals Pozitiv: Identificarea unui obiect neînsuflețit ca fiind o persoană.
Fals Negativ: Omiterea unei persoane aflate într-o zonă umbrită.
1.3 Obiectivele Proiectului
Obiectivele specifice ale proiectului sunt:
Optimizarea semnalului video: Implementarea unui etaj de pre-procesare (DSP) care să compenseze neliniaritățile senzorului optic și să îmbunătățească contrastul local înainte de inferența AI.
Reducerea ratei de Fals-Pozitiv: Eliminarea detecțiilor eronate prin utilizarea unui model YOLOv8 Medium și a pragurilor de încredere adaptive.
Eficiență computațională: Realizarea unui sistem capabil să ruleze în timp real pe hardware de uz general (CPU), folosind tabele de căutare (LUT) pentru optimizarea calculelor matematice.
Numărare automată și Vizualizare: Implementarea logicii de cuantificare a subiecților și afișarea datelor printr-o interfață grafică informativă.
1.4 Structura Lucrării
Lucrarea este structurată în șase capitole. După introducere, Capitolul 2 prezintă fundamentele teoretice ale imaginii digitale și rețelelor neurale. Capitolul 3 detaliază arhitectura sistemului, iar Capitolul 4 explică implementarea software specifică în Python. Capitolul 5 analizează rezultatele obținute în diverse scenarii, iar lucrarea se încheie cu concluziile și direcțiile viitoare de dezvoltare.

CAPITOLUL 2. FUNDAMENTE TEORETICE
2.1 Procesarea Digitală a Semnalelor (DSP) în Imagini
2.1.1 Natura Discretă a Imaginii: Pixelul și Rezoluția
În procesarea digitală, o imagine nu este un mediu continuu, ci o reprezentare discretă a realității vizuale, obținută prin două procese fundamentale: eșantionare (discretizare spațială) și cuantizare (discretizare valorică a intensității).
a) Pixelul (Picture Element) Termenul "pixel" provine din contracția expresiei englezești Picture Element și reprezintă cea mai mică unitate indivizibilă a unei imagini digitale raster. Din punct de vedere matematic, într-o imagine definită ca o funcție f(x,y), pixelul este valoarea funcției într-un punct de coordonate întregi (x,y).
În formatul BGR utilizat de OpenCV, un pixel este un vector tridimensional:
P(x, y) = 〖{B(x, y)  G(x, y),  R(x, y) 〗
Unde fiecare componentă B,G,R0,255. 
Numărul total de stări posibile pentru un singur pixel este milioane de culori.
b) Rezoluția Spațială Rezoluția se referă la densitatea pixelilor care compun imaginea, definită ca produsul M x N . În Computer Vision, rezoluția introduce un compromis critic:
Rezoluțiile mari (1080p) oferă detalii dar cresc latența, ai nevoie de putere de procesare pentru a rula la un frame rate decent.
Rezoluțiile medii (640x480), permit procesarea în timp real la un frame rate bun pe un CPU bun. 
2.1.2 Histograma Imaginii și Egalizarea Adaptivă (CLAHE)
Histograma unei imagini digitale cu nivele de gri în intervalul  este o funcție discretă care descrie probabilitatea apariției unui nivel de intensitate:
prk=nkMN, k=0,1,,L-1
Unde  rk este al k-lea nivel de intensitate, nk este numărul de pixeli cu această intensitate, iar produsul M x N reprezintă numărul total de pixeli.
CLAHE (Contrast Limited Adaptive Histogram Equalization) este o rafinare a egalizării standard. În loc să proceseze întreaga imagine global, CLAHE împarte imaginea în regiuni mici ("tiles", de exemplu ) și egalizează histograma local. Parametrul "Clip Limit" este crucial deoarece limitează panta funcției de transformare pentru a preveni accentuarea zgomotului.
2.1.3 Corecția Gamma și Percepția Neliniară
Senzorii camerelor au un răspuns liniar la lumină, dar ochiul uman percepe luminozitatea logaritmic. Corecția Gamma este o operație neliniară definită de relația teoretică:
Vout=AVin

           Fig 3. Graficul Curbei Gamma
În procesarea noastră (din codul sursă), dorim să deschidem zonele întunecate. De aceea, aplicăm o transformare inversă pentru corecție, calculând noua valoare a pixelului pe baza celei vechi:
Pnou=255Pvechi2551
Unde o valoare  (în cazul nostru 1.2) va expanda zona tonurilor întunecate, făcând fețele umane mai vizibile în condiții de lumină slabă.
2.1.4 Spațiul de Culoare CIE LAB
Spre deosebire de RGB, spațiul CIE Lab este proiectat să aproximeze vederea umană. Acesta separă complet componenta de Luminanță (L) de informația cromatică (a, b).
Formula standard pentru obținerea luminanței din RGB (simplificată):
L 0.299 R + 0.587 G + 0.114 B

Fig.4 Spațiul LAB vs RGB
Această separare permite aplicarea filtrelor de contrast (CLAHE) doar pe canalul , lăsând canalele  și  nealterate. Dacă am aplica egalizarea pe RGB, culorile s-ar distorsiona semnificativ.
2.2 Inteligența Artificială și Computer Vision
2.2.1 Machine Learning vs. Deep Learning
Machine Learning (Clasic): Se bazează pe extragerea manuală a trăsăturilor (feature engineering). Algoritmi precum HOG (Histogram of Oriented Gradients) necesită ca programatorul să definească ce înseamnă o "margine".
Deep Learning : Utilizează rețele neurale cu multe straturi care învață automat trăsăturile direct din datele brute. Nu necesită definirea manuală a formelor, fiind mult mai robuste în scenarii complexe. Proiectul de față utilizează această paradigmă prin YOLOv8.
2.2.2 Rețele Neurale Convoluționale (CNN)
CNN-urile folosesc operația matematică de convoluție. Pentru o imagine de intrare și un filtru (kernel), convoluția discretă bidimensională este definită ca:
I*Ki,j=mnIm,nKi-m,j-n

	                                    Fig 5 Procesul de convolutie
Această operație permite rețelei să detecteze margini, texturi și forme complexe, fundamentale pentru identificarea persoanelor.
2.2.3 Arhitectura YOLOv8
YOLO (You Only Look Once) este un detector de tip "single-stage". Spre deosebire de alte arhitecturi (R-CNN) care scanează imaginea de mai multe ori, YOLO împarte imaginea într-o grilă  și prezice coordonatele (bounding boxes) și probabilitățile de clasă simultan.
Metrica principală de evaluare este IoU (Intersection over Union):
IoU=AriaBpredBgtAriaBpredBgt


Fig 6. ARHITECTURA YOLO
2.3 Constrângeri de Sistem - Latența
Latența sistemului  este suma timpilor de execuție pentru captură, pre-procesare DSP, inferență AI și afișare:
Ltotal=tcaptură+tDSP+tAI+tdisplay
Pentru funcționarea "Real-Time", este necesar ca frecvența cadrelor (FPS) să respecte condiția:
FPS=1Ltotal15

CAPITOLUL 3. ARHITECTURA SISTEMULUI PROPUS
3.1 Schema Bloc a Sistemului
Sistemul este structurat pe un pipeline de procesare secvențială a datelor. Fluxul de informație este unidirecțional, de la senzorul optic către interfața cu utilizatorul.
  
3.2 Modulul de Achiziție (Input)
Sistemul preia fluxul video folosind biblioteca OpenCV (cv2.VideoCapture). Imaginile sunt extrase în format BGR. Rezoluția de intrare este dinamică, fiind adaptată la capacitățile camerei conectate (implicit 640x480 sau 1080p).
3.3 Modulul de Pre-procesare Hibrid (DSP)
Acesta este elementul inovator al proiectului. Rolul său este de a "curăța" imaginea înainte de a ajunge la inteligența artificială.
Corecția Gamma: Expandarea zonelor întunecate folosind LUT.
Conversie LAB: Separarea informației de luminozitate.
Egalizare Locală: Aplicarea CLAHE cu parametri conservatori (clipLimit=1.1) pentru a nu introduce zgomot.
3.4 Modulul de Detecție (AI Core)
Semnalul procesat este introdus în rețeaua YOLOv8 Medium. Această variantă a modelului a fost aleasă pentru echilibrul dintre precizie (mAP) și viteza de inferență pe CPU. Se aplică un filtru de clasă (class_id=0 pentru Persoane) și un prag de încredere 0.75.

CAPITOLUL 4. IMPLEMENTARE SOFTWARE
4.1 Mediul de Dezvoltare
Aplicația a fost dezvoltată în limbajul Python 3.13(PyCharm), folosind bibliotecile OpenCV pentru DSP și Ultralytics pentru componenta de Deep Learning.
4.2 Algoritmul de Corecție Gamma (LUT)
O componentă esențială a proiectului este funcția adjust_gamma. Deoarece operația de ridicare la putere este costisitoare computațional dacă este aplicată individual pentru fiecare pixel, am implementat o optimizare prin Look-Up Table (LUT). Această tehnică pre-calculează valorile transformate pentru toate cele 256 de nivele de intensitate posibile.
Secvență din codul sursă:
Python



In bucla principală, se apelează adjust_gamma [“frame_gamma = adjust_gamma(frame, gamma=1.2)”], ceea ce deschide umbrele din imagine instantaneu.
4.3 Implementarea Egalizării CLAHE în spațiul LAB
Pentru a îmbunătăți contrastul local, s-a utilizat algoritmul CLAHE. Codul implementează conversia spațiului de culoare pentru a proteja canalele cromatice.

Secvență din codul sursă:
Python






S-a redus parametrul clipLimit de la valoarea standard 2.0 la 1.1. Testele au arătat că o valoare prea mare introducea artefacte vizuale care scădeau precizia detecției.

4.4 Integrarea Modelului YOLOv8 și Filtrarea
Modelul este instanțiat folosind greutățile yolov8m.pt. Inferența se realizează pe imaginea procesată (frame_processed), nu pe cea brută.




Python
Parametrul conf=0.75 asigură că sistemul ignoră orice obiect despre care nu este sigur în proporție de minim 70%.
4.5 Interfața Grafică (Dashboard)
Pentru a oferi feedback vizual, s-a implementat un overlay semitransparent care afișează numărul de persoane și starea algoritmilor. Se folosește cv2.addWeighted pentru a crea transparența panoului informativ, iar culoarea textului se schimbă dinamic (Roșu/Verde) în funcție de prezența persoanelor în cadru.


CAPITOLUL 5. REZULTATE EXPERIMENTALE
5.1 Metodologia de Testare
Sistemul a fost testat într-un mediu de birou, variind condițiile de iluminare și obiectele din fundal. Metricile urmărite au fost FPS-ul (pentru performanță) și acuratețea vizuală a detecției.
5.2 Scenariul 1: Obiecte Confuze (Testul Scaunului)
S-a pus un scaun de birou cu o haină pe el în cadru.
Rezultat Standard (Fără filtrare): Rețeaua tindea să detecteze scaunul cu 40-50% încredere.
Rezultat Hibrid: Datorită pragului conf=0.75 și a detaliilor de textură evidențiate de CLAHE, rețeaua a clasificat corect obiectul ca nefiind o persoană. Rata de Fals-Pozitiv a fost 0.
5.3 Scenariul 2: Iluminare Slabă
S-a testat sistemul în condiții de semi-întuneric.
Efectul DSP: Corecția Gamma a recuperat detaliile feței care erau invizibile în spectrul liniar al camerei.
Rezultat: Sistemul a menținut detecția persoanei chiar și în colțurile umbrite ale încăperii.
5.4 Analiza Performanței
Sistemul a rulat pe un procesor performant.
Timp mediu pre-procesare DSP: 2-3 ms
Timp mediu inferență YOLO (Medium): 40-60 ms
FPS Rezultat: ~24 FPS, validând funcționarea în timp real.

CAPITOLUL 6. CONCLUZII
6.1 Concluzii Generale
Proiectul demonstrează eficacitatea abordării interdisciplinare în ingineria software. Prin combinarea pre-procesării matematice a semnalului (DSP) cu inteligența artificială, s-a obținut un sistem robust, capabil să funcționeze în condiții non-ideale unde sistemele standard ar eșua. Utilizarea tehnicilor precum LUT pentru Gamma și procesarea în spațiul LAB demonstrează că optimizarea algoritmică poate compensa limitările hardware.
6.2 Direcții de Dezvoltare
Pentru viitor, se propune implementarea algoritmului DeepSORT pentru urmărirea identității persoanelor (tracking persistent) și optimizarea codului pentru a rula pe plăci de dezvoltare embedded (Raspberry Pi), folosind versiunea YOLOv8 Nano.

BIBLIOGRAFIE
Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing  (4th Edition). Pearson Education.
Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Disponibil online la: https://arxiv.org/abs/1506.02640 [Accesat Decembrie 2025]
Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 Documentation. GitHub & Ultralytics Docs. Disponibil online la: https://docs.ultralytics.com [Accesat Decembrie 2025]
Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools. Documentație tehnică disponibilă la: https://opencv.org [Accesat Decembrie 2025]
Reza, A. M. (2004). Realization of the Contrast Limited Adaptive Histogram Equalization (CLAHE). Journal of VLSI Signal Processing Systems for Signal, Image and Video Technology. [Accesat Decembrie 2025]
Rosebrock, A. (2021). Computer Vision and Deep Learning Resource Guide. PyImageSearch. Disponibil online la: https://pyimagesearch.com [Accesat Decembrie 2025]

