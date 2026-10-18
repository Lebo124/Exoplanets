# Exoplaneten: SOM op nieuwe NASA-data

Uitgevoerde proef voor Leo, 18 oktober 2026 (Nederlandse tijd).

Open **Exoplaneten_SOM_NASA_2026.ipynb** voor uitleg, uitgevoerde tabellen en grafieken. De HTML-versie laat dezelfde resultaten zien zonder Python te hoeven draaien.

## Uitkomst

Van 6.451 opgehaalde planeten zijn 5.481 bruikbaar voor tien kenmerken. Alle twaalf vooraf ingestelde SOM-plekken zijn bezet (25 tot 1.153 planeten per plek). De kaart onderscheidt onder meer kleine hete planeten, grote planeten met korte banen en zware planeten met langere banen. Dit bewijst geen twaalf natuurlijke planeetsoorten.

2.825 van de gebruikte massa’s en 1.094 stralen zijn door NASA ingevuld; 819 massa’s zijn minimum-massa’s. De aantallen overlappen. Een controle zonder ingevulde massa/straal en met ware massa’s omvat 1.546 planeten; die indeling verschilt (ARI 0,407 op de gemeenschappelijke planeten). De controle verandert ook de populatie en de schaalgrenzen, dus het verschil is niet alleen aan ingevulde waarden toe te schrijven.

Twee andere startgewichten geven onder dezelfde vaste rijvolgorde dezelfde indeling (ARI 1,000). Een eenmalig geschudde vaste volgorde geeft een vergelijkbare indeling (ARI 0,975). De topografische fout is 27,1%: de kaart is geen overal vloeiende ordening.

## Wat is behouden en veranderd?

De online update, 12×1 kaart, 1.600 epochs, alpha 0,05, sigma 4 en min-maxschaling uit de oude code zijn behouden. De oude TensorFlow-lussen zijn naar NumPy/Numba overgezet; een kleine numerieke vergelijking met de oude formule is geslaagd. Numba versnelt de berekening zonder het leeralgoritme te vervangen. De definitieve gewichten worden na de laatste update opgehaald (in de oude code waren ze een epoch achter).

Tien van de dertien oude kenmerken hebben een NASA-tegenhanger. Oppervlaktedruk, planeethelderheid op maan-afstand en schijnbare stergrootte ontbreken. Geen nieuwe schatting voor die drie variabelen toegevoegd. Daarom geen directe voor/na-vergelijking met het oude 13-kenmerkenmodel.

NASA-lichtkracht is eerst van log10 naar lineaire zonne-eenheden omgerekend. Dichtheid wordt in g/cm³ getoond; dit verandert na min-maxschaling geen afstand ten opzichte van aarddichtheidseenheden. Onvolledige rijen, betwiste bevestigingen, gerapporteerde limieten en niet-positieve waarden zijn uitgesloten. Niets door ons geïmputeerd. De tabel bevat wel door NASA berekende en modelafhankelijke literatuurwaarden.

Het oorspronkelijke databestand was ongeveer op omlooptijd gesorteerd; de huidige hoofdproef gebruikt alfabetische volgorde. De vaste volgorde per epoch is behouden, met een aparte volgordecontrole.

## Bestanden

- `download_data.py`: nieuwe momentopname ophalen; overschrijft lokale snapshot en provenance.
- `analysis.py`: datafilter, SOM, controles en grafieken.
- `data/nasa_pscomppars.csv`: ongewijzigde NASA-download, inclusief verwijzingen en vlaggen.
- `data/provenance.json`: exacte query, ophaaltijd, SHA-256 en documentatie.
- `results/planeten_met_groepen.csv`: gebruikte planeten met groepsnummer, gegevens en herkomst.
- `results/groepsprofielen.csv`: aantal en medianen per groep.
- `results/samenvatting.json`: instellingen en uitgevoerde controles.
- `results/controle_ware_massa_en_straal.csv`: apart getrainde controlepopulatie.
- `results/som_model.npz`: getrainde gewichten, kenmerken en min-maxgrenzen.

## Opnieuw uitvoeren

Python 3.12 is gebruikt. Installeer zo nodig `pip install -r requirements.txt` vanuit de map `onderzoek_2026` en voer daar `python analysis.py` uit. Open het notebook in Jupyter om de analyse met uitleg opnieuw te doorlopen. Een verse download is optioneel: standaard wordt exact de bewaarde snapshot gebruikt. De vaste versievereisten beschrijven de uitvoeromgeving; nieuwere Python-versies kunnen andere pakketversies vereisen.

Het notebook is van boven naar beneden met IPython uitgevoerd en met nbformat gevalideerd. Deze omgeving blokkeert TCP-kernelsockets; daarom is de uitvoering in hetzelfde Python-proces gedaan. Drie grafieken zijn als notebookuitvoer bewaard en visueel gecontroleerd.

## Bronnen

- https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html
- https://exoplanetarchive.ipac.caltech.edu/docs/pscp_calc.html
- https://exoplanetarchive.ipac.caltech.edu/docs/API_TD_columns.html

Alle resultaten zijn verkennend. Meetfouten worden bewaard waar opgehaald, maar zijn nog niet doorgerekend met Monte Carlo. Schaling en onderling afhankelijke kenmerken zijn nog niet aangepast; er is hier geen ander clustermodel getest.
