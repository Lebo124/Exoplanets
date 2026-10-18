# Exoplanetenonderzoek — Leo Bode

Deze repository bevat het oorspronkelijke SOM-onderzoek en de voortzetting met NASA-gegevens van oktober 2026.

## Onderzoek van oktober 2026

Alle nieuwe bestanden staan in [onderzoek_2026](onderzoek_2026). De oorspronkelijke bestanden in de hoofdmap zijn behouden.

| Onderdeel | Uitgevoerd notebook |
| --- | --- |
| SOM met oorspronkelijke schaling | [NASA 2026](onderzoek_2026/Exoplaneten_SOM_NASA_2026.ipynb) |
| SOM met logaritmische schaling | [Log — twaalf groepen](onderzoek_2026/Exoplaneten_SOM_log_12_groepen.ipynb) |
| SOM met PCA | [PCA — twaalf groepen](onderzoek_2026/Exoplaneten_SOM_PCA_12_groepen.ipynb) |
| Bekende begeleiders van groep 1 | [Gasreuzen en begeleiders](onderzoek_2026/Groep1_bekende_gasreuzen.ipynb) |
| Detecteerbaarheid van hete gasreuzen bij groep 4 | [Detecteerbaarheid groep 4](onderzoek_2026/Groep4_detecteerbare_hete_reuzen.ipynb) |

Bij de notebooks zijn HTML-versies opgeslagen. Download een HTML-bestand en open het in een browser om het volledige verslag te lezen.

De laatste analyse omvat 256 sterren met groep-4-planeten. Voor 52 sterren zijn gepubliceerde injectie- en terugvindtests beschikbaar; 24 voldoen aan onze conservatieve selectie voor het onderzochte bereik (minimum-massa 0,3–3 Jupitermassa's, omlooptijd 2–20 dagen). Van die 24 hebben er 22 geen bekende groep-1-planeet. De testfractie is geen garantie voor iedere mogelijke planeet. Voor de overige 204 sterren is hiermee geen uitspraak gedaan over detecteerbaarheid.

Groepsnummers horen bij het betreffende model. De twaalf ingestelde SOM-groepen bewijzen geen twaalf natuurlijke planeetsoorten. De analyses zijn verkennend.

## Gegevens en herhaalbaarheid

De map bevat de bewaarde NASA-momentopname, bronverantwoording, scripts, getrainde modellen, tabellen, figuren en gepubliceerde terugvindtests. Zie de [uitgebreide toelichting](onderzoek_2026/README.md) en de bronbestanden bij elke analyse.

Installeer de pakketten uit `onderzoek_2026/requirements.txt` en voer scripts uit vanuit `onderzoek_2026`. Een nieuwe download is niet nodig om de bewaarde gegevens te gebruiken.
