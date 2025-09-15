# Doel

In de databases met hydraulische belastingen data bij Riskeer zijn doorgaans geen locaties opgenomen nabij waterkeringen die binnen een havenbekken liggen. HB Havens ondersteunt het genereren van databases waarin dergelijke uitvoerlocaties wel zijn opgenomen. Daarbij worden hydraulische belastingen data van locaties buiten het havenbekken getransformeerd naar de nieuwe locaties binnen het havenbekken.

# Installer

De applicatie bestaat uit python scripts met data. Er is geen installer. In de gebruikershandleiding staan aanwijzingen voor het installeren van HB Havens.

# Gebruik en functionaliteit

De tool ondersteunt de gebruiker in:
* visualiseren shape bestanden voor het haventerrein, de havendammen en de uitvoerlocaties en toevoegen verschillende achtergrondkaarten
* genereren uitvoerlocaties nabij de waterkering
* kiezen steunpuntlocatie uit een geselecteerde HRD
* doorlopen stappenplan voor zowel de eenvoudige als de geavanceerde methode om de golfcondities te bepalen, in de vorm van een (stappen)wizard
* kiezen modelonzekerheden voor de uit te voeren databases
* genereren SQLite-databases die voldoen aan het SQLite-formaat van RisKeer (ook toepasbaar in Hydra-NL)

Binnen de geavanceerde golftransformatiemethode wordt gebruik gemaakt van de modellen SWAN, Pharos en/of Hares. Deze modellen maken echter geen deel uit van de tool HB Havens zelf en het maken van een modelschematisatie voor deze modellen wordt ook niet door HB Havens ondersteund. Wel ondersteunt HB Havens voor deze modellen:
* het klaarzetten van de invoerbestanden (de bestanden met verschillende waterstanden en golfcondities als randvoorwaarden)
* het inlezen en verwerken van de rekenresultaten.

# Versies en aanpassingen

De actuele versie is 25.1.1 en dateert van augustus 2025.
De belangrijkste aanpassingen ten opzichte van de vorige release (versie 23.1.1, uit december 2023) betreffen aanpassingen in de eenvoudige methode:
* de bepaling van de rekenwaarde van de strijklengte voor de lokale golfgroei 
* de bepaling van de golfrichting als gevolg van diffractie
Dit heeft invloed op de rekenresultaten van de eenvoudige methode.

# Technische informatie

Voor de benodigde Python versie en Python packages wordt verwezen naar de gebruikershandleiding.

# Achtergrondinformatie

De tool is in opdracht van Rijkswaterstaat-WVL ontwikkeld door HKV Lijn in Water en Aktis Hydraulics. De tool wordt voor Rijkswaterstaat beheerd in een Deltares (Subversion) repository, waar ook een testbank aanwezig is. Voor onderhoud en doorontwikkeling van de tool wordt deze laatstgenoemde repository gebruikt; de GitHub (Git) repository wordt alleen gebruikt voor de beschikbaarstelling van de tool. Neem in geval van behoefte aan onderhoud of doorontwikkeling s.v.p. contact op met ondergenoemde contactpersoon.

# Contactpersoon

Robert Slomp (robert.slomp@rws.nl) [Rijkswaterstaat-WVL](https://www.rijkswaterstaat.nl/over-ons/onze-organisatie/organisatiestructuur/water-verkeer-en-leefomgeving/index.aspx)

