# Versies en aanpassingen

Dit is versie 2.0, de eerste gedistribueerde versie.

# Doel

In de databases met hydraulische belastingen data bij Ringtoets/Riskeer zijn doorgaans geen locaties opgenomen nabij waterkeringen die binnen een havenbekken liggen. HB Havens ondersteunt het genereren van databases waarin dergelijke uitvoerlocaties wel zijn opgenomen. Daarbij worden hydraulische belastingen data van locaties buiten het havenbekken getransformeerd naar de nieuwe locaties binnen het havenbekken.

# Installer

De applicatie bestaat uit python scripts met data. Er is geen installer. 

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

# Technische informatie

Voor de benodigde Python versie en Python packages wordt verwezen naar de gebruikershandleiding.

# Achtergrondinformatie

Betreft versie 2.0, uit september 2019.

De tool is in opdracht van Rijkswaterstaat-WVL ontwikkeld door HKV Lijn in Water en Aktis Hydraulics. De tool wordt voor Rijkswaterstaat beheerd in een Deltares (Subversion) repository, waar ook een testbank aanwezig is. Voor onderhoud en doorontwikkeling van de tool wordt deze laatstgenoemde repository gebruikt; de GitHub (Git) repository wordt alleen gebruikt voor de beschikbaarstelling van de tool. Neem in geval van behoefte aan onderhoud of doorontwikkeling s.v.p. contact op met ondergenoemde contactpersoon.


# Contactpersoon

[Robert Slomp](robert.slomp@rws.nl) [Rijkswaterstaat-WVL](https://www.rijkswaterstaat.nl/over-ons/onze-organisatie/organisatiestructuur/water-verkeer-en-leefomgeving/index.aspx)

