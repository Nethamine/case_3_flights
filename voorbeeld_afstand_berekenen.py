# Dit is een uitleg voor het berekenen van de
# hemelsbrede (Great Circle Distance) afstand in python

# Helemsbrede afstand is de kortste afstand tussen 2 punten op de aarde
# oftewel tussen 2 coordinaten

# 1. Zorg dat het bestand vinc.py in dezelfde map zit als het bestand waar je in werkt

# 2. importeer de functie v_direct met de onderstaande code
from vinc import v_direct

# 3. definieerd 2 coordinaten, (latitude, longitude)
point1 = (52.320726, 4.876430)
point2 = (53.147404, 6.370976)

# 4. bereken de afstand door de in de functie v_direct te stoppen, de afstand is gegeven in meters
afstand = v_direct(point1, point2)
print(afstand)