import matplotlib.pyplot as plt

# Heures de la journée
heures = [8, 10, 12, 14, 16, 18, 23, 27]

# Températures correspondantes en degrés Celsius
temperatures = [18, 22, 28, 32, 32, 27, 22, 17]

temperaturesroad=[13,15,29,35,38,30,24,18]
# Affichage de la courbe d'évolution de la température
plt.plot(heures, temperaturesroad, marker='o', linestyle='-', color='r', label='Température de la route')
plt.plot(heures, temperatures, marker='o', linestyle='-', color='b', label='Température ambiante')

plt.xlabel('Heure')
plt.ylabel('Température (°C)')
plt.title("Évolution de la température de la route et de la température ambiante")
plt.grid(True)
plt.legend()

