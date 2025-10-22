import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def minimgradiente(x,y,m,b,alfa):
  continuar = True
  i = 0

  plt.ion()  # Activar modo interactivo
  fig, ax = plt.subplots()
  ax.scatter(x, y, color='blue', label='Datos reales')
  linea_prediccion, = ax.plot(x, m * x + b, color='red', label='Predicción')
  ax.legend()
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('Descenso de Gradiente')

  while (continuar and i < 1000):
    i += 1
    y_eval = x * m + b
    erroract = np.sum((y - y_eval)**2)
    dEm = -2 * np.sum(x * (y - y_eval))
    dEb = -2 * np.sum(y - y_eval)

    Mag = math.sqrt((dEm**2) + (dEb**2))
    Pm = -dEm / Mag
    Pb = -dEb / Mag

    Mnueva = m + (alfa * Pm)
    Bnueva = b + (alfa * Pb)
    Y_evnueva = x * Mnueva + Bnueva
    Errnuevo = np.sum((y - Y_evnueva)**2)



    if(Errnuevo > erroract):
      continuar = False
    else:
      m = Mnueva
      b = Bnueva
        
        # Actualizar gráfico
    linea_prediccion.set_ydata(m * x + b)
    fig.canvas.draw()
    fig.canvas.flush_events()

  plt.ioff()  # Turn off interactive mode
  plt.show()
  return erroract, m, b, i

# Cargamos los datos
archivo = 'Datos.csv'
datos = pd.read_csv(archivo)
datos.iloc[:, 0] = datos.iloc[:, 0].str.replace('π', '')

X = datos.iloc[:, 0].values.astype(float)
Y = datos.iloc[:, 1].values
M = 1
B = 1
alfa = 0.001

error, m, b, i = minimgradiente(X,Y,M,B,alfa)
print(error)
print(m)
print(b)
print(i)