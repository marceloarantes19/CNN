# Exemplo 11 
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
  # Dimensões da imagem e do Kernel
  (iH, iW) = image.shape[:2]
  (kH, kW) = K.shape[:2]
  print("Dimenssões: ", iH, iW, " - ", kH, kW)

  # Aloca memória para a imagem de saida ... atenção com o pad (i, j - na borda)
  # precisam ser o centro do Kernel, então, é necessário gerar um pad
  # a imagem de entrada não é reduzida 
  pad = (kW - 1) // 2
  image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
  output = np.zeros((iH, iW), dtype="float")


  # loop na imagem movendo o kernel em cada coordinada (x, y)
  # partindo da esquerda para a direita e de cima para baixo
  for y in np.arange(pad, iH + pad):
    for x in np.arange(pad, iW + pad):
      # extrai a Região de Interesse (ROI) da imagem a partir
      # da coordenada (x, y) que é o centro da parte da imagem
      # a ser avaliada
      roi = image[y - pad:y + pad+1, x - pad:x + pad+1]
      #print("Roi: ", roi)
      # Executa a convolução para isso:
      # multiplicase cada elemento da ROI pelo kernel
      # e soma-os
      k = (roi * K).sum()

      # armazena o valor convolvido na coordenada (x, y)
      # da matriz de saída
      output[y - pad, x-pad] = k
    
  # Modifica a escala de saída em um limite entre [0, 255]
  output = rescale_intensity(output, in_range=(0,255))
  output = (output * 255).astype("uint8")

  # retorna a imagem de saída
  return output 

# construindo um parse e uma argumento para parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Caminho para a Imagem")
args = vars(ap.parse_args())


# construindo diversos Kernels
smallBlur = np.ones((7,7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21,21), dtype="float") * (1.0 / (21 * 21))

# Filtro para Sharpening
sharpen = np.array((
  [0, -1, 0],
  [-1, 5, -1],
  [0, -1, 0]), dtype = "int")

# Filtro para laplace
laplacian = np.array((
  [0, 1, 0],
  [1, -4, 1],
  [0, 1, 0]), dtype = "int")

# Filtro para Sobels
sobelX = np.array((
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]), dtype = "int")

sobelY = np.array((
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1]), dtype = "int")

# Filtro para Emboss
emboss = np.array((
  [-2, -1, 0],
  [-1, 1, 1],
  [0, 1, 2]), dtype = "int")

kernelBank = (
  ("smallBlur", smallBlur),
  ("largeBlur", largeBlur),
  ("sharpen", sharpen),
  ("laplacian", laplacian),
  ("sobelX", sobelX),
  ("sobelY", sobelY),
  ("emboss", emboss)
)

dim = (256, 256)
image = cv2.imread(args["image"])
image = cv2.resize(image, dim)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", gray)

for (kernelName, K) in kernelBank:
  print("[Info] aplicando o kernel {}".format(kernelName))
  convolveOutput = convolve(gray, K)
  opencvOutput = cv2.filter2D(gray, -1, K)
  cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
  cv2.imshow("{} - OpenCV".format(kernelName), opencvOutput)

cv2.waitKey(0)
#cv2.destroyAllWindows()
