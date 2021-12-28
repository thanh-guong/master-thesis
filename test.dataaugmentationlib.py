import dataaugmentationlib
import numpy as np
import math

theta = math.pi/2
sigma = 0.1

signal = [
    [1, 2],     # I
    [3, 4]      # Q
]

signals = [signal]
signals = np.array(signals)

labels = [0]
labels = np.array(labels)

rotation = dataaugmentationlib.rotate(signals, labels, theta=theta, increment_percentage=1)
h_flip = dataaugmentationlib.horizontal_flip(signals, labels, increment_percentage=1)
v_flip = dataaugmentationlib.vertical_flip(signals, labels, increment_percentage=1)
gauss = dataaugmentationlib.add_gaussian_noise(signals, labels, standard_deviation=sigma, increment_percentage=1)

print("=============== Rotation ==========================")
print("signal")
print(signal)
print("Rotated signal by " + str(theta))
print(rotation[0][1])

print("=============== Horizontal flipping ===============")
print("signal")
print(signal)
print("Flipped signal")
print(h_flip[0][1])

print("=============== Vertical flipping =================")
print("signal")
print(signal)
print("Flipped signal")
print(v_flip[0][1])

print("=============== Gauss Noise =======================")
print("signal")
print(signal)
print("Disturbed signal with gaussian noise with standard deviation" + str(sigma))
print(gauss[0][1])
