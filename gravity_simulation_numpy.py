import sys
import time
from math import sqrt

import numpy as np
import pygame

dt = 0.01

G = 6.67408e-11 * 100_000_000  # Otherwise the bodies would not move given the small value of gravitational constant
NUM_OF_BODIES = 100
WIDTH = 1200
HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (109, 196, 255)

D = 400
exist = np.ones((NUM_OF_BODIES,), dtype=np.int)

# Velocity = np.zeros((NUM_OF_BODIES, Dimensions,), dtype=np.float)
vx = np.zeros((NUM_OF_BODIES,), dtype=np.float)
vy = np.zeros((NUM_OF_BODIES,), dtype=np.float)
vz = np.zeros((NUM_OF_BODIES,), dtype=np.float)

# Position = np.zeros((NUM_OF_BODIES, Dimensions,), dtype=np.float)
px = np.random.uniform(low=D-100, high=D+100, size=NUM_OF_BODIES)
py = np.random.uniform(low=D-100, high=D+100, size=NUM_OF_BODIES)
pz = np.random.uniform(low=D-100, high=D+100, size=NUM_OF_BODIES)

m = np.random.randint(1, 100, size=NUM_OF_BODIES)
m = np.where(m == 0, 1, m)
r = np.array(np.sqrt(np.abs(m)))

'''
exist = np.ones((NUM_OF_BODIES,), dtype=np.int)

vx = np.zeros((NUM_OF_BODIES,), dtype=np.float)
vy = np.zeros((NUM_OF_BODIES,), dtype=np.float)

px = np.random.uniform(low=10, high=WIDTH - 10, size=NUM_OF_BODIES)
py = np.random.uniform(low=10, high=HEIGHT - 10, size=NUM_OF_BODIES)

m = np.random.randint(1, 100, size=NUM_OF_BODIES)
m = np.where(m == 0, 1, m)
r = np.array(np.sqrt(np.abs(m)))

fx = np.zeros((NUM_OF_BODIES,), dtype=float)
fy = np.zeros((NUM_OF_BODIES,), dtype=float)
'''

pygame.init()
size = WIDTH, HEIGHT
screen = pygame.display.set_mode(size)

font = pygame.font.SysFont('Arial', 16)
text = font.render('0', True, BLUE)
textRect = text.get_rect()
while True:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    in_t = time.time()
    # TotalEnergy = 0
    for i in range(0, NUM_OF_BODIES):
        if exist[i] == 0:
            continue
        # TotalEnergy += m[i] * pow(sqrt(vx[i] ** 2 + vy[i] ** 2), 2) / 2
        xdiff = (px - px[i])
        ydiff = (py - py[i])

        distance = np.sqrt(xdiff ** 2 + ydiff ** 2)
        # U = G * m[i] * np.divide(m, distance) / 2
        # TotalEnergy += np.ma.masked_invalid(U).sum()
        '''
        eff_distance = distance - abs(r) - abs(r[i])
        for index in range(0, NUM_OF_BODIES):
            if index == i:
                pass
            elif eff_distance[index] <= 0:
                vx[i] = (m[i] * vx[i] + m[index] * vx[index]) / (m[i] + m[index])
                vy[i] = (m[i] * vy[i] + m[index] * vy[index]) / (m[i] + m[index])
                px[i] = (m[i] * px[i] + m[index] * px[index]) / (m[i] + m[index])
                py[i] = (m[i] * py[i] + m[index] * py[index]) / (m[i] + m[index])
                m[i] += m[index]
                r[i] = np.sqrt(np.abs(m[i]))
                exist[index] = 0
                vx[index] = 0
                vy[index] = 0
                m[index] = 0
        '''
        f = G * m[i] * np.divide(m, distance ** 2)

        sin = np.divide(ydiff, distance)
        cos = np.divide(xdiff, distance)

        fx_total = np.nansum(np.multiply(f, cos))
        fy_total = np.nansum(np.multiply(f, sin))

        vx[i] = vx[i] + (fx_total / m[i]) * dt
        vy[i] = vy[i] + (fy_total / m[i]) * dt

        px[i] = px[i] + vx[i]
        py[i] = py[i] + vy[i]
        # color = (255, 255, 255)
        if m[i] > 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        pygame.draw.circle(screen, color, (px[i], py[i]), r[i])
        #pygame.draw.rect(screen, color, pygame.Rect(px[i], py[i], abs(m[i]), abs(m[i])))
    # print(time.time() - in_t)
    # print(TotalEnergy)
    pygame.display.flip()
