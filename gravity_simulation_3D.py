import sys
import time

import numpy as np
#import pygame

from vpython import *

scene = canvas(title='Gravity',
     width=1000, height=1000)

dt = 100
Dimensions = 6

G = 6.67408e-11 * 100_000_000  # Otherwise the bodies would not move given the small value of gravitational constant
NUM_OF_BODIES = 100
D = 0
'''
WIDTH = 800
HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (109, 196, 255)
D = 400
'''
exist = np.ones((NUM_OF_BODIES), dtype=int)

Velocity = np.zeros((NUM_OF_BODIES, Dimensions), dtype=float)
Position = np.random.uniform(low=D - 100, high=D + 100, size=(NUM_OF_BODIES, 3))
m = np.random.randint(1, 10, size=NUM_OF_BODIES)
m = np.where(m == 0, 1, m)
r = np.array(np.sqrt(np.abs(m)))

balls = [sphere(color=(color.red if m[i] > 0 else color.blue), radius=r[i], pos=vector(Position[i][0], Position[i][1], Position[i][2]), p=vector(0, 0, 0)) for i in range(NUM_OF_BODIES)]

'''
pygame.init()
size = WIDTH, HEIGHT
screen = pygame.display.set_mode(size)

font = pygame.font.SysFont('Arial', 16)
text = font.render('0', True, BLUE)
textRect = text.get_rect()
'''
while True:
    '''
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    '''

    # in_t = time.time()
    #TotalEnergy = 0
    for i in range(0, NUM_OF_BODIES):
        if exist[i] == 0:
            continue
        #TotalEnergy += m[i] * pow(np.linalg.norm(Velocity[i]), 2) / 2

        distance = Position - Position[i]
        scalar_distance = np.linalg.norm(distance, axis=1)

        #U = G * m[i] * np.divide(m, scalar_distance) / 2
        #TotalEnergy += np.ma.masked_invalid(U).sum()

        #'''
        eff_distance = scalar_distance - np.abs(r) - abs(r[i])
        for index in range(0, NUM_OF_BODIES):
            if index == i or balls[index] == 0:
                pass
            elif eff_distance[index] <= 0:
                Velocity[i] = (m[i] * Velocity[i] + m[index] * Velocity[index]) / (m[i] + m[index])
                Position[i] = (m[i] * Position[i] + m[index] * Position[index]) / (m[i] + m[index])
                m[i] += m[index]
                r[i] = np.sqrt(np.abs(m[i]))
                balls[i].radius = r[i]
                balls[index].visible = False
                ball = balls[index]
                del ball
                balls[index] = 0
                exist[index] = 0
                Velocity[i] = np.zeros(Dimensions)
                m[index] = 0
        #'''

        f = np.reshape(G * m[i] * np.divide(m, np.power(np.ma.masked_invalid(scalar_distance), 3)), (NUM_OF_BODIES, 1))
        F = (distance * np.ma.masked_invalid(f)).sum(axis=0) / m[i] * dt
        position = Position[i]
        position += F

        balls[i].pos = vector(position[0], position[1], position[2])
        '''
        color = (255, 255, 255)
        if m[i] > 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        pygame.draw.circle(screen, color, (position[0], position[1]), r[i])
        '''
        Position[i] = position
    # print(time.time() - in_t)
    # print(TotalEnergy)
    # pygame.display.flip()
