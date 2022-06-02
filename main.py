from network import random_network, mate, Network
from game import Game
"""
This file is mainly used for personal debugging whilst creating this repo.
This file might get removed or used for other purposes later down the line.
"""

import pygame
fonts = pygame.font.get_fonts()
for font in fonts:
    if "bold" in font:
        print(font)