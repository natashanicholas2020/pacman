# creates the game sprites
import pygame
import math

CELL_SIZE = 50


# draws pacman
def draw_pacman(screen, pacman_pos, mouth_direction):

    # location
    px = pacman_pos[1]*CELL_SIZE + CELL_SIZE//2
    py = pacman_pos[0]*CELL_SIZE + CELL_SIZE//2

    # body
    radius = CELL_SIZE//3
    pygame.draw.circle(screen, (255, 255, 0), (px, py), radius)

    # mouth
    angle_offset = 15

    # up
    if mouth_direction == 0:
        start_angle = 270 - angle_offset
        end_angle = 270 + angle_offset

    # down
    elif mouth_direction == 1:
        start_angle = 90 - angle_offset
        end_angle = 90 + angle_offset
    
    # left
    elif mouth_direction == 2:
        start_angle = 180 - angle_offset
        end_angle = 180 + angle_offset

    # right
    else:
        start_angle = -angle_offset
        end_angle = angle_offset

    # degree to radian
    start_rad = math.radians(start_angle)
    end_rad = math.radians(end_angle)

    # triangle points
    p1 = (px, py)
    p2 = (px + radius*math.cos(start_rad), py + radius*math.sin(start_rad))
    p3 = (px + radius*math.cos(end_rad), py + radius*math.sin(end_rad))

    pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3])


# draws ghost
def draw_ghost(screen, ghost_pos, color=(255, 0, 0)):

    px = ghost_pos[1]*CELL_SIZE + CELL_SIZE//2
    py = ghost_pos[0]*CELL_SIZE + CELL_SIZE//2

    radius = CELL_SIZE//3
    body_height = radius

    # head
    pygame.draw.circle(screen, color, (px, py-radius//2), radius)

    # body
    pygame.draw.rect(
        screen,
        color,
        (px-radius, py-radius//2, radius*2, body_height)
    )

    # bottom 
    wave_radius = radius//3

    for i in range(-2, 3):
        pygame.draw.circle(
            screen,
            color,
            (px + i*wave_radius, py + body_height//2),
            wave_radius
        )

    # eyes
    eye_offset_x = radius//2
    eye_offset_y = -radius//3

    # whites of eyes
    pygame.draw.circle(screen, (255, 255, 255),
                       (px-eye_offset_x, py+eye_offset_y), radius//4)
    pygame.draw.circle(screen, (255, 255, 255),
                       (px+eye_offset_x, py+eye_offset_y), radius//4)

    # pupils
    pygame.draw.circle(screen, (0, 0, 255),
                       (px-eye_offset_x, py+eye_offset_y), radius//8)
    pygame.draw.circle(screen, (0, 0, 255),
                       (px+eye_offset_x, py+eye_offset_y), radius//8)
