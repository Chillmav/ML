import pygame
from tsm import TSMGenetic
import numpy as np

model = TSMGenetic()
model.train()
cities = model.cities
size = 800
offset = 100
radius = 10
plane_speed = 1000

pygame.init()
pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 30)
screen = pygame.display.set_mode((1280, 647))
background = pygame.image.load("ML/023 - genetic_algorithms/map.PNG").convert()
background = pygame.transform.scale(background, (1280, 647))

plane_img = pygame.image.load("ML/023 - genetic_algorithms/plane.png").convert_alpha()
plane_img = pygame.transform.scale(plane_img, (40, 40))

b_plane_img = pygame.image.load("ML/023 - genetic_algorithms/best_plane.png").convert_alpha()
b_plane_img = pygame.transform.scale(b_plane_img, (40, 40))

clock = pygame.time.Clock()
running = True
cords = np.zeros((len(cities), 2))
epochs = model.routes

def drawStats(e, shortest):

    text = str(f"Epoch: {e}")
    shortest_path = str(f"Shortest path: {int(np.round(shortest))} pixels")
    epoch_tracker = my_font.render(text, True, "black")
    screen.blit(epoch_tracker, (10, 10))
    shortest_tracker = my_font.render(shortest_path, True, "black")
    screen.blit(shortest_tracker, (10, 30))
    
def render_background(e, shortest):

    
    screen.blit(background, (0, 0))
    drawStats(e + 1, shortest)

    for i, city in enumerate(cities):
    
        pygame.draw.circle(screen, "blue" if i == 0 else "brown", (city[0] + offset, city[1] + offset), radius=radius)
        cords[i][0] = city[0] + offset
        cords[i][1] = city[1] + offset
        label = i if i != 0 else 0
        text = my_font.render(str(label), True, "white")
        text_rect = text.get_rect()
        text_rect.center = (city[0] + offset, city[1] + offset)
        screen.blit(text, text_rect)

def render(dt, p, e, i, k, move):

    if i == model.n:
        i = 0

    plane = plane_img if k != (model.pop_size-1) else b_plane_img

    routes = epochs[e]
    route = routes[k]
    plane_pos = p

    x_cord = cords[int(route[i])][0]
    y_cord = cords[int(route[i])][1]
    target_pos = pygame.Vector2((x_cord, y_cord)) 
    direction = target_pos - plane_pos
    distance = direction.length()

    if distance > 10 and move:  

        direction = direction.normalize()
        plane_pos += direction * plane_speed * dt
        plane_positions[k] = plane_pos
        rect = plane.get_rect(center=plane_pos)
        screen.blit(plane, rect)
        return distance <= 5
    
    else:

        rect = plane.get_rect(center=plane_pos)
        screen.blit(plane, rect)
        return True

e = 0
shortest = model.select_best(epochs[e])

render_background(e, shortest)

pygame.display.flip()

plane_positions = []
plane_indices = np.ones(model.pop_size, dtype=int)

for p in range(model.pop_size):
    
    plane_positions.append(pygame.Vector2((cords[0][0], cords[0][1])))

epoch_input = ""
typing_epoch = False

while running:

    dt = clock.tick(60) / 1000 # delta time in seconds
    events = pygame.event.get()

    for event in events:
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_e:
                typing_epoch = True
                epoch_input = ""

            elif typing_epoch:

                # Confirm input
                if event.key == pygame.K_RETURN:
                    if epoch_input.isdigit():
                        new_epoch = int(epoch_input)
                        if 0 <= new_epoch < len(epochs):
                            e = new_epoch
                            shortest = model.select_best(epochs[e])

                            plane_indices = np.ones(model.pop_size, dtype=int)
                            plane_positions = [
                                pygame.Vector2((cords[0][0], cords[0][1]))
                                for _ in range(model.pop_size)
                            ]
                    typing_epoch = False

                # Cancel
                elif event.key == pygame.K_ESCAPE:
                    typing_epoch = False

                # Delete
                elif event.key == pygame.K_BACKSPACE:
                    epoch_input = epoch_input[:-1]

                # Add digits
                elif event.unicode.isdigit():
                    epoch_input += event.unicode

    render_background(e, shortest)

    if typing_epoch:
        input_surface = my_font.render(
            f"Go to epoch: {epoch_input}",
            True,
            "black"
        )
        screen.blit(input_surface, (10, 60))

    route_len = len(epochs[e][0])

    if np.all(plane_indices > route_len):

        plane_indices = np.ones(model.pop_size, dtype=int)

        if e < len(epochs) - 1:
            e += 1
            shortest = model.select_best(epochs[e])

    for k in range(len(epochs[e])):
        
        if plane_indices[k] <= route_len:

            arrived = render(
                dt,
                plane_positions[k],
                e,
                plane_indices[k],
                k,
                True
            )

            if arrived:

                plane_indices[k] += 1

        else:

            render(
                dt,
                plane_positions[k],
                e,
                route_len - 1,
                k,
                False
            )

    pygame.display.flip()
    pygame.display.update()
    clock.tick(60) 

pygame.quit()
