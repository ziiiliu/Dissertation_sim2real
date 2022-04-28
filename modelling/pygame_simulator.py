import pygame
import numpy as np
import os

"""Define color"""
RED = (200, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 155, 0)
yellow = (155, 155, 0)
white = (255, 255, 255)
BLACK = (0, 0, 0)
PINK = (255, 192, 203)


class Robot(pygame.sprite.Sprite):

    def __init__(self, init_x, init_y, width=40, height=40, color=RED):
        super(Robot, self).__init__()
        self.surf = pygame.Surface((width, height))
        self.surf.fill(color)
        self.rect = self.surf.get_rect()
        self.x, self.y = init_x, init_y
        self.width, self.height = width, height

class Simulator(object):

    def __init__(self, frequency, statelist=None, modelled_statelist=None, reflist=None, make_video=False, movie_name="movie", png_folder="animation/pngs"):
        self.frequency = frequency
        self.dt = 1/frequency
        self.statelist, self.modelled_statelist, self.reflist = statelist, modelled_statelist, reflist
        self.draw_ideal = reflist is not None
        # self.robot_x, self.robot_y = 300, 300
        if make_video:
            self.frame_count = 0
            self.png_path = png_folder
            self.png_name = "capture"
            self.movie_name = movie_name

    def make_mp4(self):
        os.system(f"ffmpeg -r {self.frequency} -i {self.png_path}/{self.png_name}%08d.png -vcodec mpeg4 -q:v 0 -y animation/videos/{self.movie_name}.mp4")

    def make_png(self, screen):
        self.frame_count+=1
        fullpath = self.png_path + "/" + self.png_name + "%08d.png"% self.frame_count
        pygame.image.save(screen,fullpath)

    def main(self, screen):
        clock = pygame.time.Clock()
        start_x, start_y = 500,300
        real_robot = Robot(init_x=start_x, init_y=start_y)
        modelled_robot = Robot(init_x=start_x, init_y=start_y, color=GREEN)
        ideal_robot = Robot(init_x=start_x, init_y=start_y, color=BLUE)

        running = True
        
        real_coordinates = [(real_robot.x+real_robot.width/2, real_robot.y+real_robot.height/2)]
        modelled_coordinates = [(modelled_robot.x+modelled_robot.width/2, modelled_robot.y+modelled_robot.height/2)]
        ideal_coordinates = [(ideal_robot.x+ideal_robot.width/2, ideal_robot.y+ideal_robot.height/2)]
        # Initialize a font for displaying text on the screen
        font = pygame.font.SysFont(None, 25)

        if self.statelist is not None:
            index = 0
            while running:
                clock.tick(self.frequency)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running=False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running=False
                screen.fill(white)

                # Updating the coordinates of the robot in each timestep
                real_robot.x +=  self.dt * self.statelist[index][0] * 40
                real_robot.y += self.dt * self.statelist[index][1] * 40
                modelled_robot.x +=  self.dt * self.modelled_statelist[index][0] * 40
                modelled_robot.y +=  self.dt * self.modelled_statelist[index][1] * 40
                ideal_robot.x += self.dt * self.reflist[index][0] * 40
                ideal_robot.y += self.dt * self.reflist[index][1] * 40
                

                # Keeping track of the coordinates for plotting the trajectories
                real_coordinates.append((real_robot.x+real_robot.width/2, real_robot.y+real_robot.height/2))
                modelled_coordinates.append((modelled_robot.x+modelled_robot.width/2,modelled_robot.y+modelled_robot.height/2))
                ideal_coordinates.append((ideal_robot.x+ideal_robot.width/2, ideal_robot.y+ideal_robot.height/2))

                # Calculate the difference metric between the modelled and real robots
                displacement = pow((real_robot.x - modelled_robot.x) ** 2 + (real_robot.y - modelled_robot.y) **2, 0.5)
                displacement_text = font.render(f"Displacement: {displacement:.2f}", True, BLACK)

                index += 1
                # If the states are depleted, we stop.
                if index >= len(self.statelist) or index >= len(self.modelled_statelist):
                    running=False
                screen.blit(real_robot.surf,(real_robot.x,real_robot.y))
                screen.blit(modelled_robot.surf,(modelled_robot.x,modelled_robot.y))
                screen.blit(ideal_robot.surf,(ideal_robot.x,ideal_robot.y))

                screen.blit(displacement_text, (450,440))

                title_text = font.render(f"Simulation timestep: {self.frame_count}", True, BLACK)
                screen.blit(displacement_text, (100,30))

                pygame.draw.lines(screen, PINK, False, real_coordinates, width=4)
                pygame.draw.lines(screen, GREEN, False, modelled_coordinates, width=4)
                pygame.draw.lines(screen, BLUE, False, ideal_coordinates, width=4)
                
                self.make_png(screen)
                pygame.display.flip()


if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption("Move")
    screen = pygame.display.set_mode((640,480))

    # statelist = np.load("../first_collection/cur_states.npy")
    # reflist = np.load("../first_collection/ref_states.npy")
    statelist = np.load("../second_collection_slower/cur_states.npy")
    reflist = np.load("../second_collection_slower/ref_states.npy")

    # modelled_statelist = np.load("data/simplepredictor_5000_steps_from_start.npy")
    # modelled_statelist = np.load("data/psnn_visible_5_5000_steps_from_start.npy")
    # modelled_statelist = np.load("data\simplepredictor_1_layer_linear_differential_360_steps_shift_50.npy")
    modelled_statelist = np.load("data/2nd_psnn_visible_100_differential_10000_steps.npy")
    print(statelist.shape)

    movie_name = "2nd_psnn_visible_100_differential_10000_steps"
    png_folder = "animation/2nd_100_visible_10000_steps"

    simulator = Simulator(frequency=100, statelist=statelist,
                        modelled_statelist=modelled_statelist,
                        reflist=reflist, 
                        make_video=True, movie_name=movie_name, png_folder=png_folder)
    simulator.main(screen)

    simulator.make_mp4()