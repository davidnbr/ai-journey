import pygame
from pong import Game
import neat
import os
import pickle # To save the best genome trained on NEAT

class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, genome, config):

        net = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            output = net.activate((self.right_paddle.y, self.ball.y, 
                                   abs(self.right_paddle.x-self.ball.x)))
            decision = output.index(max(output))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True) # Move the right paddle up
            else:
                self.game.move_paddle(left=False, up=False) # Move the right paddle down

            game_info = self.game.loop()
            #print(game_info.left_score, game_info.right_score)
            self.game.draw(True, False)
            pygame.display.update()
        
        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            # Outputs are the numeric values associated with the paddle's movement
            output1 = net1.activate((self.left_paddle.y, self.ball.y, abs(self.left_paddle.x-self.ball.x)))
            decision1 = output1.index(max(output1)) #index of the highest value in the output (0 still, 1 up, 2 down)

            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True) # Move the left paddle up
            else:
                self.game.move_paddle(left=True, up=False) # Move the left paddle down

            output2 = net2.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x-self.ball.x)))
            decision2 = output2.index(max(output2))

            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True) # Move the right paddle up
            else:
                self.game.move_paddle(left=False, up=False) # Move the right paddle down

            game_info = self.game.loop()

            self.game.draw(draw_score = False, draw_hits = True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                # after paddle miss one ball, inmediately stop the game
                self.calculate_fitness(genome1, genome2, game_info)
                break
    
    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits


def eval_genomes(genomes, config):
    """
    Function to evaluate the genomes in the population
    Here we will create the logic to evaluate the genomes,
    the best approach here is to train each AI against each other,
    so AI 1 will play against AI 2, AI 3, etc.
    """
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))
    
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes)-1:
            break

        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = PongGame(window, width, height)
            game.train_ai(genome1, genome2, config)



def run_neat(config):
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9') # Load the best genome from the last generation (comment for running from scratch)
    #p = neat.Population(config) # Create a population of genomes using config file (comment if starting at checkpoint)
    p.add_reporter(neat.StdOutReporter(True)) # Add a reporter to the population, sending best fitness, etc
    stats = neat.StatisticsReporter() # Save statistics of the population
    p.add_reporter(stats) # Add the stats reporter to the population
    p.add_reporter(neat.Checkpointer(1)) # Save the best genome every generation

    winner = p.run(eval_genomes, 1) # Run the population for X generations and return the best node (genome) from these generations

    # Pickle lets you save an entire Python object
    with open("best.pickle", "wb") as f: # wb mode is to write in binary mode
        pickle.dump(winner, f) # Save the best genome in a file (binary mode)

def test_ai(config):

    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)

    game = PongGame(window, width, height)
    game.test_ai(winner, config)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__) # Get the current directory
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path) # Load the configuration file
    #run_neat(config) # Run the NEAT algorithm (can comment to test the best genome)
    test_ai(config) # Comment for just training the AI