# pylint: disable=missing-function-docstring
"""------------------------------------------------------------------------------+
#
# Nathan A. Rooy
# Simple Particle Swarm Optimization (PSO) with Python
# Last update: 2018-JAN-26
# Python 3.6
#
# ------------------------------------------------------------------------------+"""

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

import random
# import yaml  # type: ignore
from ultralytics import YOLO  # type: ignore
from torch import cuda

N_DIM = 3
# lr0, lrf, momentum

# N_DIM = 3
# batch, weight_decay, cls

# N_DIM = 3
# hsv_h, hsv_s, hsv_v

# N_DIM = 2
# conf, iou

# --- MAIN ---------------------------------------------------------------------+
# with open("args.yaml", "r", encoding="utf-8") as file:
#     args = yaml.load(file, Loader=yaml.FullLoader)
w = 0.7  # constant inertia weight (how much to weigh the previous velocity)
c1 = 2  # cognative constant
c2 = 2  # social constant


def get_map(x, n_iter: int, n_part: int, plots: bool = False) -> float:
    model = YOLO("yolov8n.pt")

    # Use the model
    model.train(
        data=r"/home/maria/TFM/data/datasets/filtered_DATASET_v2/cfg.yaml",
        project=r"/home/maria/TFM/data/results/filtered_DATASET_v2/PSO_batch",
        name=f"{n_iter}_{n_part}_{x[0]}_{x[1]}_{x[2]}",
        epochs=40,
        patience=5,
        imgsz=608,
        device="cuda:0" if cuda.is_available() else "cpu",
        exist_ok=True,
        seed=444,
        optimizer="Adam",
        # close_mosaic=0,
        # hsv_h=0,
        # hsv_s=0,
        # hsv_v=0,
        # translate=0,
        # scale=0,
        # fliplr=0,
        # mosaic=0,
        verbose=False,
        cos_lr=True,
        batch=16,
        lr0=x[0],
        lrf=x[1],
        momentum=x[2],
        weight_decay=1e-06,
        cls=0.4973647142714877,
        plots=plots,
    )

    return model.val().box.map


class Particle:
    """A single particle of the swarm"""

    def __init__(self, x0):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.map_best_i = -1  # best error individual
        self.map_i = -1  # error individual

        for i in range(0, N_DIM):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, cost_func, n_iter: int, n_part: int):
        self.map_i = cost_func(self.position_i, n_iter, n_part, (n_iter % 3) == 0)

        # check to see if the current position is an individual best
        if self.map_i > self.map_best_i or self.map_best_i == -1:
            self.pos_best_i = self.position_i.copy()
            self.map_best_i = self.map_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):

        for i in range(0, N_DIM):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, N_DIM):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if necessary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]
            
            # for batch, or any integer hyperparameter
            if i == 0:
                self.position_i[i] = round(self.position_i[i])


def minimize(cost_func, x0, bounds, num_particles, maxiter, verbose=False):

    if len(x0) != N_DIM:
        raise ValueError(
            "The number of dimensions must be equal to the number of elements in x0"
        )

    map_best_g = -1  # best error for group
    pos_best_g = []  # best position for group

    # establish the swarm
    swarm = []
    for i in range(0, num_particles):
        swarm.append(Particle(x0))

    # begin optimization loop
    i = 0
    while i < maxiter:
        # write particle information to file
        with open("/home/maria/TFM/data/results/filtered_DATASET_v2/PSO_batch/particle_info.txt", "a", encoding="utf-8") as file:
            for j in range(0, num_particles):
                file.write(f"Iteration: {i-1}, Particle: {j}\n")
                file.write(f"Position: {swarm[j].position_i}\n")
                file.write(f"Velocity: {swarm[j].velocity_i}\n")
                file.write(f"Best Position: {swarm[j].pos_best_i}\n")
                file.write(f"Best Error: {swarm[j].map_best_i}\n")
                file.write("\n")
            file.write(f"\nBest Solution: {map_best_g}. Best position: {pos_best_g}")
        if verbose:
            print(f"iter: {i:>4d}, best solution: {map_best_g:10.6f}")

        # cycle through particles in swarm and evaluate fitness
        for j in range(0, num_particles):
            swarm[j].evaluate(cost_func, i, j)

            # determine if current particle is the best (globally)
            if swarm[j].map_i > map_best_g or map_best_g == -1:
                pos_best_g = list(swarm[j].position_i)
                map_best_g = float(swarm[j].map_i)

        # cycle through swarm and update velocities and position
        for j in range(0, num_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position(bounds)
        i += 1

    # print final results
    if verbose:
        print("\nFINAL SOLUTION:")
        print(f"   > {pos_best_g}")
        print(f"   > {map_best_g}\n")

    return map_best_g, pos_best_g


# --- END ----------------------------------------------------------------------+

if __name__ == "__main__":
    # --- RUN ----------------------------------------------------------------------+
    seed = 444
    random.seed(seed)
    x0 = [16, 0.0005, 0.6]
    bounds = [(4, 128), (1e-6, 1e-2), (0.1, 5)]
    opt = minimize(get_map, x0, bounds, num_particles=10, maxiter=15, verbose=False)
