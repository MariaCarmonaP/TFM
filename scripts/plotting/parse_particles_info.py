with open("/home/maria/TFM/data/results/filtered_DATASET_v2/PSO_batch/particle_info.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

    # parse particle information
    particle_info = []
    for i, line in enumerate(lines):
        if line.startswith("Iteration"):
            particle_info.append({
                "iteration": int(line.split()[1].strip(",")),
                "particle": int(line.split()[3]),
                "position": list(map(float, lines[i + 1].split()[1:])),
                "velocity": list(map(float, lines[i + 2].split()[1:])),
                "best_position": list(map(float, lines[i + 3].split()[2:])),
                "best_error": float(lines[i + 4].split()[2])
            })

    # plot particle information
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    for i in range(4):
        ax = axs[i // 2, i % 2]
        ax.set_title(f"Particle {i}")
        ax.plot([info["iteration"] for info in particle_info], [info["position"][0] for info in particle_info], label="Position")
        ax.plot([info["iteration"] for info in particle_info], [info["velocity"][0] for info in particle_info], label="Velocity")
        ax.plot([info["iteration"] for info in particle_info], [info["best_position"][0] for info in particle_info], label="Best Position")
        ax.plot([info["iteration"] for info in particle_info], [info["best_error"] for info in particle_info], label="Best Error")
        ax.legend()
    plt.tight_layout()
    plt.show()