import numpy as np
from utils import oned_well, functions
import matplotlib.pyplot as plt
from alive_progress import alive_bar

generations = 1000000
generation_size = 100000
n_best = 1000
n_variables = 15
well_length = 5
convergence_threshold = 1e-10

for generation_size in [100000, 10000, 1000000]:
    for n_best in [1000, 100, 10000]:
        for convergence_threshold in [0.001, 1e-6, 1e-10, 1e-15, 1e-20]:
            for n_variables in [25, 50, 100]:
                index = (
                    f"{convergence_threshold}_{n_best}_{generation_size}_{n_variables}"
                )
                # Initialization
                population = np.random.random([generation_size, n_variables])
                mean_elite_scores = []
                elite_stds = []

                # Generation loop
                with alive_bar(
                    0,
                    force_tty=True,
                    bar=None,
                    theme="scuba",
                    stats=False,
                    stats_end=False,
                ) as bar:
                    for generation in range(generations):
                        bar()
                        # scores = np.apply_along_axis(oned_well.compute_z, 1, population) # Not parallel
                        scores = functions.parallel_apply_along_axis(
                            oned_well.compute_z_triangles, 1, population
                        )
                        # max_scores_index = np.argpartition(scores, -n_best)[-n_best:] # (si fuese maximizar)
                        min_scores_index = np.argpartition(scores, n_best)[:n_best]
                        best_individuals = population[min_scores_index]
                        best_individuals_mean = best_individuals.transpose().mean(
                            axis=1
                        )
                        best_individuals_std = best_individuals.transpose().std(axis=1)
                        mean_elite_score = scores[min_scores_index].mean()
                        elite_std = best_individuals_std.max()
                        bar.text(
                            "- elite mean score: {:.5g}, elite std: {:.5g}".format(
                                mean_elite_score, elite_std
                            )
                        )
                        mean_elite_scores.append(mean_elite_score)
                        elite_stds.append(elite_std)
                        if best_individuals_std.max() < convergence_threshold:
                            population = best_individuals
                            break
                        else:
                            population = np.random.normal(
                                best_individuals_mean,
                                best_individuals_std,
                                [generation_size, n_variables],
                            )

                plt.plot(
                    mean_elite_scores,
                    label="Last gen: {:.5g}".format(mean_elite_scores[-1]),
                )
                plt.title("Mean score for elite at each generation")
                plt.xlabel("Generation")
                plt.ylabel("Score")
                plt.yscale("log")
                plt.legend()
                plt.savefig(f"results/plots/mean_scores_{index}.png")
                plt.close()

                plt.plot(elite_stds, label="Last gen: {:.5g}".format(elite_stds[-1]))
                plt.title("Sexual transmitted disease for elite at each generation")
                plt.xlabel("Generation")
                plt.ylabel("STD")
                plt.yscale("log")
                plt.legend()
                plt.savefig(f"results/plots/std_elite_{index}.png")
                plt.close()

                # scores = np.apply_along_axis(oned_well.compute_z, 1, population) # Not parallel
                scores = functions.parallel_apply_along_axis(
                    oned_well.compute_z_triangles, 1, population
                )
                best_score = min(scores)
                best_individual = population[np.argmin(scores)]

                def Psi_n(n):
                    return (2 / well_length) ** 0.5 * np.sin(
                        n
                        * np.pi
                        * np.linspace(0, well_length, n_variables)
                        / well_length
                    )

                integral = 0
                for i in range(len(best_individual)):
                    integral = integral + (best_individual[i] ** 2) * (
                        well_length / (n_variables - 1)
                    )

                B = (1 / integral) ** 0.5
                best_individual = B * best_individual

                plt.plot(
                    np.linspace(0, 5, n_variables),
                    best_individual,
                    label="Numerical",
                    marker="o",
                )
                plt.plot(
                    np.linspace(0, 5, n_variables), Psi_n(1), "--", label="Analytical"
                )
                plt.gca().set_title("$\Psi(x)$ - Wave Function")
                plt.gca().set_xlabel("x")
                plt.gca().set_ylabel("$\Psi(x)$")
                plt.legend()
                plt.savefig(f"results/plots/wave_function_{index}.png")
                plt.close()
