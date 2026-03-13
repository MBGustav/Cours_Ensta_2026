#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <omp.h>
#include "fractal_land.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
#include "renderer.hpp"
#include "window.hpp"
#include "rand_generator.hpp"

static double eps = 0.8;  // Coefficient d'exploration
constexpr size_t total_iterations = 4000;

void advance_time(const fractal_land& land, pheronome& phen,
                  const position_t& pos_nest, const position_t& pos_food,
                  std::vector<int>& ants_x,
                  std::vector<int>& ants_y,
                  std::vector<ant::state>& ants_state,
                  std::vector<uint32_t>& ants_seeds,
                  std::size_t& cpteur)
{
    advance(phen, land, pos_food, pos_nest,
            ants_x, ants_y, ants_state, ants_seeds, cpteur, eps);

    phen.do_evaporation();
    phen.update();
}

int main(int nargs, char* argv[])
{
    SDL_Init(SDL_INIT_VIDEO);
    const std::size_t seed = 2026;
    const int nb_ants = 5000;
    const double eps = 0.8;
    const double alpha = 0.7;
    const double beta = 0.999;

    position_t pos_nest{256,256};
    position_t pos_food{500,500};

    // ========================
    // Geração do terreno
    // ========================
    fractal_land land(8,2,1.,1024);
    double max_val = 0.0;
    double min_val = 0.0;

    // Paralelização da busca de max/min
    #pragma omp parallel for collapse(2) reduction(max:max_val) reduction(min:min_val)
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }

    double delta = max_val - min_val;

    // Normalização paralela
    #pragma omp parallel for collapse(2)
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j)
            land(i,j) = (land(i,j)-min_val)/delta;

    ant::set_exploration_coef(eps);

    // ========================
    // Inicialização das formigas
    // ========================
    std::vector<int> ants_x(nb_ants);
    std::vector<int> ants_y(nb_ants);
    std::vector<ant::state> ants_state(nb_ants);
    std::vector<uint32_t> ants_seeds(nb_ants);

    #pragma omp parallel for
    for (size_t i = 0; i < nb_ants; ++i) {
        std::size_t local_seed = seed + i;
        int x = rand_int32(0, land.dimensions()-1, local_seed);
        int y = rand_int32(0, land.dimensions()-1, local_seed);
        ants_x[i] = x;
        ants_y[i] = y;
        ants_state[i] = ant::unloaded;
        ants_seeds[i] = static_cast<uint32_t>(local_seed); // sementes independentes
    }

    // ========================
    // Criação do feromônio e janela
    // ========================
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);
    Window win("Ant Simulation", 2*land.dimensions()+10, land.dimensions()+266);
    Renderer renderer(land, phen, pos_nest, pos_food, ants_x, ants_y);

    size_t food_quantity = 0;
    SDL_Event event;
    bool cont_loop = true;
    bool not_food_in_nest = true;
    std::size_t it = 0;

    auto begin = std::chrono::high_resolution_clock::now();
    std::ofstream csv_file("loop_time.csv", std::ios::out | std::ios::app);
    csv_file << "iteration,elapsed_seconds\n";

    // ========================
    // Loop principal da simulação
    // ========================
    while (cont_loop) {
        auto start_loop = std::chrono::high_resolution_clock::now();
        ++it;

        while (SDL_PollEvent(&event))
            if (event.type == SDL_QUIT)
                cont_loop = false;

        advance_time(land, phen, pos_nest, pos_food, ants_x, ants_y, ants_state, ants_seeds, food_quantity);

        renderer.display(win, food_quantity);
        win.blit();

        if (not_food_in_nest && food_quantity > 0) {
            std::cout << "Primeira comida chegou ao ninho na iteracao " << it << std::endl;
            not_food_in_nest = false;
        }

        if (it == total_iterations) cont_loop = false;
        if(it % 10 == 0) SDL_Delay(10);

        auto end_loop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_loop - start_loop;
        csv_file << it << "," << elapsed.count() << "\n";
    }

    std::ofstream total_csv("total_time.csv", std::ios::out | std::ios::app);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = end - begin;
    total_csv << "total_elapsed_time\n" << total_elapsed.count() << "\n";

    total_csv.close();
    csv_file.close();

    SDL_Quit();
    return 0;
}
