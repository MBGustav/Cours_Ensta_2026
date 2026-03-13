#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include "fractal_land.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
# include "renderer.hpp"
# include "window.hpp"
# include "rand_generator.hpp"

static double eps = 0.8;  // Coefficient d'exploration
constexpr size_t total_iterations = 4000;

void advance_time( const fractal_land& land, pheronome& phen, 
    const position_t& pos_nest, const position_t& pos_food,
    std::vector<int>& ants_x, 
    std::vector<int>& ants_y, 
    std::vector<ant::state>& ants_state, std::vector<uint32_t>& ants_seeds, std::size_t& cpteur )
    {
        const size_t nb_ants = ants_x.size();
        
        advance(phen, land, pos_food, pos_nest,ants_x, ants_y, ants_state, ants_seeds, cpteur, eps);
        phen.do_evaporation();
        phen.update();
    }
    
    int main(int nargs, char* argv[])
    {
        SDL_Init( SDL_INIT_VIDEO );
        std::size_t seed    = 2026;    // Graine pour la génération aléatoire ( reproductible )
        const int nb_ants   = 5000;   // Nombre de fourmis
        const double eps    = 0.8;     // Coefficient d'exploration
        const double alpha  = 0.7;     // Coefficient de chaos
        const double beta   = 0.999;    // Coefficient d'évaporation
        
        position_t pos_nest{256,256};
        position_t pos_food{500,500};
        fractal_land land(8,2,1.,1024);
        double max_val = 0.0;
        double min_val = 0.0;
        const auto dim = land.dimensions();
        
        
        #pragma GCC ivdep /*independent loops*/
        #pragma GCC unroll 4
        for (fractal_land::dim_t i = 0; i < dim; ++i)
        for (fractal_land::dim_t j = 0; j < dim; ++j)
        {
            double val = land(i,j);
            max_val = std::max(max_val, val);
            min_val = std::min(min_val, val);
        }
        
        const double delta = max_val - min_val;
        
        #pragma GCC ivdep
        #pragma GCC unroll 4
        for (fractal_land::dim_t i = 0; i < dim; ++i){
            for (fractal_land::dim_t j = 0; j < dim; ++j)
            {
                land(i,j) = (land(i,j) - min_val) / delta;
            }
        }
        
        // Définition du coefficient d'exploration de toutes les fourmis.
        ant::set_exploration_coef(eps);
        // On va créer des fourmis un peu partout sur la carte :
        
        
        
        std::vector<int> ants_x(nb_ants);
        std::vector<int> ants_y(nb_ants);
        std::vector<ant::state> ants_state(nb_ants);
        std::vector<uint32_t> ants_seeds(nb_ants);
        
        // CHANGE: using pointers with restrict 
        int* __restrict ants_x_ptr = ants_x.data();
        int* __restrict ants_y_ptr = ants_y.data();
        ant::state* __restrict ants_state_ptr = ants_state.data();
        uint32_t* __restrict ants_seeds_ptr = ants_seeds.data();
        
        auto gen_ant_pos = [&land, &seed]()
        {
            return rand_int32(0, land.dimensions()-1, seed);
        };
        
        #pragma GCC ivdep
        #pragma GCC unroll 4
        for (size_t i = 0; i < nb_ants; ++i)
        {
            int x = gen_ant_pos();
            int y = gen_ant_pos();
            
            ants_x_ptr[i] = x;
            ants_y_ptr[i] = y;
            ants_state_ptr[i] = ant::unloaded;
            ants_seeds_ptr[i] = seed;
        }
        
        
        pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);
        
        Window win("Ant Simulation", 2*land.dimensions()+10, land.dimensions()+266);
        Renderer renderer(land, phen, pos_nest, pos_food, ants_x, ants_y);
        // Compteur de la quantité de nourriture apportée au nid par les fourmis
        size_t food_quantity = 0;
        SDL_Event event;
        bool cont_loop = true;
        bool not_food_in_nest = true;
        std::size_t it = 0;
        std::ofstream csv("result_vectorized.csv", std::ios::out | std::ios::app);
        
        csv << "iteration,elapsed_seconds\n";
        auto beg = std::chrono::high_resolution_clock::now();
        while (cont_loop) {
            auto start_time = std::chrono::high_resolution_clock::now();
            ++it;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                cont_loop = false;
            }
            advance_time( land, phen, pos_nest, pos_food, ants_x, ants_y, ants_state, ants_seeds, food_quantity );
            renderer.display( win, food_quantity );
            win.blit();
            if ( not_food_in_nest && food_quantity > 0 ) {
                std::cout << "La première nourriture est arrivée au nid a l'iteration " << it << std::endl;
                not_food_in_nest = false;

            }
            
            
            if ( it == total_iterations) cont_loop = false;
            //SDL_Delay(10);
            
            const auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
            csv << it << ',' << elapsed << '\n';
        }

        // csv.close();
        SDL_Quit();
        
        auto end = std::chrono::high_resolution_clock::now();

        // Also write a CSV file that is easy to load with pandas (pd.read_csv)
        csv.close();
        
        std::ofstream total_time_csv("total_time.csv", std::ios::out | std::ios::app);
        total_time_csv << "total_time\n";
        total_time_csv << std::chrono::duration<double>(end - beg).count() << '\n';
        total_time_csv.close();

        return 0;
    }