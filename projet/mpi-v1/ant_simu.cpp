#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <algorithm>
#include <string>
#include <cstdint>
#include <mpi.h>
#include "fractal_land.hpp"
#include "ant.hpp"

#include "pheronome.hpp"
# include "renderer.hpp"
# include "window.hpp"
# include "rand_generator.hpp"

static double eps = 0.8;  // Coefficient d'exploration
constexpr size_t total_iterations = 4000;

// ============================================================
// MPI CHANGE: advance_time now accepts the local slice of ants
//   [local_begin, local_end[ and the evaporation band
//   [evap_begin, evap_end[ assigned to this MPI rank.
//   After ant movement, buffers are merged via MPI_Allreduce
//   (MPI_MAX), then evaporation is applied only on the local
//   band (OMP-parallelised), then a second MPI_Allreduce
//   propagates the evaporated values to all ranks.
// ============================================================
void advance_time( const fractal_land& land, pheronome& phen,
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<int>& ants_x,
                   std::vector<int>& ants_y,
                   std::vector<ant::state>& ants_state,
                   std::vector<uint32_t>& ants_seeds,
                   std::size_t& cpteur,
                   // MPI CHANGE: local ant slice indices
                   std::size_t local_ant_begin, std::size_t local_ant_end,
                   // MPI CHANGE: local evaporation band (row indices in [0, dim[)
                   std::size_t evap_begin, std::size_t evap_end, size_t iter,size_t sync_interval)
{
    // --- Phase 1 : mouvement des fourmis locales (OMP inside advance()) ---
    // CHANGE: only advance the slice of ants owned by this rank
    advance(phen, land, pos_food, pos_nest,
            ants_x, ants_y, ants_state, ants_seeds,
            cpteur, eps,
            local_ant_begin, local_ant_end, iter);

    // --- Phase 2 : MPI_Allreduce(MAX) sur le buffer des phéromones ---
    // CHANGE MPI: merge mark_pheronome writes from all ranks; the spec says
    phen.sync_buffer_mpi();

    // --- Phase 3 : évaporation sur la bande locale (OMP) ---
    // CHANGE MPI: each rank evaporates only its assigned rows
    phen.do_evaporation(evap_begin, evap_end);

    // --- Phase 4 : MPI_Allreduce(MAX) pour diffuser l'évaporation ---
    {
        std::size_t total = (land.dimensions()+2) * (land.dimensions()+2) * 2;
        MPI_Allreduce(MPI_IN_PLACE,
                      phen.buffer_data(),
                      static_cast<int>(total),
                      MPI_DOUBLE,
                      MPI_MIN,
                      MPI_COMM_WORLD);
    }

    // --- Phase 5 : swap buffer → map, restauration des constantes ---
    phen.update();
}

int main(int nargs, char* argv[])
{
    // --------------------------------------------------------
    // MPI CHANGE: initialisation MPI
    // --------------------------------------------------------
    MPI_Init(&nargs, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    std::cout << "MPI rank " << mpi_rank << " / " << mpi_size << " started." << std::endl;
    // Only rank 0 opens the SDL window
    if (mpi_rank == 0) SDL_Init( SDL_INIT_VIDEO );

    std::size_t seed = 2026;    // Graine pour la génération aléatoire ( reproductible )
    const int nb_ants = 5000;   // Nombre de fourmis
    const double eps = 0.8;     // Coefficient d'exploration
    const double alpha=0.7;     // Coefficient de chaos
    const double beta=0.999;    // Coefficient d'évaporation

    position_t pos_nest{256,256};
    position_t pos_food{500,500};

    // Chaque processus génère le paysage fractal complet (reproductible,
    // même graine : même résultat sur tous les rangs)
    fractal_land land(8,2,1.,1024);
    double max_val = 0.0;
    double min_val = 0.0;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }
    double delta = max_val - min_val;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j )
            land(i,j) = (land(i,j)-min_val)/delta;

    ant::set_exploration_coef(eps);

    // CHANGE MPI: SoA layout (unchanged variable names)
    std::vector<int>        ants_x(nb_ants);
    std::vector<int>        ants_y(nb_ants);
    std::vector<ant::state> ants_state(nb_ants);
    std::vector<uint32_t>   ants_seeds(nb_ants);

    auto gen_ant_pos = [&land, &seed] () { return rand_int32(0, land.dimensions()-1, seed); };
    for ( size_t i = 0; i < (size_t)nb_ants; ++i ){
        position_t pos{gen_ant_pos(), gen_ant_pos()};
        ants_x[i]     = pos.x;
        ants_y[i]     = pos.y;
        ants_state[i] = ant::unloaded;
        ants_seeds[i] = seed;
    }

    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    // --------------------------------------------------------
    // MPI CHANGE: partition des fourmis entre les rangs
    //   Rang r gère les fourmis dans [local_ant_begin, local_ant_end[
    // --------------------------------------------------------
    std::size_t local_ant_begin = (std::size_t) mpi_rank      * nb_ants / mpi_size;
    std::size_t local_ant_end   = (std::size_t)(mpi_rank + 1) * nb_ants / mpi_size;

    // --------------------------------------------------------
    // MPI CHANGE: partition des lignes pour l'évaporation
    //   Rang r évapore les lignes dans [evap_begin, evap_end[
    // --------------------------------------------------------
    std::size_t dim = land.dimensions();
    std::size_t evap_begin = (std::size_t)mpi_rank       * dim / mpi_size;
    std::size_t evap_end   = (std::size_t)(mpi_rank + 1) * dim / mpi_size;

    // SDL + renderer uniquement sur le rang 0
    Window*   win      = nullptr;
    Renderer* renderer = nullptr;
    if (mpi_rank == 0) {
        win      = new Window("Ant Simulation", 2*land.dimensions()+10, land.dimensions()+266);
        renderer = new Renderer(land, phen, pos_nest, pos_food, ants_x, ants_y);
    }

    std::uint64_t food_quantity = 0;
    bool   cont_loop        = true;
    bool   not_food_in_nest = true;
    std::size_t it = 0;
    
    // --------------------------------------------------------
    // Timing MPI : mesure des phases séparément
    // --------------------------------------------------------
    double t_ant_total   = 0.;
    double t_sync_total  = 0.;
    double t_evap_total  = 0.;
    
    SDL_Event event;
    
    std::ofstream timing_file("timing_rank_" + std::to_string(mpi_rank) + ".txt");
    while (cont_loop) {
        ++it;

        if (mpi_rank == 0) {
            while (SDL_PollEvent(&event))
                if (event.type == SDL_QUIT) cont_loop = false;
            
        }
        // Broadcast loop-control from rank 0 to all
        // CHANGE MPI: synchronise la variable cont_loop sur tous les rangs
        int cont_int = cont_loop ? 1 : 0;
        MPI_Bcast(&cont_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        cont_loop = (cont_int == 1);
        if (!cont_loop) break;

        // --- Timed ant movement phase ---
        double t0 = MPI_Wtime();
        // Advance only local ants (OMP threads inside advance())
        advance(phen, land, pos_food, pos_nest,
                ants_x, ants_y, ants_state, ants_seeds,
                food_quantity, eps,
                local_ant_begin, local_ant_end, it);
        double t1 = MPI_Wtime();
        t_ant_total += (t1 - t0);

        // --- Timed pheromone sync (MPI_Allreduce MAX) ---
        t0 = MPI_Wtime();
        phen.sync_buffer_mpi();
        t1 = MPI_Wtime();
        t_sync_total += (t1 - t0);

        // --- Timed evaporation (OMP on local band) ---
        t0 = MPI_Wtime();
        phen.do_evaporation(evap_begin, evap_end);
        t1 = MPI_Wtime();
        t_evap_total += (t1 - t0);

        // Second Allreduce(MIN) to propagate evaporated values everywhere
        // Allreduce(MIN) on full buffer (simpler, correct, but plus de données échangées)
        std::size_t total = (land.dimensions()+2) * (land.dimensions()+2) * 2;
        MPI_Allreduce(MPI_IN_PLACE,
                        phen.buffer_data(),
                        static_cast<int>(total),
                        MPI_DOUBLE,
                        MPI_MIN,
                        MPI_COMM_WORLD);
        phen.update();

        // Reduce food_quantity to rank 0
        // CHANGE MPI: agrège le compteur de nourriture depuis tous les rangs
        std::uint64_t global_food = 0;
        MPI_Reduce(&food_quantity, &global_food, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            food_quantity = global_food;
            renderer->display(*win, food_quantity);
            win->blit();

            if (not_food_in_nest && food_quantity > 0) {
                std::cout << "La première nourriture est arrivée au nid a l'iteration " << it << std::endl;
                not_food_in_nest = false;
            }
        }

        if (it == total_iterations) cont_loop = false;
        double loop_time = t_ant_total + t_sync_total + t_evap_total;
        // CSV output (pandas-friendly)
        if (it == 1) {
            timing_file << "iter,rank,ant_move_s,phen_sync_s,evaporation_s,total_s\n";
        }
        timing_file << it << ","
                << mpi_rank << ","
                << t_ant_total << ","
                << t_sync_total << ","
                << t_evap_total << ","
                << loop_time
                << "\n";
    }

    timing_file.close();


    if (mpi_rank == 0) {
        delete renderer;
        delete win;
        SDL_Quit();
    }

    MPI_Finalize();
    return 0;
}