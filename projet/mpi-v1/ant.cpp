#include "ant.hpp"
#include <iostream>
#include "rand_generator.hpp"


void advance(pheronome& phen, const fractal_land& land,
    const position_t& pos_food, const position_t& pos_nest,
    std::vector<int>& ants_x, std::vector<int>& ants_y,
    std::vector<ant::state>& ants_state,
    std::vector<uint32_t>& ants_seeds,
    std::size_t& cpteur_food, double m_eps,
    // MPI CHANGE: tranche locale de fourmis
    std::size_t local_begin, std::size_t local_end, size_t iter)
    {
        
        
        std::size_t seed_copy=0;// Graine privée par fourmi, initialisée à partir du tableau d'entrée
        
        // Lambdas (inchangées)
        auto ant_choice = [&seed_copy]() mutable { return rand_double( 0., 1., seed_copy ); };
        auto dir_choice = [&seed_copy]() mutable { return rand_int32( 1, 4, seed_copy ); };
        double consumed_time = 0.;
        
        // CHANGE MPI: itère seulement sur [local_begin, local_end[
        //             au lieu de [0, nb_ants[
        std::size_t local_count = local_end - local_begin;
        std::size_t local_cpteur_food = 0;
        
        #pragma omp parallel for reduction(+:local_cpteur_food) private(consumed_time, seed_copy)
        for(size_t k = 0; k < local_count; ++k) {
            // CHANGE MPI: indice global de la fourmi = local_begin + k
            size_t i = local_begin + k;
            
            double consumed_time = 0.;
            uint32_t my_seed = ants_seeds[i]; 
            ant::state state = ants_state[i];
            
            seed_copy = my_seed;
            
            // Tant que la fourmi peut encore bouger dans le pas de temps imparti
            while ( consumed_time < 1. ) {
                int        ind_pher    = ( state == ant::state::loaded ? 1 : 0 );
                double     choix       = ant_choice( );
                position_t old_pos_ant = position_t{ants_x[i], ants_y[i]};
                position_t new_pos_ant = old_pos_ant;
                
                double phen_d =  phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher];
                double phen_u =  phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher];
                double phen_l =  phen( new_pos_ant.x    , new_pos_ant.y - 1 )[ind_pher];
                double phen_h =  phen( new_pos_ant.x    , new_pos_ant.y + 1 )[ind_pher];
                double max_phen = std::max( {phen_d, phen_u, phen_l, phen_h} );
                
                if ( ( choix > m_eps ) || ( max_phen <= 0. ) ) {
                    do {
                        new_pos_ant = old_pos_ant;
                        int d = dir_choice();                
                        new_pos_ant.x  += (d==3) - (d==1);
                        new_pos_ant.y  += (d==4) - (d==2);
                    } while ( phen[new_pos_ant][ind_pher] == -1 );
                } else {
                    new_pos_ant.x +=  -(phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher] == max_phen);
                    new_pos_ant.x +=   (phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher] == max_phen);
                    new_pos_ant.y +=  -(phen( new_pos_ant.x, new_pos_ant.y - 1 )[ind_pher] == max_phen);
                    new_pos_ant.y +=   (phen( new_pos_ant.x, new_pos_ant.y + 1 )[ind_pher] == max_phen);
                }
                
                consumed_time += land( new_pos_ant.x, new_pos_ant.y);
                phen.mark_pheronome( new_pos_ant );
                
                if (new_pos_ant == pos_nest ) {
                    if ( state == ant::state::loaded ) {
                        local_cpteur_food += 1;
                    }
                }
                if (new_pos_ant == pos_food ) {
                    state = ant::state::loaded;
                }
                
                ants_state[i] = state;
                ants_x[i]     = new_pos_ant.x;
                ants_y[i]     = new_pos_ant.y;
            }
            // CHANGE MPI: sauvegarde de la graine mise à jour par cette fourmi
            ants_seeds[i] = static_cast<uint32_t>(seed_copy);
        }
        cpteur_food += local_cpteur_food;
    }