#include "ant.hpp"
#include <iostream>
#include "rand_generator.hpp"

double ant::m_eps = 0.;

void advance(pheronome& phen, const fractal_land& land, const position_t& pos_food, const position_t& pos_nest,
             std::vector<int>& ants_x, std::vector<int>& ants_y, std::vector<ant::state>& ants_state,
             std::vector<uint32_t>& ants_seeds, // Cada formiga tem sua semente!
             std::size_t& cpteur_food, double m_eps) 
{
        std::size_t seed_copy;

        // Lamdas
        auto ant_choice = [&seed_copy]() mutable { return rand_double( 0., 1., seed_copy ); };
        auto dir_choice = [&seed_copy]() mutable { return rand_int32( 1, 4, seed_copy ); };
        double consumed_time = 0.;
        // CHANGE: using reduction locally
        const size_t nb_ants = ants_x.size();
        std::size_t local_cpteur_food = 0;
        #pragma omp parallel for reduction(+:local_cpteur_food) private(consumed_time, seed_copy)
        for(size_t i= 0; i<nb_ants; ++i) {
            double consumed_time = 0.;
            uint32_t my_seed = ants_seeds[i]; 
            position_t pos{ants_x[i], ants_y[i]};
            ant::state state = ants_state[i];

            
            // Tant que la fourmi peut encore bouger dans le pas de temps imparti
            while ( consumed_time < 1. ) {
                // Si la fourmi est chargée, elle suit les phéromones de deuxième type, sinon ceux du premier.
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
                            // CHANGE: Better than 4 ifs, we could use with arithmetic op.
                            new_pos_ant = old_pos_ant;
                            int d = dir_choice();                
                            new_pos_ant.x  += (d==3) - (d==1);
                            new_pos_ant.y  += (d==4) - (d==2);
                            
                        } while ( phen[new_pos_ant][ind_pher] == -1 );
                    } else {
                        
                        // On choisit la case où le phéromone est le plus fort.
                        // CHANGE: Better than 4 ifs, we could use with arithmetic op.
                        new_pos_ant.x +=  -(phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher] == max_phen);
                        new_pos_ant.x +=   (phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher] == max_phen);
                        new_pos_ant.y +=  -(phen( new_pos_ant.x, new_pos_ant.y - 1 )[ind_pher] == max_phen);
                        new_pos_ant.y +=   (phen( new_pos_ant.x, new_pos_ant.y + 1 )[ind_pher] == max_phen);
                    }
                    
                    consumed_time += land( new_pos_ant.x, new_pos_ant.y);
                    phen.mark_pheronome( new_pos_ant );
                    
                    // La fourmi se déplace (update position)
                    // m_position = new_pos_ant; 
                    
                    if (new_pos_ant == pos_nest ) {
                        if ( state == ant::state::loaded ) {
                            local_cpteur_food += 1;
                        }
                    }
                    if (new_pos_ant == pos_food ) {
                        state = ant::state::loaded;
                    }

                    // update state and position
                    ants_state[i] = state;
                    ants_x[i] = new_pos_ant.x;
                    ants_y[i] = new_pos_ant.y;
                }
            }
            cpteur_food += local_cpteur_food;
        }