#include "ant.hpp"
#include <iostream>
#include "rand_generator.hpp"

double ant::m_eps = 0.;

void ant::advance( pheronome& phen, const fractal_land& land, const position_t& pos_food, const position_t& pos_nest,
                   std::size_t& cpteur_food ) 
{
    auto ant_choice = [this]() mutable { return rand_double( 0., 1., this->m_seed ); };
    auto dir_choice = [this]() mutable { return rand_int32( 1, 4, this->m_seed ); };
    double                                   consumed_time = 0.;
    // Tant que la fourmi peut encore bouger dans le pas de temps imparti
    while ( consumed_time < 1. ) {
        // Si la fourmi est chargée, elle suit les phéromones de deuxième type, sinon ceux du premier.
        int        ind_pher    = ( is_loaded( ) ? 1 : 0 );
        double     choix       = ant_choice( );
        position_t old_pos_ant = get_position( );
        position_t new_pos_ant = old_pos_ant;
        double max_phen    = std::max( {phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher],
                                        phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher],
                                        phen( new_pos_ant.x    , new_pos_ant.y - 1 )[ind_pher],
                                        phen( new_pos_ant.x    , new_pos_ant.y + 1 )[ind_pher]} );
        if ( ( choix > m_eps ) || ( max_phen <= 0. ) ) {
            do {
                // CHANGE: Better than 4 ifs, we could use with arithmetic op.
                new_pos_ant = old_pos_ant;
                int d = dir_choice();                
                new_pos_ant.x  +=  (d==3) - (d==1);
                new_pos_ant.y  +=  (d==4) - (d==2);

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

        m_position = new_pos_ant; // La fourmi se déplace
        if ( get_position( ) == pos_nest ) {
            if ( is_loaded( ) ) {
                //@TODO: Multiple access - problem to solve later
                cpteur_food += 1;
            }
            unset_loaded( );
        }
        if ( get_position( ) == pos_food ) {
            set_loaded( );
        }
    }
}