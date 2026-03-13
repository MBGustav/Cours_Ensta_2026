#include "ant.hpp"
#include <iostream>
#include "rand_generator.hpp"

double ant::m_eps = 0.;

void advance(pheronome& phen, const fractal_land& land, const position_t& pos_food, const position_t& pos_nest,
             std::vector<int>& ants_x, std::vector<int>& ants_y, std::vector<ant::state>& ants_state,
             std::vector<uint32_t>& ants_seeds,
             std::size_t& cpteur_food, double m_eps)
{
    const size_t nb_ants = ants_x.size();
    std::size_t local_cpteur_food = 0;

    // lookup table direção
    static const int dx[5] = {0,-1,0,1,0};
    static const int dy[5] = {0,0,-1,0,1};

    // ponteiros restrict para vetorizar melhor
    int* __restrict ants_x_ptr = ants_x.data();
    int* __restrict ants_y_ptr = ants_y.data();
    ant::state* __restrict ants_state_ptr = ants_state.data();
    uint32_t* __restrict ants_seeds_ptr = ants_seeds.data();

#pragma omp parallel reduction(+:local_cpteur_food)
    {
        std::size_t thread_local_cpteur = 0;
    #pragma omp for
        for(size_t i = 0; i < nb_ants; ++i)
        {
            double consumed_time = 0.;

            size_t seed_copy = ants_seeds_ptr[i];

            int xi = ants_x_ptr[i];
            int yi = ants_y_ptr[i];

            ant::state state = ants_state_ptr[i];

            while(consumed_time < 1.)
            {
                int ind_pher = (state == ant::state::loaded ? 1 : 0);

                // RNG inline, sem lambda
                double choix = rand_double(0.,1.,seed_copy);

                // acessar feromônio só uma vez por célula
                double phen_d = phen(xi-1, yi)[ind_pher];
                double phen_u = phen(xi+1, yi)[ind_pher];
                double phen_l = phen(xi, yi-1)[ind_pher];
                double phen_h = phen(xi, yi+1)[ind_pher];

                double max_phen = std::max(std::max(phen_d,phen_u),
                                           std::max(phen_l,phen_h));

                int new_x = xi;
                int new_y = yi;

                if((choix > m_eps) || (max_phen <= 0.))
                {
                    // branchless escolha aleatória
                    do
                    {
                        int d = rand_int32(1,4,seed_copy);
                        new_x = xi + dx[d];
                        new_y = yi + dy[d];
                    } while(phen(new_x,new_y)[ind_pher] == -1);
                }
                else
                {
                    // branchless feromônio máximo
                    new_x += (phen_u == max_phen) - (phen_d == max_phen);
                    new_y += (phen_h == max_phen) - (phen_l == max_phen);
                }

                consumed_time += land(new_x,new_y);

                phen.mark_pheronome(new_x,new_y);

                if(new_x == pos_nest.x && new_y == pos_nest.y)
                    if(state == ant::state::loaded)
                        thread_local_cpteur += 1;

                if(new_x == pos_food.x && new_y == pos_food.y)
                    state = ant::state::loaded;

                xi = new_x;
                yi = new_y;
            }

            ants_x_ptr[i] = xi;
            ants_y_ptr[i] = yi;
            ants_state_ptr[i] = state;
            ants_seeds_ptr[i] = seed_copy;
        }

        local_cpteur_food += thread_local_cpteur;
    }

    cpteur_food += local_cpteur_food;
}
