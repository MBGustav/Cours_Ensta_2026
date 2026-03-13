#ifndef _PHERONOME_HPP_
#define _PHERONOME_HPP_
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include "basic_types.hpp"

/**
* @brief Carte des phéronomes
* @details Gère une carte des phéronomes avec leurs mis à jour ( dont l'évaporation )
*
*/
class pheronome {
    public:
    using size_t      = unsigned long;
    using pheronome_t = std::array< double, 2 >;
    
    /**
    * @brief Construit une carte initiale des phéronomes
    * @details La carte des phéronomes est initialisées à zéro ( neutre )
    *          sauf pour les bords qui sont marqués comme indésirables
    *
    * @param dim Nombre de cellule dans chaque direction
    * @param alpha Paramètre de bruit
    * @param beta Paramêtre d'évaporation
    */
    pheronome( size_t dim, const position_t& pos_food, const position_t& pos_nest,
        double alpha = 0.7, double beta = 0.9999 )
        : m_dim( dim ),
        m_stride( dim + 2 ),
        m_alpha(alpha), m_beta(beta),
        m_map_of_pheronome( m_stride * m_stride, {{0., 0.}} ),
        m_buffer_pheronome( ),
        m_pos_nest( pos_nest ),
        m_pos_food( pos_food ) 
        {
            m_map_of_pheronome[index(pos_food)][0] = 1.;
            m_map_of_pheronome[index(pos_nest)][1] = 1.;
            cl_update( );
            m_buffer_pheronome = m_map_of_pheronome;
        }
        pheronome( const pheronome& ) = delete;
        pheronome( pheronome&& )      = delete;
        ~pheronome( )                 = default;
        
        pheronome_t& operator( )( size_t i, size_t j ) {
            return m_map_of_pheronome[( i + 1 ) * m_stride + ( j + 1 )];
        }
        
        const pheronome_t& operator( )( size_t i, size_t j ) const {
            return m_map_of_pheronome[( i + 1 ) * m_stride + ( j + 1 )];
        }
        
        pheronome_t& operator[] ( const position_t& pos ) {
            return m_map_of_pheronome[index(pos)];
        }
        
        const pheronome_t& operator[] ( const position_t& pos ) const {
            return m_map_of_pheronome[index(pos)];
        }
        
        void do_evaporation() {
            const std::size_t dim = m_dim;
            const double beta = m_beta;
            auto* __restrict buffer = m_buffer_pheronome.data();
            
            // Cada linha é um bloco contíguo
            #pragma omp parallel for
            for (std::size_t i = 1; i <= dim; ++i) {
                const std::size_t row_start = i * m_stride + 1;
                
                #pragma GCC ivdep
                #pragma GCC unroll 4
                for (std::size_t j = 0; j < dim; ++j) {
                    std::size_t idx = row_start + j;
                    buffer[idx][0] *= beta;
                    buffer[idx][1] *= beta;
                }
            }
        }
        
        void mark_pheronome(std::size_t i, std::size_t j)
{
    // calcular índice uma vez
    std::size_t idx = (i + 1) * m_stride + (j + 1);

    // acesso direto às células vizinhas
    const auto& left   = (*this)(i - 1, j);
    const auto& right  = (*this)(i + 1, j);
    const auto& up     = (*this)(i, j - 1);
    const auto& down   = (*this)(i, j + 1);

    // valores >= 0
    double v1_left   = left[0]   > 0. ? left[0]   : 0.;
    double v2_left   = left[1]   > 0. ? left[1]   : 0.;
    double v1_right  = right[0]  > 0. ? right[0]  : 0.;
    double v2_right  = right[1]  > 0. ? right[1]  : 0.;
    double v1_up     = up[0]     > 0. ? up[0]     : 0.;
    double v2_up     = up[1]     > 0. ? up[1]     : 0.;
    double v1_down   = down[0]   > 0. ? down[0]   : 0.;
    double v2_down   = down[1]   > 0. ? down[1]   : 0.;

    // branchless max entre 4 valores
    auto max4 = [](double a,double b,double c,double d){
        double m1 = a > b ? a : b;
        double m2 = c > d ? c : d;
        return m1 > m2 ? m1 : m2;
    };

    double v1_max = max4(v1_left,v1_right,v1_up,v1_down);
    double v2_max = max4(v2_left,v2_right,v2_up,v2_down);

    double v1_sum = v1_left + v1_right + v1_up + v1_down;
    double v2_sum = v2_left + v2_right + v2_up + v2_down;

    m_buffer_pheronome[idx][0] = m_alpha * v1_max + (1. - m_alpha) * 0.25 * v1_sum;
    m_buffer_pheronome[idx][1] = m_alpha * v2_max + (1. - m_alpha) * 0.25 * v2_sum;
}

        
        void update( ) {
            m_map_of_pheronome.swap( m_buffer_pheronome );
            cl_update( );
            m_map_of_pheronome[( m_pos_food.x + 1 ) * m_stride + m_pos_food.y + 1][0] = 1;
            m_map_of_pheronome[( m_pos_nest.x + 1 ) * m_stride + m_pos_nest.y + 1][1] = 1;
        }
        
        private:
        size_t index( const position_t& pos ) const
        {
            return (pos.x+1)*m_stride + pos.y + 1;
        }
        /**
        * @brief Mets à jour les conditions limites sur les cellules fantômes
        * @details Mets à jour les conditions limites sur les cellules fantômes :
        *     pour l'instant, on se contente simplement de mettre ces cellules avec
        *     des valeurs à -1 pour être sûr que les fourmis évitent ces cellules
        */
        void cl_update( ) {
            // On mets tous les bords à -1 pour les marquer comme indésirables :
            for ( unsigned long j = 0; j < m_stride; ++j ) {
                m_map_of_pheronome[j]                            = {{-1., -1.}};
                m_map_of_pheronome[j + m_stride * ( m_dim + 1 )] = {{-1., -1.}};
                m_map_of_pheronome[j * m_stride]                 = {{-1., -1.}};
                m_map_of_pheronome[j * m_stride + m_dim + 1]     = {{-1., -1.}};
            }
        }
        unsigned long              m_dim, m_stride;
        double                     m_alpha, m_beta;
        std::vector< pheronome_t > m_map_of_pheronome, m_buffer_pheronome;
        position_t m_pos_nest, m_pos_food;
    };
    
    #endif