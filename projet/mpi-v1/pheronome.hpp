#ifndef _PHERONOME_HPP_
#define _PHERONOME_HPP_
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include <mpi.h>
#include "basic_types.hpp"

/**
* @brief Carte des phéronomes
* @details Gère une carte des phéronomes avec leurs mis à jour ( dont l'évaporation )
*          Version MPI+OMP : chaque processus possède une copie complète de la carte.
*          Après mark_pheronome(), un MPI_Allreduce(MPI_MAX) fusionne les mises à jour
*          de tous les processus (on prend le maximum, comme spécifié dans le sujet).
*/
class pheronome {
    public:
    using size_t      = unsigned long;
    using pheronome_t = std::array< double, 2 >;
    
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

        // ----------------------------------------------------------------
        // MPI CHANGE: évaporation parallélisée OMP sur une bande de lignes
        //   appartenant à ce processus ( [i_begin, i_end[ )
        // ----------------------------------------------------------------
        void do_evaporation( std::size_t i_begin, std::size_t i_end ) {
            // CHANGE OMP: collapse(2) pour vectoriser les deux boucles imbriquées
            #pragma omp parallel for collapse(2)
            for (std::size_t i = i_begin + 1; i <= i_end; ++i) {
                for (std::size_t j = 1; j <= m_dim; ++j) {
                    std::size_t idx = i * m_stride + j;
                    m_buffer_pheronome[idx][0] *= m_beta;
                    m_buffer_pheronome[idx][1] *= m_beta;
                }
            }
        }

        // ----------------------------------------------------------------
        // MPI CHANGE: après que chaque processus a marqué ses fourmis dans
        //   m_buffer_pheronome, on fusionne via MPI_Allreduce(MPI_MAX) afin
        //   que tous les processus aient la valeur maximale globale pour
        //   chaque cellule (conformément au sujet : on prend le max).
        // ----------------------------------------------------------------
        void sync_buffer_mpi() {
            // Le buffer contient des pheronome_t (2 doubles) en layout contigu.
            // On envoie le tableau brut de doubles.
            std::size_t total = m_stride * m_stride * 2; // 2 doubles par cellule
            MPI_Allreduce(MPI_IN_PLACE,
                          m_buffer_pheronome.data()->data(),
                          static_cast<int>(total),
                          MPI_DOUBLE,
                          MPI_MAX,
                          MPI_COMM_WORLD);
        }
        
        // MPI CHANGE: accès brut au buffer pour MPI_Allreduce
        double* buffer_data() {
            return m_buffer_pheronome.data()->data();
        }

        void mark_pheronome( const position_t& pos ) {
            std::size_t i = pos.x;
            std::size_t j = pos.y;
            assert( i >= 0 );
            assert( j >= 0 );
            assert( i < m_dim );
            assert( j < m_dim );
            pheronome&         phen        = *this;
            const pheronome_t& left_cell   = phen( i - 1, j );
            const pheronome_t& right_cell  = phen( i + 1, j );
            const pheronome_t& upper_cell  = phen( i, j - 1 );
            const pheronome_t& bottom_cell = phen( i, j + 1 );
            double             v1_left     = std::max( left_cell[0], 0. );
            double             v2_left     = std::max( left_cell[1], 0. );
            double             v1_right    = std::max( right_cell[0], 0. );
            double             v2_right    = std::max( right_cell[1], 0. );
            double             v1_upper    = std::max( upper_cell[0], 0. );
            double             v2_upper    = std::max( upper_cell[1], 0. );
            double             v1_bottom   = std::max( bottom_cell[0], 0. );
            double             v2_bottom   = std::max( bottom_cell[1], 0. );
            m_buffer_pheronome[( i + 1 ) * m_stride + ( j + 1 )][0] =
            m_alpha * std::max( {v1_left, v1_right, v1_upper, v1_bottom} ) +
            ( 1 - m_alpha ) * 0.25 * ( v1_left + v1_right + v1_upper + v1_bottom );
            m_buffer_pheronome[( i + 1 ) * m_stride + ( j + 1 )][1] =
            m_alpha * std::max( {v2_left, v2_right, v2_upper, v2_bottom} ) +
            ( 1 - m_alpha ) * 0.25 * ( v2_left + v2_right + v2_upper + v2_bottom );
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
        void cl_update( ) {
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