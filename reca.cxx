#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "mpi.h"

#include "util/arguments.hxx"

#include "mpi/mpi_ant_colony_optimization_new.hxx"
#include "mpi/mpi_particle_swarm.hxx"
#include "mpi/mpi_differential_evolution.hxx"

#include "asynchronous_algorithms/particle_swarm.hxx"
#include "asynchronous_algorithms/differential_evolution.hxx"


class Lattice2D {
    private:
        uint32_t height;
        uint32_t width;

        vector< vector<double> > theta;
        vector< vector< vector<double> > > A;
        vector< vector< vector<double> > > J;

        const static uint32_t RIGHT = 0;
        const static uint32_t DOWN = 1;
        const static uint32_t LEFT = 2;
        const static uint32_t UP = 3;

    public:

        Lattice2D(int _height, int _width) {
            height = _height;
            width = _width;

            theta = vector< vector<double> >(height, vector<double>(width, 0.0));

            A = vector< vector< vector<double> > >(height, vector< vector<double> >(width, vector<double>(2, 0.0)));
            for (uint32_t y = 0; y < A.size(); y++) {
                for (uint32_t x = 0; x < A[y].size(); x++) {
                    A[y][x][RIGHT] = (x * y) - ((x + 1) * y);
                    A[y][x][DOWN] = (x * (y + 1)) - (x * y);
                }
            }

            J = vector< vector< vector<double> > >(height, vector< vector<double> >(width, vector<double>(2, 1.0)));

        }

        double get_A(uint32_t y, uint32_t x, uint32_t direction) {
            switch(direction) {
                case RIGHT:
                    return A[y][x][RIGHT];
                case DOWN:
                    return A[y][x][DOWN];
                case LEFT:
                    return -A[y][x - 1][RIGHT];
                case UP:
                    return -A[y - 1][x][DOWN];
                default:
                    cerr << "ERROR, unknown direction for getting A[" << y << "][" << x << "]: " << direction << endl;
                    exit(1);
            }
        }

        double get_J(uint32_t y, uint32_t x, uint32_t direction) {
            switch(direction) {
                case RIGHT:
                    return J[y][x][RIGHT];
                case DOWN:
                    return J[y][x][DOWN];
                case LEFT:
                    return -J[y][x - 1][RIGHT];
                case UP:
                    return -J[y - 1][x][DOWN];
                default:
                    cerr << "ERROR, unknown direction for getting J[" << y << "][" << x << "]: " << direction << endl;
                    exit(1);
            }
        }

        double get_energy(uint32_t y, uint32_t x, uint32_t direction) {
            switch(direction) {
                case RIGHT:
                    return get_J(y, x, RIGHT) * cos(theta[y][x] - theta[y][x + 1] - get_A(y, x, RIGHT));
                case DOWN:
                    return get_J(y, x, DOWN) * cos(theta[y][x] - theta[y + 1][x] - get_A(y, x, DOWN));
                case LEFT:
                    return get_J(y, x, LEFT) * cos(theta[y][x] - theta[y][x - 1] - get_A(y, x, LEFT));
                case UP:
                    return get_J(y, x, UP) * cos(theta[y][x] - theta[y - 1][x] - get_A(y, x, UP));
                default:
                    cerr << "ERROR, unknown direction for getting energy[" << y << "][" << x << "]: " << direction << endl;
                    exit(1);
            }
        }

        double get_energy() {
            double energy = 0.0;

            //four corners
            //top left
            energy -= get_energy(0, 0, RIGHT);
            energy -= get_energy(0, 0, DOWN);

            //top right
            energy -= get_energy(0, width - 1, LEFT);
            energy -= get_energy(0, width - 1, DOWN);

            //bottom left
            energy -= get_energy(height - 1, 0, RIGHT);
            energy -= get_energy(height - 1, 0, UP);

            //bottom right
            energy -= get_energy(height - 1, width - 1, LEFT);
            energy -= get_energy(height - 1, width - 1, UP);

            //left
            uint32_t x = 0;
            for (uint32_t y = 1; y < height - 1; y++) {
                energy -= get_energy(y, x, RIGHT);
                energy -= get_energy(y, x, DOWN);
                energy -= get_energy(y, x, UP);
            }

            //top
            uint32_t y = 0;
            for (uint32_t x = 1; x < width; x++) {
                energy -= get_energy(y, x, RIGHT);
                energy -= get_energy(y, x, LEFT);
                energy -= get_energy(y, x, DOWN);
            }

            //right
            x = width - 1;
            for (uint32_t y = 1; y < height - 1; y++) {
                energy -= get_energy(y, x, LEFT);
                energy -= get_energy(y, x, DOWN);
                energy -= get_energy(y, x, UP);
            }

            //bottom
            y = height - 1;
            for (uint32_t x = 1; x < width - 1; x++) {
                energy -= get_energy(y, x, RIGHT);
                energy -= get_energy(y, x, LEFT);
                energy -= get_energy(y, x, UP);
            }

            //center values
            for (uint32_t y = 1; y < height - 1; y++) {
                for (uint32_t x = 1; x < width - 1; x++) {
                    energy -= get_energy(y, x, RIGHT);
                    energy -= get_energy(y, x, DOWN);
                    energy -= get_energy(y, x, LEFT);
                    energy -= get_energy(y, x, UP);
                }
            }

            return energy;
        }

        void set_thetas(const vector<double> &parameters) {
            if (parameters.size() != width * height) {
                cerr << "ERROR, parameters.size(): " << parameters.size() << "!= width * height: " << width * height << endl;
                exit(1);
            }

            uint32_t current = 0;
            for (uint32_t y = 0; y < height; y++) {
                for (uint32_t x = 0; x < width; x++) {
                    theta[y][x] = parameters[current];
                    current++;
                }
            }
        }
};

Lattice2D *lattice2d = NULL;

double objective_function(const vector<double> &parameters) {
    lattice2d->set_thetas(parameters);
    return -lattice2d->get_energy();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    vector<string> arguments = vector<string>(argv, argv + argc);

    uint32_t height;
    get_argument(arguments, "--height", true, height);

    uint32_t width;
    get_argument(arguments, "--width", true, width);

    uint32_t number_of_weights = height * width;

    vector<double> min_bound(number_of_weights, 0 * M_PI); 
    vector<double> max_bound(number_of_weights, 2.0 * M_PI); 

    lattice2d = new Lattice2D(height, width);

    string search_type;
    get_argument(arguments, "--search_type", true, search_type);

    if (search_type.compare("ps") == 0) {
        ParticleSwarm ps(min_bound, max_bound, arguments);
        ps.iterate(objective_function);

    } else if (search_type.compare("de") == 0) {
        DifferentialEvolution de(min_bound, max_bound, arguments);
        de.iterate(objective_function);

    } else if (search_type.compare("ps_mpi") == 0) {
        ParticleSwarmMPI ps(min_bound, max_bound, arguments);
        ps.go(objective_function);

    } else if (search_type.compare("de_mpi") == 0) {
        DifferentialEvolutionMPI de(min_bound, max_bound, arguments);
        de.go(objective_function);

    } else {
        cerr << "Improperly specified search type: '" << search_type.c_str() <<"'" << endl;
        cerr << "Possibilities are:" << endl;
        cerr << "    de     -       differential evolution" << endl;
        cerr << "    ps     -       particle swarm optimization" << endl;
        cerr << "    de_mpi -       MPI parallel differential evolution" << endl;
        cerr << "    ps_mpi -       MPI parallel particle swarm optimization" << endl;
        exit(1);
    }
}
