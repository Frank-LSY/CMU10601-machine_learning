#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include"tiles.h"
#include<array>
#include<string>
#include<random>
#include<cmath>
#include<cassert>

template<typename T>
T clip(T val, T low, T high)
{
    if(val < low)
    {
        return low;
    }
    if(val > high)
    {
        return high;
    }
    return val;
}

template<typename T>
class MountainCar
{
    private:
        double min_position;
        double max_position;
        double max_speed;
        double goal_position;
        double force;
        double gravity;
        std::array<double, 2> low;
        std::array<double, 2> high;
        std::array<double, 2> state;
        int action_space;
        int default_action;
        int state_space;
        std::string mode;

        IHT<std::vector<int>> iht;
        
        // RNG
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution;

    public:
        decltype(state_space) get_state_space() const
        {
            return state_space;
        }

        decltype(state_space) get_action_space() const
        {
            return action_space;
        }

        MountainCar(std::string mode) : min_position(-1.2),
                                        max_position(0.6),
                                        max_speed(0.07),
                                        goal_position(0.5),
                                        force(0.001),
                                        gravity(0.0025),
                                        action_space(3),
                                        default_action(1),
                                        mode(mode),
                                        iht(2048),
                                        distribution(-0.6, -0.4)
        {
            if(mode == "tile")
            {
                state_space = 2048;
            }
            else if(mode == "raw")
            {
                state_space = 2;
            }
            else
            {
                auto err = "Invalid environment mode. Must be tile or raw";
                std::cerr << err << "\n";
                throw err;
            }
            seed();
            reset();
        }

        void seed(int seed = 0)
        {
            generator.seed(seed);            
        }

        std::map<int, T> reset()
        {
            state[0] = distribution(generator);
            state[1] = 0;
            return transform();
        }

        std::map<int, T> transform()
        {
            auto position = (state[0] + 1.2) / 1.8;
            auto velocity = (state[1] + 0.07) / 0.14;
            assert(position >= 0);
            assert(position <= 1);
            assert(velocity >= 0);
            assert(velocity <= 1);
            position *= 2;
            velocity *= 2;
            if(mode == "tile")
            {
                std::vector<int> v1 = tiles(iht, 64, std::vector<double>{position, velocity}, std::vector<int> {0});
                auto v2 = tiles(iht, 64, std::vector<double>{position}, std::vector<int> {1});
                auto v3 = tiles(iht, 64, std::vector<double>{velocity}, std::vector<int> {2});

                std::map<int, T> retval;
                for(auto& x : v1)
                {
                    retval[x] = 1;
                }
                for(auto& x : v2)
                {
                    retval[x] = 1;
                }
                for(auto& x : v3)
                {
                    retval[x] = 1;
                }
                return retval;
            }
            else if(mode == "raw")
            {
                std::map<int, T> retval;
                retval[0] = state[0];
                retval[1] = state[1];
                return retval;
            }
            else
            {
                // Ensuring that there is no warning
                std::cerr << "Invalid environment mode. Must be tile or raw\n";
                throw "Invalid environment mode. Must be tile or raw";
            }
        }

        std::tuple<std::map<int, T>, double, bool> step(const int action)
        {
            assert(action == 0 or action == 1 or action == 2);

            auto position = state[0];
            auto velocity = state[1];
            velocity += (action - 1) * force - cos(3 * position) * (gravity);
            velocity = clip(velocity, -max_speed, max_speed);
            position += velocity;
            position = clip(position, min_position, max_position);

            if(position == min_position and velocity < 0)
            {
                velocity = 0;
            }

            bool done = (position >= goal_position);
            double reward = -1.0;

            state[0] = position;
            state[1] = velocity;
            return make_tuple(transform(), reward, done);
        }
};

#endif
