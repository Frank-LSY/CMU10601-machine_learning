#ifndef TILES_H
#define TILES_H

#include<iostream>
#include<map>
#include<vector>
#include<algorithm>

int hash(const std::vector<int>& v)
{

    /* This is definitely one of the
     * worst possible hash functions.
     *
     * However, we do not expect this to be used */
    std::cerr << "This hash function should not have been called\n";
    throw "This hash function should not have been called";
    if(v.empty())
    {
        return 0;
    }
    unsigned int hash = 1;
    for(auto& x : v)
    {
        hash *= static_cast<unsigned int>(x);
        hash %= static_cast<unsigned int>(1e9 + 7);
    }
    return static_cast<int>(hash);
}

template<typename T>
class IHT
{
    private:
        int size;
        int overfullCount;
        std::map<T, int> dictionary;

    public:
        IHT(int sizeval) : size(sizeval), overfullCount(0), dictionary(){}

        int count() const
        {
            return static_cast<int>(dictionary.size());
        }

        bool fullp() const
        {
            return static_cast<int>(dictionary.size()) >= size;
        }

        int getindex(const T& obj, const bool readonly=false)
        {
            auto it = dictionary.find(obj);
            if(it != dictionary.end())
            {
                return it -> second;
            }
            else if(readonly)
            {
                return -1;
            }

            if(count() >= size)
            {
                if(overfullCount == 0)
                {
                    std::cout << "IHT full, starting to allow collisions\n";
                }
                overfullCount++;
                return hash(obj) % size;
            }
            else
            {
                auto cnt = count();
                dictionary[obj] = cnt;
                return cnt;
            }
        }

        std::ostream& to_string(std::ostream& os) const
        {
            os << "Collision table: size:" << size 
               << " overfullCount:" << overfullCount
               << " dictionary:" << dictionary.size() << " items";
            return os;
        }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const IHT<T>& iht)
{
    return iht.to_string(os);
}

template<typename T>
int hashcoords(const T& coordinates, IHT<T>& m, const bool readonly=false)
{
    return m.getindex(coordinates, readonly);
}

template<typename T>
std::vector<int> tiles(T& ihtORsize, const int numtilings, const std::vector<double> floats, const std::vector<int> ints={}, const bool readonly=false)
{
    std::vector<int> qfloats(floats.size());
    std::transform(floats.begin(), floats.end(), qfloats.begin(),
            [numtilings](double f){return static_cast<int>(f * numtilings);});
    std::vector<int> Tiles;
    for(int tiling = 0; tiling < numtilings; tiling++)
    {
        auto tilingX2 = tiling * 2;
        std::vector<int> coords {tiling};
        auto b = tiling;
        for(auto& q : qfloats)
        {
            coords.push_back((q + b) / numtilings);
            b += tilingX2;
        }
        for(auto& x: ints)
        {
            coords.push_back(x);
        }
        Tiles.push_back(hashcoords(coords, ihtORsize, readonly));
    }
    return Tiles;
}

#endif
