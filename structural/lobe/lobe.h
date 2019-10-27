#ifndef LOBE
#define LOBE

#include <vector>
#include <algorithm>

#include "node.h"

class lobe{
    std::vector<lobe*> connections;
    std::vector<node*> nodes;

public:
    void addConnection(lobe* n){
        connections.push_back(n);
    }
    void removeConnection(lobe* n){
        connections.erase(std::remove(connections.begin(), connections.end(), n), connections.end());
    }
};

#endif