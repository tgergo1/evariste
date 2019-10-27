#ifndef NODE
#define NODE

#include <vector>
#include <algorithm>

class node{
    std::vector<node*> connections;

public:
    void addConnection(node* n){
        connections.push_back(n);
    }
    void removeConnection(node* n){
        connections.erase(std::remove(connections.begin(), connections.end(), n), connections.end());
    }
};

#endif