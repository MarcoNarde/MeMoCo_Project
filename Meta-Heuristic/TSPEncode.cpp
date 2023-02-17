#include "TSPEncode.h"

TSPEncode::TSPEncode(int n): cost(0){sequence.reserve(n);}

TSPEncode::TSPEncode( const TSP& tsp){
  cost = 0;
  sequence.reserve(tsp.numberNodes + 1);
  for ( int i = 0; i < tsp.numberNodes ; ++i ) {
    sequence.push_back(i);
  }
  sequence.push_back(0);

  
  for(size_t i = 0; i < sequence.size()-1; i++){
    cost += tsp.values[sequence[i]][sequence[i+1]];
  }
}

TSPEncode::TSPEncode( const TSPEncode& tspSol ) {
    sequence.reserve(tspSol.sequence.size());
    for (size_t i = 0; i < tspSol.sequence.size(); ++i ) {
      sequence.push_back(tspSol.sequence[i]);
    }
    cost = tspSol.cost;
}

TSPEncode::TSPEncode( const std::vector<int>& seq , double& weight) : sequence(seq), cost(weight)
{}

TSPEncode& TSPEncode::operator=(const TSPEncode& right) {
  // Handle self-assignment:
  if(this == &right) return *this;
  for ( size_t i = 0; i < sequence.size(); i++ ) {
    sequence[i] = right.sequence[i];
  }
  cost = right.cost;
  return *this;
}

std::ostream& operator<<(std::ostream &out, const TSPEncode &obj) {
    for (auto i : obj.sequence) {
        out << i << " ";
    }
    out << " COST: " << obj.cost<< "\n";
    return out;
}

bool TSPEncode::operator==(const TSPEncode &other) const{
  if (this->sequence.size() != other.sequence.size())
    return false;
  for (size_t i = 0; i < this->sequence.size(); i++)
  {
    if (this->sequence[i] != other.sequence[i])
      return false;
  }
  return true;
}