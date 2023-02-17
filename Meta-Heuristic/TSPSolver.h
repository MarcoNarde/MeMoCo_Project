/**
 * @file TSP.h
 * @brief TSP data file
 *
 */

#ifndef TSPSOLVER_H
#define TSPSOLVER_H

#include <random>
#include <chrono>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "TSPEncode.h"


// Define the hash function that allows to use the unordere_set with the sequences for the
// solutions
struct hashFunction 
{
  size_t operator()(const std::vector<int> &myVector) const 
  {
    std::hash<int> hasher;
    size_t answer = 0;
    for (int i : myVector) 
    {
      answer ^= hasher(i) + 0x9e3779b9 + (answer << 6) + (answer >> 2);
    }
    return answer;
  }
};

class TSPSolver
{
private:

// Create random number generator
std::mt19937 gen;

// Create uniform distribution between 0 e 1
std::uniform_real_distribution<double> dist;

const TSP& tsp;
std::vector<TSPEncode> population;
const std::vector<int>& initialSequence;
const int& populationSize;
const int& maxIteration;
const int& maxNumberNotImprovingIteration;
const int& maxTime;
const int& selectedIndividualsNumber;
const double& probUseOrdCrossover;
const double& probMutation;
const double& probFastOpt;
const int& maxIterationFastLS;
const int& hammingDistanceThreshold;
const bool& useFinalOpt;
const bool& useLinearRanking;
const int& tournamentSize;

public:
  
  /** Constructor 
  * initialize the TSPSolver attributes
  * @param tsp: the instance for the tsp problem
  * @param sequence: the initial sequence e.i [0-1-...n-0]
  * @param populationSize: the size for the population
  * @param maxIteration: max iteration number that algorithm can compute
  * @param maxNumberNotImprovingIteration: max interation number without (consecutive) improvement
  * @param maxTime: maximum time of execution for the algorithm
  * @param selectedIndividualsNumber: number of individuals to select at each iteration
  * @param probUseOrdCrossover: probability of using ordered crossover instead of PBX crossover
  * @param probMutation: probability of applying mutation to offsprings
  * @param probFastOpt: probability of applying fast local search to offsprings
  * @param maxIterationFastLS: the max number of iteteration in fast local search
  * @param avgHammingDistance: threshold for the AVG hamming distance in the population
  * @param useFinalOpt: 0 = not using final optimization, 1 = use it
  * @param useLinearRanking: 0 = use tournament selection, 1 = use linear ranking
  * @param tournamentSize: size of the tournament's groups
  */
  TSPSolver ( const TSP& tsp, const std::vector<int>& sequence, const int& populationSize, const int& maxIteration,
              const int& maxNumberNotImprovingIteration, const int& maxTime, const int& selectedIndividualsNumber, const double& probUseOrdCrossover, const double& probMutation,
              const double& probFastOpt, const int& maxIterationFastLS, const int& avgHammingDistance, const bool& useFinalOpt , const bool& useLinearRanking, const int& tournamentSize):
  tsp(tsp),
  initialSequence(sequence),
  populationSize(populationSize),
  maxIteration(maxIteration),
  maxNumberNotImprovingIteration(maxNumberNotImprovingIteration),
  maxTime(maxTime),
  selectedIndividualsNumber(selectedIndividualsNumber),
  probUseOrdCrossover(probUseOrdCrossover),
  probMutation(probMutation),
  probFastOpt(probFastOpt),
  maxIterationFastLS(maxIterationFastLS),
  hammingDistanceThreshold(avgHammingDistance),
  useFinalOpt(useFinalOpt),
  useLinearRanking(useLinearRanking),
  tournamentSize(tournamentSize) {
    gen = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count() ^ random_device{}());
    dist = std::uniform_real_distribution<double>(0.0, 1.0);
  };

  /**
   * Start genetic algorithm to solve the TSP problem
   * @return cost of the solution
   */
  double solve ();

  // Main Genetic Algorithm functions
  /**
   * Generate the initial population for the algorithm
  */
  void generate_population();

  /**
   * Select every time a group of individuals in the population using Linear Ranking || Tournament Selection
   * @param selectedIndividuals: vector that will contain the individuals selected
  */
  void select_solutions(std::vector<TSPEncode> &selectedIndividuals);
  
  /**
   * Implement Tournament Selection Criterio for the selection algoritithm
   * @param partecipant: vector containing the individual to select
   * @param selectedIndividuals: vector that will contain the individuals selected
   * @param selectedIndividualsNumber: number of individuals to select
   * @param tournamentSize: size of the group to form each tournament
  */
  void tournament_selection(std::vector<TSPEncode> &partecipant, std::vector<TSPEncode> &selectedIndividuals, const int& selectedIndividualsNumber, int tournamentSize);

  /**
   * Implement Linear Ranking Criterio for the selection algoritithm
   * @param partecipant: vector containing the individual to select
   * @param selectedIndividuals: vector that will contain the individuals selected
   * @param selectedIndividualsNumber: number of individuals to select
  */
  void linear_ranking_selection(std::vector<TSPEncode> &partecipant, std::vector<TSPEncode> &selectedIndividuals, const int& selectedIndividualsNumber);

  /**
   * Recombine the individuals selected in the previous operation, trying to recombine them and form new individuals
   * @param selectedIndividuals: vector containing the selected individuals
   * @param newIndividuals: vector that will contain the new individuals generated (offsprings)
  */
  void recombine_individuals(std::vector<TSPEncode> &selectedIndividuals, std::vector<TSPEncode> &newIndividuals);

  /**
   * Method to apply mutation on the offspring, by swapping 2 nodes in the sequence (70% prob.)
   * or by reversing a subsequence in the sequence (30% prob.)
   * (% OF PROBABILITY CAN CHANGE)
   * @param offspring: the individual to apply the mutation
  */
  void apply_mutation(TSPEncode &offspring);

  /**
   * Implement the ordered crossover, used in the recombination to generate new sequences (2) at each time
   * @param offspring1: the first offspring generated
   * @param offspring2: the second offspring generated
   * @param selectedIndividual1: the first individual where the crossover takes the genes
   * @param selectedIndividual2: the second individual where the crossover takes the genes
   * @param numGenes: the total number of genes in the individuals
  */
  void ordered_crossover(TSPEncode &offspring1, TSPEncode &offspring2, TSPEncode &selectedIndividual1, TSPEncode &selectedIndividual2, const int &numGenes); //OK

  /**
   * Implement the PBX crossover, used in the recombination to generate new sequences (2) at each time
   * @param offspring1: the first offspring generated
   * @param offspring2: the second offspring generated
   * @param selectedIndividual1: the first individual where the crossover takes the genes
   * @param selectedIndividual2: the second individual where the crossover takes the genes
   * @param numGenes: the total number of genes in the individuals
  */
  void PBXCrossover(TSPEncode &offspring1, TSPEncode &offspring2, TSPEncode &selectedIndividual1, TSPEncode &selectedIndividual2, const int &numGenes);

  /**
   * Replace the current individual in the population, trying to select the best n (size of the population) individuals from
   * actual population + new individuals generated, also mantaining some heterogenous in the population using the Hamming Distance
   * @param newIndividuals: the new individuals in the population
  */
  void replace_population(std::vector<TSPEncode> &newIndividuals);

  /**
   * Evaluate the cost (fitness) of the tour describe by the sequence for an individual
   * @param sequence: the sequence to evaluate
   * @param costs: the cost-matrix for the tsp instance
   * @return the total cost of the tour
  */
  double evaluate_fitness(const std::vector<int> &sequence, const std::vector<std::vector<double>> costs);

  // Heuristics (and random) methods to generate population
  /**
   * Generate a set of sequence at random, trying to avoid repetitions
   * @param n: number of sequence to generate
   * @param values: the values for the nodes in the sequence
   * @return an unordered set of sequence
  */
  std::unordered_set<std::vector<int>, hashFunction> generate_random_populations(const int &n, const std::vector<int> &values );

  /**
   * Generate a set of sequence using best inserction heuristic method, starting from random nodes at the beginning
   * @param n: number of sequence to generate
   * @param values: the values for the nodes in the sequence
   * @return a new sequence
  */
  std::vector<int> best_insertion_random(const int &n, const std::vector<std::vector<double>> &costs);

  /**
   * Generate a set of sequence using farthest inserction heuristic method, starting from random nodes at the beginning
   * @param n: number of sequence to generate
   * @param values: the values for the nodes in the sequence
   * @return a new sequence
  */
  std::vector<int> farthest_insertion_random(const int &n, const std::vector<std::vector<double>> &costs);

  // Local Search helper functions
  /**
   * Implement a fast local search (limited number of iterations) using 2_opt strategy, to improve the solution in the recombination phase
   * @param offspring: the individual to educate (trying to find better solution in the sequence)
  */
  void local_search_2_opt_fast(TSPEncode &offspring);

  /**
   * Implement the Simulated Annealing Algorithm, to improve the final solution (if it is possible)
   * @param offspring: the individual to improve
  */
  void simulated_annealing(TSPEncode &offspring);

  //void local_search_2_opt_complete(TSPEncode &solution); // Maybe DELETE IT

  /**
   * Auxiliar function for the SA e LS (fast), calculate the fitness of the sequence if we apply 2 opt on position i and j, without applying the reverse
   * @param oldCost: starting cost
   * @param i: position i in the sequence to reverse
   * @param j: position j in the sequence to reverse
   * @param h: position i-1 to calculate the cost from h-i and h-j
   * @param l: position j+1 to calculate the cost from j-l and i-l
   * @param values: the cost-matrix for the tsp instance
   * @return the cost of the (eventually) new sequence
  */
  double evaluate_fitness_ls(const double &oldCost, int  &i , int &j, int &h, int &l, const std::vector<std::vector<double>> &values); //OK

  // Auxiliar methods
  /**
   * Calculate the hamming distance between 2 sequences, used to evaluate the avg hamming distance in the population
   * @param seq1: first sequence
   * @param seq2: second sequence
   * @return total distance of the 2 sequence (different genes)
  */
  int hamming_distance(const std::vector<int> &seq1, const std::vector<int> &seq2);
};

#endif /* TSPSOLVER_H */