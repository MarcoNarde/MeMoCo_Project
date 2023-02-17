/**
 * @file Helpers.h
 * @brief file containing helper functions
 *
 */

#ifndef HELPERS_H
#define HELPERS_H

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <climits>

/**
 * Struct used to retrieve the default parameter and using them in the program
*/
struct Parameters {
  int PS;
  int MI;
  int MIWI;
  std::string MT;
  int SIN;
  double PUOC;
  double PM;
  double PFLS;
  int MIFLS;
  int AVGHD;
  bool UFO;
  bool ULR;
  std::string TS;
};

class Helpers
{
public:

  /**
   * Display the message for the first use of the user, if it calls only ./main
  */
  static void display_first_use();

  /**
   * Display the help message for the user, help to use the program and explain the parameter to put
  */
  static void display_help();

  /**
   * Display the paramaters used by the algorithm in the current execution
  */
  static void print_parameters(int& numberNodes, const int& populationSize, const int& maxIteration,
              const int& maxNumberNotImprovingIteration, const int& maxTimer, const int& selectedIndividualsNumber, const double& probUseOrdCrossover, const double& probMutation,
              const double& probFastLocalSearch, const int& maxIterationFastLS,const int& avgHammingDistance,const bool& useFinalOpt , const bool& useLinearRanking, const int& tournamentSize);
  
  /**
   * Define the paramaters of execution, by looking at the default parameter in the file default_parameters.txt
  */
  static void define_parameters(const int& numberNodes, int& populationSize, int& maxIteration,
              int& maxNumberNotImprovingIteration, int& maxTimer, int& selectedIndividualsNumber,  double& probUseOrdCrossover,  double& probMutation,
              double& probFastLocalSearch, int& maxIterationFastLS, int& avgHammingDistance, bool& useFinalOpt ,  bool& useLinearRanking,  int& tournamentSize);
  
  /**
   * Create an initial sequence from 1 to n
  */
  static std::vector<int> create_initial_vector(int n);
};

#endif /* HELPERS_H */
