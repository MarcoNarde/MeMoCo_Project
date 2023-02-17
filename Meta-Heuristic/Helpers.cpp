#include "Helpers.h"
void Helpers::display_help() {
    std::cout << "Multiple types of usage:" << std::endl;
    std::cout << "\n1) Usage: ./main [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help\t\tDisplay this help message" << std::endl;
    std::cout << "\n2) Usage: ./main [filename]" << std::endl;
    std::cout << "Allows you to enter the file to be tested, with the parameters chosen directly from the program" << std::endl;
    std::cout << "Example: ./main .\\dataset\\dataset_10_01.dat   -> Run the programm on the file dataset_10_01.dat (Check documentation for the file format)" << std::endl;
    std::cout << "\n3) Usage: ./main filename.dat [populationSize] [maxIteration] [maxIterationWithoutImprovement] [maxTimer] [selectedIndividualsNumber] [probUseOrdCrossover] [probMutation] [probFastOpt] [maxIterationFastLS] [avgHammingDistance] [useFinalOpt] [useLinearRanking] [tournamentDimension]" << std::endl;
    std::cout << "- populationSize: The size of the population to be taken into consideration (INTEGER)" << std::endl;
    std::cout << "- maxIteration: The maximum number of iterations that the algorithm must perform (INTEGER)" << std::endl;
    std::cout << "- maxIterationWithoutImprovement: The maximum number of iterations without an improvement on the solution that the algorithm must perform (INTEGER)" << std::endl;
    std::cout << "- maxTimer: The maximum time (in seconds) for executing the algorithm loop (does not take into account a possible final optimization!), if you don't want to take or the time put '/' (INTEGER)" << std::endl;
    std::cout << "- selectedIndividualsNumber: The number of individuals selected from the population at each iteration (INTEGER)" << std::endl;
    std::cout << "- probUseOrdCrossover: The probability of using ORDERED CROSSOVER in generating new individuals instead of PBX CROSSOVER (DECIMAL between 0-1) EX: if is 0.7 -> prob(ORDCROS.) = 0.7 and prob(PBXCROS.) = 0.3" << std::endl;
    std::cout << "- probMutation: The probability of having mutations within the generation of new individuals (DECIMAL between 0-1)" << std::endl;
    std::cout << "- probFastOpt: The probability of using a Simulated Annealing (fast) algorithm to improve the new individuals generated (DECIMAL between 0-1)" << std::endl;
    std::cout << "- maxIterationFastLS: The maximum number of Iterations for the Simulated Annealing algorithm (fast) in case its probability is > 0 (INTEGER)" << std::endl;
    std::cout << "- avgHammingDistance: The average of different genes among gene sequences within individuals in the population (INTEGER should be between 1 and the number of genes (nodes) for each sequence) EX (hamming distance): ([0 1 2 3 0] , [0 2 1 3 0]) has 2 different genes" << std::endl;
    std::cout << "- useFinalOpt: Defines whether to use a Local Search algorithm to improve the final solution found (0/1)" << std::endl;
    std::cout << "- useLinearRanking: Defines whether you want to use a selection based on Linear Ranking (in case 1) or based on a Tournament Selectio (then 0) (0/1)" << std::endl;
    std::cout << "- tournamentDimension: If you decide to use Tournament Selection (then useLinearRanking = 0) then you need to define a value for the size of the subsets for the selection, otherwise put '/' (INTEGER)" << std::endl;
  }

void Helpers::display_first_use(){
    std::cout << "HI! it looks like you want to use the program, here are some suggestions:" << std::endl;
    display_help();
}

void Helpers::print_parameters(int& numberNodes, const int& populationSize, const int& maxIteration,
            const int& maxNumberNotImprovingIteration, const int& maxTimer, const int& selectedIndividualsNumber, const double& probUseOrdCrossover, const double& probMutation,
            const double& probFastOpt, const int& maxIterationFastLS,const int& avgHammingDistance,const bool& useFinalOpt , const bool& useLinearRanking, const int& tournamentSize){
  std::cout<<"--- PARAMETERS ---"<<std::endl;
  std::cout<<"Number of nodes (TSP INSTANCE): " << numberNodes <<std::endl;
  std::cout<<"Population size: " << populationSize <<std::endl;
  std::cout<<"Maximun number of iterations: " << maxIteration <<std::endl;
  std::cout<<"Maximun number of iterations without any improvement: " << maxNumberNotImprovingIteration <<std::endl;
  maxTimer == INT_MAX ? std::cout<<"No max time of execution set"<<std::endl : std::cout<<"Max time of execution: "<<maxTimer<<std::endl;
  std::cout<<"Number of selected individual at each iteration: " << selectedIndividualsNumber <<std::endl;
  std::cout<<"Probability of use ORDCrossover: " << probUseOrdCrossover << ", Probability of usign PBXCrossover (by conseguence): " << (1-probUseOrdCrossover) <<std::endl;
  std::cout<<"Probability of mutations (in the offsprings): " << probMutation <<std::endl;
  std::cout<<"Probability of usign fast LS to optimize (in the offsprings): " << probFastOpt <<std::endl;
  if(probFastOpt > 0.0){
      std::cout<<"Maximun Iteration in fast Local Seach: " << maxIterationFastLS <<std::endl;
  }
  std::cout<<"Threshold for the AVG hamming distance: " << avgHammingDistance <<std::endl;
  std::string useOpt = useFinalOpt ? "YES" : "NO";
  std::cout<<"Use final ls optimization (YES/NO): " << useOpt <<std::endl;
  if(useLinearRanking){
      std::cout<<"Using Linear Ranking for selection"<<std::endl;
  }else{
      std::cout<<"Using Tournament Selection for selection with size of the tournament: " << tournamentSize <<std::endl;
  }
}

std::vector<int> Helpers::create_initial_vector(int n) {
    std::vector<int> v;
    for (int i = 0; i < n; ++i) {
        v.push_back(i);
    }
    return v;
}

/**
 * Retrieve the default value for the parameters (Based on the number of nodes)
 * @param numberNodes: number of nodes for the current tsp instance
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
void Helpers::define_parameters(const int& numberNodes, int& populationSize, int& maxIteration,
          int& maxNumberNotImprovingIteration, int& maxTimer, int& selectedIndividualsNumber,  double& probUseOrdCrossover,  double& probMutation,
          double& probFastOpt, int& maxIterationFastLS, int& avgHammingDistance, bool& useFinalOpt ,  bool& useLinearRanking,  int& tournamentSize){
            
// Open stream for file "default_parameters.txt"
  std::ifstream file("default_parameters.txt");

  // Read the header line.
  std::string header;
  std::getline(file, header);

  // Read the parameters for each node count.
  std::map<int, Parameters> parameters;
  int nodes;
  std::string line;
  while (std::getline(file, line)) {
      Parameters p;
      std::stringstream ss(line);
      ss >> nodes;
      ss >> p.PS >> p.MI >> p.MIWI >> p.MT >> p.SIN >> p.PUOC >> p.PM >> p.PFLS >> p.MIFLS >> p.AVGHD >> p.UFO >> p.ULR >> p.TS;
      parameters[nodes] = p;
  }

  // Find the parameters for the closest node count.
  int closest_node_count = 0;
  for (const auto& p : parameters) {
      int count = p.first;
      if (count >= numberNodes && (closest_node_count == 0 || count < closest_node_count)) {
          closest_node_count = count;
      }
  }

  // Initialize the paramters
  populationSize = parameters[closest_node_count].PS;
  maxIteration = parameters[closest_node_count].MI;
  maxNumberNotImprovingIteration = parameters[closest_node_count].MIWI;
  try {
      maxTimer = stoi(parameters[closest_node_count].MT);
  } catch (const std::invalid_argument&) {
      maxTimer = INT_MAX;
  }
  selectedIndividualsNumber = parameters[closest_node_count].SIN;
  probUseOrdCrossover = parameters[closest_node_count].PUOC;
  probMutation = parameters[closest_node_count].PM;
  probFastOpt = parameters[closest_node_count].PFLS;
  maxIterationFastLS = parameters[closest_node_count].MIFLS;
  avgHammingDistance = parameters[closest_node_count].AVGHD;
  useFinalOpt = parameters[closest_node_count].UFO;
  useLinearRanking = parameters[closest_node_count].ULR;
  if(!useLinearRanking){
      try{
          tournamentSize = stoi(parameters[closest_node_count].TS);
      }catch(const std::invalid_argument&){
          tournamentSize = 0;
      }
  }
}