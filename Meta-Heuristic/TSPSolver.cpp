/**
 * @file TSPSolver.cpp
 * @brief TSP solver (Genetic Algorithm)
 */

#include "TSPSolver.h"

double TSPSolver::solve()
{
  try
  {
    int iter = 0;
    int iterNotImpr = 0;

    auto start = std::chrono::steady_clock::now(); // Start the calculation of time for stopping algorithm
    auto stop_time = std::chrono::seconds(maxTime); // MaxTime Converted in chrono::seconds

    // Create initial population
    population.reserve(populationSize);

    // Generate the initial population
    generate_population();

    // Sort algorithm to find actual best solution at first position
    std::sort(population.begin(), population.end(), [](const TSPEncode &a, const TSPEncode &b){
    return a.cost < b.cost;
    });

    // Initialize the actual best solution
    TSPEncode actualBestSolution(population[0]);

    std::cout<<"--- START SOLVING PROBLEM ---\n";
    std::cout<<"First best solution: " << actualBestSolution;
    
    while(iter < maxIteration && iterNotImpr < maxNumberNotImprovingIteration && (std::chrono::steady_clock::now() - start < stop_time)){
      iter++;
      
      // Select groups
      std::vector<TSPEncode> selectedIndividuals;
      if(selectedIndividualsNumber != population.size()){
        select_solutions(selectedIndividuals);
      }

      // Recombine individuals
      std::vector<TSPEncode> newIndividuals;
      std::shuffle(selectedIndividuals.begin(), selectedIndividuals.end(), gen);
      recombine_individuals(selectedIndividuals, newIndividuals);

      // Replace population
      replace_population(newIndividuals);

      std::sort(population.begin(), population.end(), [](const TSPEncode &a, const TSPEncode &b){
      return a.cost < b.cost;
      });

      // Update the best solution
      if(actualBestSolution.cost - 1e-6 > population[0].cost){
        actualBestSolution = population[0];
        iterNotImpr = 0;
        
        std::cout<<"ITERATION: "<<iter<<" - New best solution: " << actualBestSolution;
      }else{
        iterNotImpr++;
      }

      // Check stopping criterios
      if(iter >= maxIteration){
        std::cout <<"Maximum iterations reached!\n";
      }else if(iterNotImpr >= maxNumberNotImprovingIteration){
        std::cout <<"Consecutive iterations with no improvements achieved!\n";
      }else if(std::chrono::steady_clock::now() - start > stop_time){
        std::cout <<"Max time of execution achieved!\n";
      }
    }

    // Apply SA as post-optimization
    if(useFinalOpt){
      std::cout<<"Applying SA for optimization...\n";
      simulated_annealing(actualBestSolution);
    }

    std::cout<<"BEST SOLUTION: " <<actualBestSolution;
    return actualBestSolution.cost;
  }
  catch(std::exception& e)
  {
    std::cout << ">>>EXCEPTION: " << e.what() << std::endl;
    return -1;
  }
}

void TSPSolver::generate_population(){
  int randPopSize = populationSize;
  int firstHeurPopSize = 0;
  int secondtHeurPopSize = 0;

  // Different population algorithm in nodes >= 10
  if(tsp.numberNodes >= 10){
    randPopSize = populationSize - ceil(populationSize / 3.0);
    firstHeurPopSize = ceil((populationSize - randPopSize) / 2.0);
    secondtHeurPopSize = populationSize - randPopSize - firstHeurPopSize;
  } 

  // Random population part
  std::unordered_set<std::vector<int>, hashFunction> sequences = generate_random_populations(randPopSize, initialSequence);
  // Best insertion population part
  for (size_t i = 0; i < firstHeurPopSize; i++){
    std::vector<int> seq = best_insertion_random(tsp.numberNodes, tsp.values);
    if (sequences.count(seq) == 0) {
      // Add the shuffled sequence to the list of sequences
      sequences.insert(seq);
    }else{
      // Decrement the count to generate another sequence
      i--;
    }
  }
  // Farthest insertion population part
  for (size_t i = 0; i < secondtHeurPopSize; i++){
    std::vector<int> seq = farthest_insertion_random(tsp.numberNodes, tsp.values);
    if (sequences.count(seq) == 0) {
      // Add the shuffled sequence to the list of sequences
      sequences.insert(seq);
    }else{
      // Decrement the count to generate another sequence
      i--;
    }
  }
  
  // Add the sequences to the population
  for (const auto& elem: sequences) {
      double cost = evaluate_fitness(elem, tsp.values);
      TSPEncode obj(elem, cost);
      population.push_back(obj);
  }

}

void TSPSolver::select_solutions(std::vector<TSPEncode> &selectedIndividuals){
  // Here population is already sorted by cost increasing
  std::vector<TSPEncode> partecipant = population;

  if(useLinearRanking){
    // LINEAR RANKING SELECTION
    linear_ranking_selection(partecipant, selectedIndividuals, selectedIndividualsNumber);
  }else{
    // TOURNAMENT SELECTION
    // Makes more random the tournament tuples
    std::shuffle(partecipant.begin(), partecipant.end(), gen);
    tournament_selection(partecipant, selectedIndividuals, selectedIndividualsNumber, tournamentSize);
  }
}

void TSPSolver::linear_ranking_selection(std::vector<TSPEncode> &partecipant, std::vector<TSPEncode> &selectedIndividuals, const int& selectedIndividualsNumber){
  int selectedSize = 0;
  while (selectedSize < selectedIndividualsNumber)
  {
    std::vector<int> posToRemove; // Keeps track of selected individuals' positions
    std::vector<double> probability; // Stores probability for each individual
    int n = partecipant.size();

    //Define the probabilities for the actual partecipants
    for (size_t i = 0; i < partecipant.size(); i++)
    {
      probability.push_back((n-i) / (n*(n-1)/2.0));
    }

    // Selects individual based on probability
    for (size_t i = 0; i < partecipant.size(); i++)
    {
      double r = dist(gen);
      if(r < probability[i]){
        selectedIndividuals.push_back(partecipant[i]);
        selectedSize++;
        posToRemove.push_back(i);
      }
    }

    // Remove selected individual from partecipants
    for(int i = posToRemove.size() -1 ; i >= 0 ; i--)
      partecipant.erase(partecipant.begin() + posToRemove[i]);
  }
}

void TSPSolver::tournament_selection( std::vector<TSPEncode> &partecipant, 
                                      std::vector<TSPEncode> &selectedIndividuals, 
                                      const int& selectedIndividualsNumber, 
                                      int tournamentSize)
{
  // Keep track of the number of selected individuals
  int selectedSize = 0;

  // Repeat the process until the desired number of selected individuals is reached
  while (selectedSize < selectedIndividualsNumber)
  {
    // Select the first individual randomly
    std::uniform_int_distribution<int> distribution(0, partecipant.size()-1);
    int bestIndex = distribution(gen);

    // Participate in a tournament with `tournamentSize` participants
    for (int i = 1; i < tournamentSize; i++)
    {
      // Select another individual randomly
      int candidateIndex = distribution(gen);

      // Compare the fitness of the candidate and the current best individual
      if (partecipant[candidateIndex].cost < partecipant[bestIndex].cost)
      {
        // If the candidate has better fitness, replace the current best
        bestIndex = candidateIndex;
      }
    }
    // Add the winner of the tournament to the selected individuals
    selectedIndividuals.push_back(partecipant[bestIndex]);
    // Remove the selected individual from next tournament
    partecipant.erase(partecipant.begin() + bestIndex);
    // Increment the count of selected individuals
    selectedSize++;
  }
}

void TSPSolver::recombine_individuals(std::vector<TSPEncode> &selectedIndividuals, std::vector<TSPEncode> &newIndividuals){
  int numIndividuals = selectedIndividuals.size();
  int numGenes = selectedIndividuals[0].sequence.size()-1; //try to avoid last node that is equal with first

  for (int i = 0; i < numIndividuals-1; i ++) {

    // New offsprings
    TSPEncode offspring1(numGenes+1), offspring2(numGenes+1);

    // Calculate random number
    double rcross = dist(gen);

    // Apply crossover with probability "probUseOrdCrossover"
    if(rcross < probUseOrdCrossover){
      ordered_crossover(offspring1, offspring2, selectedIndividuals[i], selectedIndividuals[i+1], numGenes);
    }else{
      PBXCrossover(offspring1, offspring2, selectedIndividuals[i], selectedIndividuals[i+1], numGenes);
    }

    // Apply mutation with probability p
    double r = dist(gen);
    if(r < probMutation){
      apply_mutation(offspring1);
    }
    r = dist(gen);
    if(r < probMutation){
      apply_mutation(offspring2);
    }

    // TSP: Add first nodes to make the cycle
    offspring1.sequence.push_back(offspring1.sequence[0]);
    offspring2.sequence.push_back(offspring2.sequence[0]);

    // Update the costs
    offspring1.cost = evaluate_fitness(offspring1.sequence, tsp.values);
    offspring2.cost = evaluate_fitness(offspring2.sequence, tsp.values);

    double rl = dist(gen);
    if(rl < probFastOpt){
      local_search_2_opt_fast(offspring1);
    }
    rl = dist(gen);
    if(rl < probFastOpt){
      local_search_2_opt_fast(offspring2);
    }

    // Add the offspring to new individuals
    newIndividuals.push_back(offspring1);
    newIndividuals.push_back(offspring2);
  }

  // Implement Intensification
  std::map<std::vector<int>, int> count;
  bool repetitions = false; // Try to find sequence that are repeated 3 or more times
  for (const TSPEncode& seq : newIndividuals) {
    count[seq.sequence]++;
    if(count[seq.sequence] > (newIndividuals.size() / 3) + 2){ // Implementation choice (don't want to put more parameters)
      repetitions = true;
      break;
    }
  }

  // If there some repetition in the new generated offsprings
  if(repetitions){

    //Generate more offsprings
    int newGeneratedOffsprings = numIndividuals / 2;
    int newGenerated = 0;
    std::uniform_int_distribution<int> distribution(0, selectedIndividuals.size()-1);

    while (newGenerated < newGeneratedOffsprings)
    {
      int i = distribution(gen);
      int j = distribution(gen);
      
      // New offsprings
      TSPEncode offspring1(numGenes+1), offspring2(numGenes+1);
      PBXCrossover(offspring1, offspring2, selectedIndividuals[i], selectedIndividuals[j], numGenes);
 
      // Apply mutation with probability p
      double r = dist(gen);
      if(r < probMutation){
        apply_mutation(offspring1);
      }
      r = dist(gen);
      if(r < probMutation){
        apply_mutation(offspring2);
      }

      // TSP: Add first nodes to make the cycle
      offspring1.sequence.push_back(offspring1.sequence[0]);
      offspring2.sequence.push_back(offspring2.sequence[0]);

      // Update the costs
      offspring1.cost = evaluate_fitness(offspring1.sequence, tsp.values);
      offspring2.cost = evaluate_fitness(offspring2.sequence, tsp.values);

      newIndividuals.push_back(offspring1);
      newIndividuals.push_back(offspring2);

      newGenerated += 2;
    }
  }
}

void TSPSolver::PBXCrossover(TSPEncode &offspring1, TSPEncode &offspring2, TSPEncode &selectedIndividual1, TSPEncode &selectedIndividual2, const int &numGenes){
  // Calculate the number of positions to be selected
  int numberPos = numGenes / 2;
  // Store the selected positions
  std::vector<int> position(numGenes);
  // Store if a position is selected or not
  std::vector<bool> selected(numGenes, false);

  // Initialize position vector
  for (int i = 0; i < numGenes; i++) {
    position[i] = i;
  }

  // Select a random set of positions
  std::shuffle(position.begin(), position.end(), gen);
  for (int i = 0; i < numberPos; i++) {
    selected[position[i]] = true;
  }

  // Need to "initialize" the offsprings
  for (size_t i = 0; i < numGenes; i++)
  {
    offspring1.sequence.push_back(-1);
    offspring2.sequence.push_back(-1);
  }
  

  // Fill offspring using the selected positions
  for (int i = 0; i < numGenes; i++) {
    if (selected[i]) {
      offspring1.sequence[i] = selectedIndividual1.sequence[i];
      offspring2.sequence[i] = selectedIndividual2.sequence[i];
    }
  }

  // Fill the rest of offspring1 && offspring2
  for (int i = 0; i < numGenes; i++) {
    if (!selected[i]) {
      int j = 0;
      // Create a set to store the values of offspring1
      std::unordered_set<int> offspring1_set(offspring1.sequence.begin(), offspring1.sequence.end());
      // Find the first un-selected value of selectedIndividual2
      while (offspring1_set.count(selectedIndividual2.sequence[j]) != 0) {
        j++;
      }
      offspring1.sequence[i] = selectedIndividual2.sequence[j];

      j = 0;
      // Create a set to store the values of offspring2
      std::unordered_set<int> offspring2_set(offspring2.sequence.begin(), offspring2.sequence.end());
      // Find the first un-selected value of selectedIndividual1
      while (offspring2_set.count(selectedIndividual1.sequence[j]) != 0) {
        j++;
      }
      offspring2.sequence[i] = selectedIndividual1.sequence[j];
    }
  }
}

void TSPSolver::ordered_crossover(TSPEncode &offspring1, TSPEncode &offspring2, TSPEncode &selectedIndividual1, TSPEncode &selectedIndividual2, const int &numGenes){
  // Generate 2 random positions
  std::uniform_int_distribution<int> distribution(2, numGenes - 3);
  int start = distribution(gen);
  int end = distribution(gen);    

  // Ensure start is smaller than end
  if (start > end) {
    int temp = start;
    start = end;
    end = temp;
  }

  // Will contains genes already putted
  std::vector<int> secondGeneEnd(selectedIndividual2.sequence);
  std::vector<int> firstGeneEnd(selectedIndividual1.sequence);

  // Remove unused genes
  secondGeneEnd.erase(secondGeneEnd.begin() + start, secondGeneEnd.begin() + end);
  firstGeneEnd.erase(firstGeneEnd.begin() + start, firstGeneEnd.begin() + end);

  // Create 2 sets to improve the find efficiency
  std::unordered_set<int> secondGeneEndSet(secondGeneEnd.begin(), secondGeneEnd.end());
  std::unordered_set<int> firstGeneEndSet(firstGeneEnd.begin(), firstGeneEnd.end());

  // Offspring : from 0 to start -> push genes of the first/second individual
  for (int j = 0; j < start; j++) {
      offspring1.sequence.push_back(selectedIndividual1.sequence[j]);
      offspring2.sequence.push_back(selectedIndividual2.sequence[j]);
  }

  // Offspring : from start to end -> push genes of the second/first individual not already in the sequence
  for (int j = 0; j < numGenes; j++) {
    int gen2 = selectedIndividual2.sequence[j];
    // Check if gene is already in the offspring
    if (firstGeneEndSet.count(gen2) == 0) {
      offspring1.sequence.push_back(gen2);
    }

    int gen1 = selectedIndividual1.sequence[j];
    // Check if gene is already in the offspring
    if (secondGeneEndSet.count(gen1) == 0) {
      offspring2.sequence.push_back(gen1);
    }
  }

  // Offspring : from end to last -> push genes of the first/second individual
  for (int j = end; j < numGenes; j++) {
      offspring1.sequence.push_back(selectedIndividual1.sequence[j]);
      offspring2.sequence.push_back(selectedIndividual2.sequence[j]);
  }
}

void TSPSolver::apply_mutation(TSPEncode &offspring){
  // Generate 2 random positions
  std::uniform_int_distribution<int> distribution(0, offspring.sequence.size() - 1);
  int i = distribution(gen);
  int j = distribution(gen);

  // Use for diversification
  double r = dist(gen);
  if(r < 0.66){ // This value can change but need more tests
    // Swap 2 positions
    std::swap(offspring.sequence[i],offspring.sequence[j]);
  }else{
    if(j < i){
      int temp = j;
      j = i;
      i = j;
    }
    // Reverse the subsequence
    std::reverse(offspring.sequence.begin() + i, offspring.sequence.begin() + j + 1);
  }
}

void TSPSolver::replace_population(std::vector<TSPEncode> &newIndividuals) {
  int n = population.size();

  // Initialize total population (actual population + new individuals)
  std::vector<TSPEncode> totPopulation = population;
  for (size_t i = 0; i < newIndividuals.size(); i++) {
    totPopulation.push_back(newIndividuals[i]);
  }

  // Sort the array based on fitness (Cost)
  std::sort(totPopulation.begin(), totPopulation.end(), [](const TSPEncode &a, const TSPEncode &b){
    return a.cost < b.cost;
  });

  std::vector<TSPEncode> newPopulation;
  newPopulation.reserve(n);
  // Take the first n best individuals
  for (size_t i = 0; i < n; i++)
  {
    newPopulation.push_back(totPopulation[i]);
  }
  

  // Calculate the average Hamming distance between individuals in the new population
  int avgHammingDist = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      avgHammingDist += hamming_distance(newPopulation[i].sequence, newPopulation[j].sequence);
    }
  }
  avgHammingDist /= (n * (n - 1) / 2);

  // If the average Hamming distance is lower than MIN_THRESHOLD
  if (avgHammingDist < hammingDistanceThreshold) {
    // Try to augment the average Hamming distance - avoid removing first best option
    std::uniform_int_distribution<int> distribution(1, n - 1);
    for (int i = n; i < totPopulation.size(); i++) {
      int random_index = distribution(gen);
      if (hamming_distance(totPopulation[i].sequence, newPopulation[random_index].sequence) > avgHammingDist) {
        newPopulation[random_index] = totPopulation[i];
      }
    }
  }

  population = newPopulation;
}

double TSPSolver::evaluate_fitness(const std::vector<int> &sequence, const std::vector<std::vector<double>> costs){
  double cost = 0.0;
  // Compute cost of the tour
  for (size_t i = 0; i < sequence.size() - 1; i++)
  {
    cost += costs[sequence[i]][sequence[i+1]];
  }

  return cost;
}

// Heuristics (and random) methods to generate population
std::unordered_set<std::vector<int>, hashFunction> TSPSolver::generate_random_populations(const int &n, const std::vector<int> &values ){
  std::unordered_set<std::vector<int>, hashFunction> sequences;

  for (int i = 0; i < n; i++) {
    // Create a copy of the input vector
    std::vector<int> temp = values;

    // Shuffle the elements in the copy using the random number generator
    shuffle(temp.begin(), temp.end(), gen);

    // check if the sequence already exists in the list of sequences
    if (sequences.count(temp) == 0) {
      temp.push_back(temp[0]);
      // Add the shuffled sequence to the list of sequences
      sequences.insert(temp);
    }else{
      // Decrement the count to generate another sequence
      i--;
    }
  }

  return sequences;
}

std::vector<int> TSPSolver::best_insertion_random(const int &n, const std::vector<std::vector<double>> &costs ){
  // Initial tour empty
  std::vector<int> tour;

  // Initial cost
  double cost = 0;

  std::vector<int> unvisited;
  for(int k = 0; k < n; k++){
    unvisited.push_back(k);
  }

  // Select random starting nodes
  std::shuffle(unvisited.begin(), unvisited.end(), gen);
  int i = unvisited[0];
  int j = unvisited[1];

  // Add the 2 nodes to the tour
  tour.push_back(i);  tour.push_back(j);  tour.push_back(i);
  unvisited.erase(remove(unvisited.begin(), unvisited.end(), i), unvisited.end());
  unvisited.erase(remove(unvisited.begin(), unvisited.end(), j), unvisited.end());

  // Update cost
  cost = costs[i][j] + costs[j][i];

  // Loop between remaining nodes
  while (!unvisited.empty())
  {
    // Initialize variables for new node
    int pos = -1;
    double min_cost = std::numeric_limits<double>::max();
    int node = -1;

    // Iterate over all remaining unvisited nodes
    for(auto k:unvisited){
      // Iterate over all positions in the tour
      for (int l = 1; l < tour.size(); l++)
      {
        // Calculate the new total cost of the tour if node k is inserted at position l
        double new_cost = cost + costs[tour[l-1]][k] + costs[k][tour[l]] - costs[tour[l-1]][tour[l]];
        // Compare the new cost with the current minimum cost
        if (new_cost < min_cost)
        {
          // Update the minimum cost, the best position and the new node
          min_cost = new_cost;
          pos = l;
          node = k;
        }
      }
    }
    // Update the total cost of the tour
    cost = min_cost;
    // Insert the new node in the best position
    tour.insert(tour.begin() + pos, node);
    // Remove the new node from the unvisited set
    unvisited.erase(remove(unvisited.begin(), unvisited.end(), node), unvisited.end());
  }
  return tour;
}

std::vector<int> TSPSolver::farthest_insertion_random(const int &n, const std::vector<std::vector<double>> &costs){
  // Initial tour with the two farthest/nearest nodes
  std::vector<int> tour;

  // Initial cost
  double cost = 0;

  std::vector<int> unvisited;
  for(int k = 0; k < n; k++){
    unvisited.push_back(k);
  }

  // Select random starting nodes
  std::shuffle(unvisited.begin(), unvisited.end(), gen);
  int i = unvisited[0];
  int j = unvisited[1];

  // Add the 2 nodes to the tour
  tour.push_back(i);  tour.push_back(j);  tour.push_back(i);
  unvisited.erase(remove(unvisited.begin(), unvisited.end(), i), unvisited.end());
  unvisited.erase(remove(unvisited.begin(), unvisited.end(), j), unvisited.end());

  // Update cost
  cost = costs[i][j] + costs[j][i];

  // Loop between remaining nodes
 while (!unvisited.empty())
  {
    // Initialize variables for new node
    int pos = -1;
    double max_cost = std::numeric_limits<double>::min();
    int node = -1;

    // Iterate over all remaining unvisited nodes
    for(auto k:unvisited){
      // Find the cost of the farthest node from the current tour
      double cost = 0;
      for (int j = 0; j < tour.size(); j++)
        cost = std::max(cost, costs[k][tour[j]]);

      // Compare the cost of the current node with the current maximum cost
      if (cost > max_cost)
      {
        // Update the maximum cost and the new node
        max_cost = cost;
        node = k;
      }
    }

    // Iterate over all positions in the tour
    double min_cost = std::numeric_limits<double>::max();
    for (int l = 1; l < tour.size(); l++)
    {
      // Calculate the new total cost of the tour if node k is inserted at position l
      double new_cost = cost + costs[tour[l-1]][node] + costs[node][tour[l]] - costs[tour[l-1]][tour[l]];
      // Compare the new cost with the current minimum cost
      if (new_cost < min_cost)
      {
        // Update the minimum cost and the best position
        min_cost = new_cost;
        pos = l;
      }
    }

    // Update the total cost of the tour
    cost = min_cost;
    // Insert the new node in the best position
    tour.insert(tour.begin() + pos, node);
    // Remove the new node from the unvisited set
    unvisited.erase(remove(unvisited.begin(), unvisited.end(), node), unvisited.end());
  }
  return tour;
}

// Local Search helper functions
void TSPSolver::local_search_2_opt_fast(TSPEncode &offspring){
  int numGenes = offspring.sequence.size(); // Number of genes in the offspring's sequence
  int iter = 0;
  double bestCost = offspring.cost; // Best cost so far
  
  // Generate a random number distribution for choosing the start and end points
  std::uniform_int_distribution<int> distribution(1, numGenes - 2);
  do {
    // Choose a random start and end point
    int start = distribution(gen);
    int end = distribution(gen);

    // Ensure that start is smaller than end  
    if(start > end){
      int temp = start;
      start = end;
      end = temp;
    }

    // Get the indices of the genes to swap
    int h = offspring.sequence[start-1];
    int i = offspring.sequence[start];
    int j = offspring.sequence[end];
    int l = offspring.sequence[end+1];

    // Calculate the new cost after swapping the genes
    double newCost = evaluate_fitness_ls(bestCost, i , j, h, l, tsp.values);

    // If the new cost is better than the best cost so far, reverse the sub-sequence between start and end and update the best cost
    if (newCost < bestCost){
      std::reverse(offspring.sequence.begin() + start, offspring.sequence.begin() + end + 1);
      bestCost = newCost;
      offspring.cost = bestCost;
    }

    iter++;
  } while (iter < maxIterationFastLS);
}

void TSPSolver::simulated_annealing(TSPEncode &offspring){
  int numGenes = offspring.sequence.size();
  double T = (numGenes-1) * 2; // Initial temperature
  double deltaE; // Difference in cost between current solution and new solution
  double acceptanceProb; // Probability of accepting new solution not improving
  double bestCost = offspring.cost; // Store the best cost found so far
  double currentCost = bestCost; // Current cost of the solution
  double coolingRate = 0.99 - (0.99 - 0.95) * log10(numGenes-1) / log10(1000); // Cooling rate for temperature


  TSPEncode best = offspring; // Store the best solution found so far

  // Random number generator for choosing starting and ending points of 2-opt swap
  // Trying to avoid choose first and last nodes
  std::uniform_int_distribution<int> distribution(1, numGenes - 2);
  while (T >= 0.1) {
    // Choose 2 random nodes positions
    int start = distribution(gen);
    int end = distribution(gen);

    // Skip the iteration if start and end points are the same
    if(start == end) continue;

    // Swap the start and end points if start is greater than end
    if(start > end){
      int temp = start;
      start = end;
      end = temp;
    }

    // Get the points h, i, j, and l for 2-opt swap
    int h = offspring.sequence[start-1];
    int i = offspring.sequence[start];
    int j = offspring.sequence[end];
    int l = offspring.sequence[end+1];

    // Evaluate new cost
    double newCost = evaluate_fitness_ls(currentCost, i , j, h, l, tsp.values);

    // Calculate the delta between the 2 costs
    deltaE = newCost - bestCost;
    if (deltaE < -1e-6) {
      // If new cost is better than best cost, update best cost and best solution
      std::reverse(offspring.sequence.begin() + start, offspring.sequence.begin() + end + 1);
      currentCost = newCost;
      offspring.cost = newCost;
      bestCost = newCost;
      best = offspring;
    } else {
      // Calculate acceptance probability
      acceptanceProb = exp(-deltaE / T);
      double random = dist(gen);
      // Accept new solution with higher cost with probability "acceptanceProb"
      if (random < acceptanceProb) {
        std::reverse(offspring.sequence.begin() + start, offspring.sequence.begin() + end + 1);
        currentCost = newCost;
        offspring.cost = currentCost;
      }
    }
    // Decrease temperature
    T *= coolingRate;
  }
  // Set the offspring to the best solution found
  offspring = best;
}

double TSPSolver::evaluate_fitness_ls(const double &actual, int  &i , int &j, int &h, int &l, const std::vector<std::vector<double>> &values){
  // Use formula: Cnew = Cold − chi − cjl + chj + cil
  double newCost = actual - values[h][i] - values[j][l] + values[h][j] + values[i][l];
  return newCost;
}

// Auxiliar methods
int TSPSolver::hamming_distance(const std::vector<int> &seq1, const std::vector<int> &seq2) {
  int distance = 0;
  // Loop through each position in the two sequences
  for (int i = 0; i < seq1.size(); i++) {
      // If elements at current (same) position are not equal -> increase distance
      if (seq1[i] != seq2[i]) {
          distance++;
      }
  }
  return distance;
}