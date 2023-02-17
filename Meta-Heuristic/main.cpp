#include <iomanip>
#include <cstring>
#include "TSPSolver.h"
#include "Helpers.h"

//Test Genetic Algoritm for TSP
int main(int argc, char *argv[])
{
    int populationSize;
    int maxIteration;
    int maxIterationWithoutImprovement;
    int maxTimer;
    int selectedIndividualsNumber;
    double probMutation;
    double probFastOpt;
    double probUseOrdCrossover; // Probability of using Ordered Crossover instead of PBX Crossover (if 1 then 100% of use ORDCrossover and 0% of use PBXCrossover)
    int maxIterationFastLS;
    int avgHammingDistance;
    bool finalOpt = false;
    bool useLinearRanking = true;
    int tournamentSize = 3;
    vector<int> initialSequence;

    bool fileread = false;

    try{
        TSP tspInstance;
        // Check correct parameters
        if(argc < 2){// < 2 then show the help and first message
            Helpers::display_first_use();
            return 0;
        }else if (argc == 2){
            if(strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0){
                // Show help message
                Helpers::display_help();
                return 0;
            }else{
                // Try to retrieve default parameters
                fileread = tspInstance.read(argv[1]);
                if(!fileread) throw runtime_error("Can't set default parameters for the file");
                else{
                    std::cout << ">>> Setting Default Parameters <<<" << std::endl;
                    Helpers::define_parameters(tspInstance.numberNodes, populationSize, maxIteration, maxIterationWithoutImprovement,
                    maxTimer, selectedIndividualsNumber, probUseOrdCrossover, probMutation, probFastOpt, maxIterationFastLS,
                    avgHammingDistance, finalOpt, useLinearRanking, tournamentSize);
                    initialSequence = Helpers::create_initial_vector(tspInstance.numberNodes);
                }
            }
        // Else if more than 2 but less than 15 then you have to put all the parameters
        }else if (argc < 15){
            throw std::runtime_error("Missing " + std::to_string(15-argc) + " parameters!\nUsage: ./main filename.dat [populationSize] [maxIteration] [maxIterationWithoutImprovement] [maxTimer (s)] [selectedIndividualsNumber] [probUseOrdCrossover] [probMutation] [probFastOpt] [maxIterationFastLS] [avgHammingDistance] [useFinalOpt(0/1)] [useLinearRanking(0/1)] [tournamentDimension('/' if previous equal 0)]\n");
        }
        // Create the instance (reading data)
        if(!fileread){
            fileread = tspInstance.read(argv[1]);
            if(!fileread) throw runtime_error("Can't start the algorithm");
            initialSequence = Helpers::create_initial_vector(tspInstance.numberNodes);

            // Set initial population size
            populationSize = stoi(argv[2]); // Should be < !numberOfNodes
            // Set the maximun nuber of iteration
            maxIteration = stoi(argv[3]);
            // Set the max number of iteration (consecutive) without improvement in the solution
            maxIterationWithoutImprovement = stoi(argv[4]); // Should be <= maxIteration

            if(maxIterationWithoutImprovement > maxIteration){
                maxIterationWithoutImprovement = maxIteration;
            }

            // Set the max time to execute the algorithm
            try {
                maxTimer =  stoi(argv[5]);
            } catch (const std::invalid_argument&) {
                maxTimer = INT_MAX;
            }

            // Set the number of individuals to select at each time from the population
            selectedIndividualsNumber = stoi(argv[6]); // should be between 1 and populationSize

            if(selectedIndividualsNumber <= 0){
                selectedIndividualsNumber = 1;
            }

            if(selectedIndividualsNumber > populationSize){
                selectedIndividualsNumber = populationSize;
            }

            // Set the probability to apply fast local search to the offsprings
            probUseOrdCrossover = stod(argv[7]); // Probability should be between 0 and 1

            if(probUseOrdCrossover < 0.0 || probUseOrdCrossover > 1.0){
                probUseOrdCrossover = 0.5;
            }

            // Set the probability to get a mutation in the offspring
            probMutation = stod(argv[8]); // Probability should be between 0 and 1

            if(probMutation < 0.0 || probMutation > 1.0){
                probMutation = 0.5;
            }

            // Set the probability to apply fast local search to the offsprings
            probFastOpt = stod(argv[9]); // Probability should be between 0 and 1

            if(probFastOpt < 0.0){
                probFastOpt = 0.0;
            }else if (probFastOpt > 1.0){
                probFastOpt = 1;
            }

            if(probFastOpt >= 0.0 && probFastOpt <= 1.0){
                try{
                    maxIterationFastLS = stoi(argv[10]);
                }catch(const std::invalid_argument&){
                    throw::runtime_error("Probability of fast LS is not 0 - please provide a valid number for max iteration in fast LS (parameter 10)"); // Don't now if say parameter 10 or 11 if the user start couting from 1
                }
            }

            // Set the AVG hamming distance threshold that should be in the population at every iteration (excluded the first one)
            avgHammingDistance = stoi(argv[11]);

            if(avgHammingDistance < 1 || avgHammingDistance > tspInstance.numberNodes){
                avgHammingDistance = 1;
            }

            // Set if want to apply the optimization at the end of the iterations
            if(stoi(argv[12]) != 0){
                finalOpt = true;
            }

            // Set if want to use Linear Ranking Selection (Or Tournament Selection)
            if(stoi(argv[13]) != 1){
                useLinearRanking = false;
            }

            if(!useLinearRanking){
                try{
                    tournamentSize = stoi(argv[14]);
                    if(tournamentSize < 1){
                        tournamentSize = 3;
                    }
                }catch(const std::invalid_argument&){
                    throw::runtime_error("In use Tournament Selection - please provide a valid number for last parameter");
                }
            }
        }

        const int TOT_REPETITIONS = 1; // Define the desidered number of repetitions
        int countOptSolution = 0;
        double avgQuality = 0;
        double totTime = 0;
        double optSol = 0; // Put the known optimal solution for test
        double minVal = std::numeric_limits<double>::max();
        double maxVal = 0;
        Helpers::print_parameters(tspInstance.numberNodes, populationSize, maxIteration, maxIterationWithoutImprovement, maxTimer, selectedIndividualsNumber, probUseOrdCrossover, probMutation, probFastOpt, maxIterationFastLS, avgHammingDistance, finalOpt, useLinearRanking, tournamentSize);
        //TSPSolver solver(tspInstance, initialSequence, populationSize, maxIteration, maxIterationWithoutImprovement, maxTimer, selectedIndividualsNumber, probUseOrdCrossover, probMutation, probFastOpt, maxIterationFastLS, avgHammingDistance,finalOpt, useLinearRanking, tournamentSize);
        // Code for the tests, change number or repetion, by default is 1 so you can have 1 execution only of the algorithm
        for (size_t i = 0; i < TOT_REPETITIONS; i++)
        {
            TSPSolver solver(tspInstance, initialSequence, populationSize, maxIteration, maxIterationWithoutImprovement, maxTimer, selectedIndividualsNumber, probUseOrdCrossover, probMutation, probFastOpt, maxIterationFastLS, avgHammingDistance,finalOpt, useLinearRanking, tournamentSize);
            ios_base::sync_with_stdio(false);
            auto start = chrono::high_resolution_clock::now();
            double sol = solver.solve();
            auto end = chrono::high_resolution_clock::now();
            double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

            time_taken *= 1e-9;

            totTime += time_taken;

            if(sol <= optSol + 1e-2 && sol >= optSol - 1e-2 ){
                countOptSolution++;
            }

            if(sol > maxVal) maxVal = sol;
            if(sol < minVal) minVal = sol;

            double difference = sol - optSol;
            avgQuality += difference;
        }

        totTime /= TOT_REPETITIONS;
        avgQuality /= TOT_REPETITIONS;

        std::cout<<"Opt. Solution find: "<<countOptSolution<<" times"<<" with min: " << minVal << " and max: " << maxVal << endl;
        std::cout<<"AVG different from opt.: "<<avgQuality<<endl;
        std::cout<<setprecision(9)<<"AVG time execution: "<<totTime<<endl;

    }catch(std::exception& e){

        std::cout << ">>>EXCEPTION: " << e.what() << std::endl;
    }

    return 0;
}
