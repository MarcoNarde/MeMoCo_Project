/**
 * @file 1LabEs.cpp
 * @brief
 */

#include "cpxmacro.h"
#include <bits/stdc++.h>
#include <chrono>
#include <iomanip>

#include "TSPFileContainer.h"
#include "TSPData.h"

using namespace std;

// error status and messagge buffer
int status;
const int TOTAL_REPETITIONS = 1; // Change number for test
char errmsg[BUF_SIZE];

const int NAME_SIZE = 512;
char name[NAME_SIZE];

/**
*	function that define the mathematical problem with constrains and decision variables
*/
void setupLP(CEnv env, Prob lp, int & numVars , const int& N, const int& A, int nameN[], double C[])
{
	// Take track of all d.v. added
	int cur_var_pos = 0;

	// Initialize matrix to map X variables.
  vector<vector<int>> map_x;
  map_x.resize(N);
  for(int i = 0; i < N; i++){
      map_x[i].resize(N);
      for(int j = 0; j<N; j++){
          map_x[i][j] = -1;
      }
  }

	/**
    Adding x vars to the problem:
    x_ij in R_+  forall (i,j) in A,j != 0
  */
	/**
	* Iterate throught all the position in the matrix of x's variables
	* and add each Xij variablec
	*/
	for (int i = 0; i < N; i++)
	{
    for(int j = 1; j < N; j++){ // x vars of type Xn0 are not considered
      if(j == i) continue; // Don't add variables of type: X11, X22, ... Xnn

      char htype = 'C'; // Define continuous variables
      double obj = 0.0; // Coefficient in the objective function (0 for x vars)
      double lb = 0.0; // Lower bound of the variables
      double ub = CPX_INFBOUND; // Upper bound of the variables
      snprintf(name, NAME_SIZE, "x_(%d,%d)", nameN[i], nameN[j]);
      char* hname = (char*)(&name[0]); // Name to identify the vars
      CHECKED_CPX_CALL( CPXnewcols, env, lp, 1, &obj, &lb, &ub, &htype, &hname ); // Create the variable in the problem environment

			// Map the position of the variable created in the matrix
      map_x[i][j] = cur_var_pos;
      cur_var_pos++;
    }
	}

	// Initialize matrix to map Y variables.
  vector<vector<int>> map_y;
  map_y.resize(N);
  for(int i = 0; i < N; i++){
      map_y[i].resize(N);
      for(int j = 0; j<N; j++){
          map_y[i][j] = -1;
      }
  }

  /**
      Adding y vars to the problem:
      y_ij in {0,1}  forall (i,j) in A
  */
  int c_index = 0;
	for (int i = 0; i < N; i++)
	{
    for(int j = 0; j<N; j++){

	    if(j == i) continue; // // Don't add variables of type: X11, X22, ... Xnn

	    char htype = 'B'; // Define binary variables (0,1)
	    double obj = C[c_index]; // Define the coefficient in the objective function (in this case the value in C parameters)
	    double lb = 0.0; // Set lower bound
	    double ub = 1.0; // Set upper bound
      snprintf(name, NAME_SIZE, "y_(%d,%d)", nameN[i], nameN[j]);
      char* hname = (char*)(&name[0]); // Name to identify the vars

	    CHECKED_CPX_CALL( CPXnewcols, env, lp, 1, &obj, &lb, &ub, &htype, &hname ); // Add y var into problem environment

	    c_index++;
			// Map the y var added in the matrix
	    map_y[i][j] = cur_var_pos;
	    cur_var_pos++;
    }
	}

	numVars = CPXgetnumcols(env, lp); // Get number of vars in the problem

	/**
    Adding flow constraint:
    [ forall k in N \ {0}, sum{i : (i,k) in A} x_ik - sum{j: (k,j) j != 0} = 1 ]
  */
  double one = 1.0; // var to indentify value 1
	for (int k = 1; k < N; k++) // Exclude k = 0
	{
		std::vector<int> idx;
		std::vector<double> coef;
		// Get x vars from X0k to Xn-1k to and set coefficient of each x vars to 1
    for(int i = 0; i < N; i++){
      if(map_x[i][k] < 0) continue; // Avoid variables not mapped
      idx.push_back(map_x[i][k]);
      coef.push_back(1.0);
    }
		// Get x vars from Xk1 to Xkn-1 to and set coefficient of each x vars to 1
    for(int j = 1; j < N; j++){
      if(map_x[k][j] < 0) continue; // Avoid variables not mapped
      idx.push_back(map_x[k][j]);
      coef.push_back(-1.0);
    }

		char sense = 'E'; // Define constraint of type equal (=)
		int matbeg = 0; // Define the beginning of the value in the matrix

		// Add row to the problem environment
    if(idx.size() != 0)
	    CHECKED_CPX_CALL( CPXaddrows, env, lp, 0, 1, idx.size(), &one, &sense, &matbeg, &idx[0], &coef[0], 0, 0 );
	}

	/*
    Adding yij ships inside constraints:
    [ forall i in N, sum{j:  (i,j) in A} y{i,j} = 1 ]
  */
	for(int i  = 0; i < N; ++i){
		std::vector<int> idx;
		std::vector<double> coef;
		// Get index of all y vars from Yi0 to Yin-1 and set coefficient equal to 1
    for(int j = 0; j < N; j++){
      if(map_y[i][j]<0) continue; // Avoid variables not mapped

      idx.push_back(map_y[i][j]);
      coef.push_back(1.0);

    }
		char sense = 'E'; // Define constraint of type equal (=)
		int matbeg = 0; // Define the beginning of the value in the matrix

		// Add row to the problem environment
    if(idx.size() != 0)
	    CHECKED_CPX_CALL( CPXaddrows, env, lp, 0, 1, idx.size(), &one, &sense, &matbeg, &idx[0], &coef[0], 0, 0 );
	}

	/**
    Addding yij ships outside constraints [ forall j in N, sum{i:  (i,j) in A} y{i,j} = 1 ]
  */
	for(int j  = 0; j < N; ++j){
		std::vector<int> idx;
		std::vector<double> coef;
		// Get index of all y vars from Y0j to Yn-1j and set coefficient equal to 1
    for(int i = 0; i < N; i++){
        if(map_y[i][j]<0) continue; // Avoid variables not mapped

        idx.push_back(map_y[i][j]);
        coef.push_back(1.0);
    }

		char sense = 'E'; // Define constraint of type equal (=)
		int matbeg = 0; // Define the beginning of the value in the matrix

		// Add row to the problem environment
    if(idx.size() != 0)
    	CHECKED_CPX_CALL( CPXaddrows, env, lp, 0, 1, idx.size(), &one, &sense, &matbeg, &idx[0], &coef[0], 0, 0 );
	}

	/**
	* Add flow amount constrains [Forall (i,j) in A, j != 0] => xij <= (|N|-1)yij
	*/
  double zero = 0.0; // var to identify value 0
	// Define the constrain in the form xij - (|N|-1)yij <= 0
	for(int i  = 0; i < N; ++i){
    for(int j = 1; j < N; ++j){ // Avoid select var with j = 0
      if(map_x[i][j]<0) continue; // Avoid variables not mapped

      if(map_y[i][j]<0) continue; // Avoid variables not mapped

      std::vector<int> idx(2);
    	std::vector<double> coef(2);

      idx[0] = map_x[i][j];
      idx[1] = map_y[i][j];

      coef[0] = 1.0; // Define the coefficient for the x var (1)
      coef[1] = (N-1) * (-1); // Define the coefficient for the y var (-(|N|-1))
      char sense = 'L'; // Define constraint of type lower (<=)
    	int matbeg = 0; // Define the beginning of the value in the matrix

			// Add row to the problem environment
    	CHECKED_CPX_CALL( CPXaddrows, env, lp, 0, 1, idx.size(), &zero, &sense, &matbeg, &idx[0], &coef[0], 0, 0 );
    }
	}

  CHECKED_CPX_CALL( CPXwriteprob, env, lp, "BoardMaker.lp", 0 );
}

int main (int argc, char const *argv[])
{
  double objval; // Will contain the value for the objective function
  try{
		// Need filename parameter
    if (argc < 2) throw std::runtime_error("usage: ./main filename.dat");
    TSPFileContainer tspFileContainer;
    tspFileContainer.read(argv[1]); // Read data from file
    TSPData dataTSP(tspFileContainer.numberNodes, tspFileContainer.values);

    double totTime = 0;
		// Start benchmark
    for(int r = 0; r <TOTAL_REPETITIONS; r++){
      auto start = chrono::high_resolution_clock::now();
      ios_base::sync_with_stdio(false);
      try
      {
        DECL_ENV( env );
        DECL_PROB( env, lp );
        int numVars;
        setupLP(env, lp, numVars, dataTSP.N, dataTSP.A, dataTSP.nameN, dataTSP.C);
        CHECKED_CPX_CALL(CPXchgobjsen, env, lp, CPX_MIN);
        CHECKED_CPX_CALL( CPXmipopt, env, lp );
        CHECKED_CPX_CALL( CPXgetobjval, env, lp, &objval );
        int n = CPXgetnumcols(env, lp);
        std::vector<double> varVals;
        varVals.resize(n);
        CHECKED_CPX_CALL( CPXgetx, env, lp, &varVals[0], 0, n - 1 );
        // TO SEE THE VARIABLES VALUE
        // for ( int i = 0 ; i < n ; ++i ) {
        // std::cout << "var in position " << i << " : " << varVals[i] << std::endl;
        // }
        CHECKED_CPX_CALL( CPXsolwrite, env, lp, "boardmaker.sol" );
        CPXfreeprob(env, &lp);
        CPXcloseCPLEX(&env);
      }
      catch(std::exception& e)
      {
          std::cout << ">>>EXCEPTION: " << e.what() << std::endl;
      }
      auto end = chrono::high_resolution_clock::now();
      double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

      time_taken *= 1e-9; // Round the time

      totTime += time_taken;
    }
    totTime /= TOTAL_REPETITIONS;

    std::cout<<"Total time: "<<totTime<<setprecision(9)<<endl;
    std::cout << "Objval: " << objval << std::endl;

    return 0;
  }catch(std::exception& e2)
  {
    std::cout << ">>>EXCEPTION: " << e2.what() << std::endl;
  }
}
