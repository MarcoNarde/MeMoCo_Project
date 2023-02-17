/**
* @file TSPEncode.h
* @brief TSP Encoding
*
*/

#ifndef TSPENCODE_H
#define TSPENCODE_H

#include "TSP.h"

/**
* TSP Solution representation: ordered sequence of nodes (path representation)
*/
class TSPEncode
{
public:
  std::vector<int> sequence;
  double cost;
public:
  /** Constructor 
  * build a standard solution as the sequence <0, 1, 2, 3 ... n-1, 0>
  * @param n size of sequence
  * @return ---
  */
  TSPEncode(int n);
  /** Constructor 
  * build a standard solution as the sequence <0, 1, 2, 3 ... n-1, 0>
  * @param tsp TSP instance
  * @return ---
  */
  TSPEncode( const TSP& tsp );

  /** Copy constructor 
  * build a solution from another
  * @param tspSol TSP solution
  * @return ---
  */
  TSPEncode( const TSPEncode& tspSol );

  /** Constructor 
  * build a solution from a sequence
  * @param tspSol TSP solution
  * @return ---
  */
  TSPEncode( const std::vector<int>& seq , double& weight);

public:
  /** assignment method 
  * copy a solution into another one
  * @param right TSP solution to get into
  * @return ---
  */
  TSPEncode& operator=(const TSPEncode& right);

  /** Print the TSP solution
  * @param out reference to output stream
  * @param obj TSP solution to be printed
  * @return reference to output stream
  */
  friend ostream& operator<<(ostream &out, const TSPEncode &obj);

  /** Equality method
  * check if two TSP solutions are equal
  * @param other TSP solution to compare with
  * @return true if equal, false otherwise
  */
  bool operator==(const TSPEncode &other) const;
};

#endif /* TSPENCODE_H */
