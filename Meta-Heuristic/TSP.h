/**
 * @file TSP.h
 * @brief TSP data file
 *
 */

#ifndef TSP_H
#define TSP_H

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
using namespace std;

class TSP
{
public:
  int numberNodes;
  std::vector<std::vector<double>> values;

  TSP();
  bool read(const std::string filename);
  
  //Helping Function to split a string
  std::vector<std::string> split(const std::string& str, const char delim);

//Helping Function to delete spaces in string
  std::string trim(const std::string& str);
};

#endif /* TSP_H */
