/**
 * @file TSPFileContainer.h
 * @brief TSP file container
 *
 */

#ifndef TSPFILECONTAINER_H
#define TSPFILECONTAINER_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <sstream>
#include <bits/stdc++.h>

class TSPFileContainer
{
public:
  TSPFileContainer() : numberNodes(0) { }
  int numberNodes;
  std::vector<std::vector<double>> values;

  void read(const std::string filename)
  {
    std::fstream MyReadFile(filename);
    int counter = 0;
    std::string myText;
    
    // Use a while loop together with the getline() function to read the file line by line
    while (getline (MyReadFile, myText)) {
    
      if(counter == 1){
        //read number nodes
        std::vector<std::string> parts = split(myText, ':');
        numberNodes = stoi(trim(parts[1]));

        values.resize(numberNodes);
        for(int i = 0; i < numberNodes; i++){
          values[i].resize(numberNodes);
          for(int j = 0; j<numberNodes; j++){
            values[i][j] = -1;
          }
        }
      }
      if(counter >= 4){
        //read value of edge
        std::vector<std::string> parts = split(myText, ' ');
        
        values[stod(trim(parts[0]))][stod(trim(parts[1]))] = stod(trim(parts[2]));
        values[stod(trim(parts[1]))][stod(trim(parts[0]))] = stod(trim(parts[2]));
      }
      counter++;
    }

    // Close the file
    MyReadFile.close();
  }
  
  //Helping Function to split a string
  std::vector<std::string> split(const std::string& str, const char delim) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream token_stream(str);
    while (getline(token_stream, token, delim)) {
      tokens.push_back(token);
    }
    return tokens;
  }

//Helping Function to delete spaces in string
  std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first) {
      return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
  }
};

#endif /* TSPFILECONTAINER_H */
