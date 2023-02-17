/**
 * @file TSPData.h
 * @brief TSP data
 *
 */

#ifndef TSPDATA_H
#define TSPDATA_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <bits/stdc++.h>

class TSPData
{
  public:
    const int N;
    const int A;
    int *nameN; //Names of nodes
    double *C;  //Cost vector

    TSPData(int n, std::vector<std::vector<double>> values): N(n), A(n*(n-1)/2){
      nameN = new int[N];
      C = new double[A*2];
    
      //initialize nameN vector
      for(int i = 0; i < N; i++){
          nameN[i] = i;
      }

      //initialize C vector
      int count = 0;
      for(int i = 0; i < N; i++){
          for(int j = 0; j< N; j++){
              if(i != j && values[i][j] != -1){
                  C[count] = values[i][j];
                  count++;
              }
          }
      }
    }

};

#endif /* TSPDATA_H */
