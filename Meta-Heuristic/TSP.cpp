#include "TSP.h"
TSP::TSP() : numberNodes(0) { }

bool TSP::read(const std::string filename)
{ 
  std::fstream MyReadFile(filename);
  int counter = 0;
  std::string myText;
  try{
    if (!MyReadFile.is_open()) throw std::runtime_error("Can't open the file, file not found");
    
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
    return true;
  }catch(const std::exception& e)
  {
    // Error handling
    std::cerr << ">>>EXCEPTION: Error while reading file: " << e.what() << std::endl;
    MyReadFile.close();
    return false;
  }
}

std::vector<std::string> TSP::split(const std::string& str, const char delim) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream token_stream(str);
  while (getline(token_stream, token, delim)) {
    tokens.push_back(token);
  }
  return tokens;
}

//Helping Function to delete spaces in string
std::string TSP::trim(const std::string& str) {
  size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first) {
    return str;
  }
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}