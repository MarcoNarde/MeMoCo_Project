CC = g++
CPPFLAGS = -g -Wall -O2
LDFLAGS =

SOURCES = main.cpp TSP.cpp TSPEncode.cpp TSPSolver.cpp Helpers.cpp
HEADERS = TSP.h TSPEncode.h TSPSolver.h Helpers.h
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE=main

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

%.o: %.cpp $(HEADERS)
	$(CC) $(CPPFLAGS) -c $< -o $@

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS)

