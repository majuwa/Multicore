CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-g -fpermissive `pkg-config --cflags opencv` -std=c++11  -I compute/include
LDFLAGS=-g 
LDLIBS=-lOpenCL -lboost_timer -lboost_system -lboost_thread `pkg-config --libs opencv`

SRCS=$(wildcard *.cpp)
OBJS=$(subst .cpp,.o,$(SRCS))

.PHONY=all run edge_lap clean dist-clean

all: edge_lap

run: edge_lap
	./$^

edge_lap: $(OBJS)
	g++  -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS) edge_lap

dist-clean: clean
	$(RM) *~ .dependtool

include .depend
