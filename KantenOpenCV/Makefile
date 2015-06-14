CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-g -fpermissive `pkg-config --cflags opencv`
LDFLAGS=-g 
LDLIBS=-lOpenCL -lboost_timer -lboost_system -lboost_thread `pkg-config --libs opencv`

SRCS=$(wildcard *.cpp)
OBJS=$(subst .cpp,.o,$(SRCS))

.PHONY=all run edge_opencv clean dist-clean

all: edge_opencv

run: edge_opencv
	./$^

edge_opencv: $(OBJS)
	g++  -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS) edge_opencv

dist-clean: clean
	$(RM) *~ .dependtool

include .depend
