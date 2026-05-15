CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic

BUILD_DIR := build
TARGET := $(BUILD_DIR)/viterbi_sim

.PHONY: all run plot clean

all: $(TARGET)

$(TARGET): src/main.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

plot: run
	python3 plot.py

clean:
	rm -rf $(BUILD_DIR) results
