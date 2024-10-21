gif-h:
	git clone https://github.com/charlietangora/gif-h

serial: gif-h
	g++ game_of_life.cpp -std=c++20 -o game_of_life

clean:
	rm -rf game_of_life game_of_life_cuda

cuda:
	nvcc game_of_life.cu -o game_of_life_cuda -O2

all: serial cuda

run: all
	./game_of_life
	./game_of_life_cuda

