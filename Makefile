gif-h:
	git clone https://github.com/charlietangora/gif-h

all: gif-h
	g++ game_of_life.cpp -std=c++20 -o game_of_life

clean:
	rm -rf game_of_life

run: all
	./game_of_life

