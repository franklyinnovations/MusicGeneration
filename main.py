from piano_trainer import *
from create_data_sets import *
from generate_music import *


def main():
	left, right = create_piano_sets()
	melody = train_piano_melody(right)
	#accompany = train_piano_accompany(left)
	generate_piano_midi([], melody)

if __name__ == '__main__':
	main()