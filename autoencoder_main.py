from piano_trainer import *
from create_data_sets import *
from generate_music import *
from autoencoder import *


def main():
	left, right = create_piano_sets()
	melodies = autoencoder_melody(right, 20)
	accompanies = autoencoder_accompany(left, 20)
	generate_piano_midi(accompanies[0][0], melodies[0][0])

if __name__ == '__main__':
	main()