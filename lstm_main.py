from piano_trainer import *
from create_data_sets import *
from generate_music import *
from autoencoder import *


def main():
	left, right = create_piano_sets()
	melody = train_piano_melody(right, sequence_length=8, temperature=1.0, epochs=20)
	accompany = train_piano_accompany(left, sequence_length=8, temperature=1.0, epochs=20)
	generate_piano_midi(accompany, melody)

if __name__ == '__main__':
	main()