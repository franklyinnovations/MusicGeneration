import os
import random
import numpy as np
from music21 import *


def create_piano_sets():
	piano_raw_path = 'JayZhou_piano_sets/'
	piano_left = []
	piano_right = []

	print('Processing music files...')
	for file_name in os.listdir(piano_raw_path):
		if file_name.endswith('.mid'):
			s = converter.parse(piano_raw_path + file_name)
			#for part in s.parts:
				#for n in part.notes:
					#print(n, n.offset, n.pitch.diatonicNoteNum)
					#if n.isNote:
						#print(n.pitch.diatonicNoteNum)

					#if n.isChord:
						#print(n.pitches[0].diatonicNoteNum)
			left = []
			right = []


			for el in s.parts[1].recurse():
				if 'Note' in el.classes:
					left.append(el.pitch.diatonicNoteNum)

				if 'Chord' in el.classes:
					left.append(el.pitches[len(el.pitches) - 1].diatonicNoteNum)
		
			for el in s.parts[0].recurse():
				if 'Note' in el.classes:
					right.append(el.pitch.diatonicNoteNum)

				if 'Chord' in el.classes:
					right.append(el.pitches[len(el.pitches) - 1].diatonicNoteNum)

			piano_left.append(left)
			piano_right.append(right)

	return piano_left, piano_right

def create_train_sequence(note_sequence, sequence_length=8, repetition_step=1, note_dict=None):
	print('Producing training sequences...')
	if note_dict is None:
		note_dict = create_note_dictionary(note_sequence)

	note_count = len(note_dict)
	sequence = []
	next_note = []
	for i in range(0, len(note_sequence) - sequence_length, repetition_step):
		sequence.append(note_sequence[i:i+sequence_length])
		next_note.append(note_sequence[i+sequence_length])

	X = np.zeros((len(sequence), sequence_length, note_count), dtype=np.bool)
	Y = np.zeros((len(sequence), note_count), dtype=np.bool)

	for i, notes in enumerate(sequence):
		for t, note in enumerate(notes):
			X[i, t, note_dict[note]] = 1
		Y[i, note_dict[next_note[i]]] = 1

	return X, Y, note_dict

def create_note_dictionary(note_sequence):
	notes = set(note_sequence)
	note_dict = {n: i for i, n in enumerate(sorted(notes))}
	return note_dict

def random_sample_note_sequence(note_sequence, sequence_length):
	index = random.randint(0, len(note_sequence) - sequence_length - 1)
	return note_sequence[index:index+sequence_length]

			
	


