from music21 import *

def generate_piano_midi(left, right):
	s = stream.Stream()
	for n in right:
		curr_note = note.Note(n)
		s.append(curr_note)
	s.write('midi', 'melody_test.mid')
