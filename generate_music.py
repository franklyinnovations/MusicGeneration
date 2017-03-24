from music21 import *

def generate_piano_midi(left, right):
	normalizing_constant = 28
	s = stream.Stream()
	p0 = stream.Part()
	p1 = stream.Part()
	for n in right:
		curr_note = note.Note(n + normalizing_constant)
		p0.append(curr_note)
	for n in left:
		curr_note = note.Note(n + normalizing_constant)
		p1.append(curr_note)
	s.insert(0, p0)
	s.insert(1, p1)
	s.write('midi', 'melody_test.mid')