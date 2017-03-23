import tflearn
from create_data_sets import *

def train_piano_melody(right, sequence_length=8, temperature=1.0, epochs=5):
	piano_melody = []
	for sequence in right:
		piano_melody.extend(sequence)

	train_melody_X, train_melody_Y, melody_dict = create_train_sequence(piano_melody)

	melody_seed = random_sample_note_sequence(piano_melody, sequence_length)
	melody_seed = [45, 46, 48, 49]

	print('Training melody...')
	melody_trainer = tflearn.input_data([None, sequence_length, len(melody_dict)])
	melody_trainer = tflearn.lstm(melody_trainer, 256, return_seq=True)
	melody_trainer = tflearn.dropout(melody_trainer, 0.5)
	melody_trainer = tflearn.lstm(melody_trainer, 256, return_seq=True)
	melody_trainer = tflearn.dropout(melody_trainer, 0.5)
	melody_trainer = tflearn.lstm(melody_trainer, 256)
	melody_trainer = tflearn.dropout(melody_trainer, 0.5)
	melody_trainer = tflearn.fully_connected(melody_trainer, len(melody_dict), activation='softmax')
	melody_trainer = tflearn.regression(melody_trainer, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

	melody_generator = tflearn.SequenceGenerator(melody_trainer, 
						dictionary=melody_dict, seq_maxlen=sequence_length, clip_gradients=5.0)
	melody_generator.fit(train_melody_X, train_melody_Y, 
							validation_set=0.1, batch_size=32, n_epoch=epochs)

	return melody_generator.generate(30, temperature=temperature, seq_seed=melody_seed)

def train_piano_accompany(left, sequence_length=8, temperature=1.0, epochs=5):
	pinao_accompany = []
	for sequence in left:
		pinao_accompany.extend(sequence)

	train_accompany_X, train_accompany_Y, accompany_dict = create_train_sequence(pinao_accompany)
	accompany_seed = random_sample_note_sequence(pinao_accompany, sequence_length)

	print('Training accompany...')
	accompany_trainer = tflearn.input_data([None, sequence_length, len(accompany_dict)])
	accompany_trainer = tflearn.lstm(accompany_trainer, 256, return_seq=True)
	accompany_trainer = tflearn.dropout(accompany_trainer, 0.5)
	accompany_trainer = tflearn.lstm(accompany_trainer, 256, return_seq=True)
	accompany_trainer = tflearn.dropout(accompany_trainer, 0.5)
	accompany_trainer = tflearn.lstm(accompany_trainer, 256)
	accompany_trainer = tflearn.dropout(accompany_trainer, 0.5)
	accompany_trainer = tflearn.fully_connected(accompany_trainer, len(accompany_dict), activation='softmax')
	accompany_trainer = tflearn.regression(accompany_trainer, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

	accompany_generator = tflearn.SequenceGenerator(accompany_trainer, 
						dictionary=accompany_dict, seq_maxlen=sequence_length, clip_gradients=5.0)
	accompany_generator.fit(train_accompany_X, train_accompany_Y, 
							validation_set=0.1, batch_size=32, n_epoch=epochs)
	return accompany_generator.generate(30, temperature=temperature, seq_seed=accompany_seed)
	
