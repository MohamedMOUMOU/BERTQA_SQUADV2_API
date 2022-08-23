import torch
from transformers import AutoTokenizer
import onnxruntime
import numpy as np

max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
n_best_size = 5 # The number of answers to consider in the calid answers array
max_answer_length = 30 # The manximum numbers of tokens an answer can have
min_answer_score = 7 # The lowest score accepted in an answer

def bertqaGetAnswers(question, paragraph):
	tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

	# encoding both the question and the answer into tokens
	encoding = tokenizer(
		text=question,
		text_pair=paragraph, 
		max_length=max_length,
		truncation="only_second",
	    	return_overflowing_tokens=True,
	    	return_offsets_mapping=True,
	    	stride=doc_stride
	)

	# The offset mappings will give us a map from token to character position in the original context. This will
	# help us compute the start_positions and end_positions.
	offset_mapping = encoding.pop("offset_mapping")

	# initializing the valid answers array
	valid_answers = []

	for i in range(0, len(encoding['input_ids'])):
		start = offset_mapping[i][0][0]
		end = offset_mapping[i][len(offset_mapping[i])-2][-1]
		inputs = encoding['input_ids'][i] # inputs_ids, mapping every word in the question and a context to a number
		sentence_embedding = encoding['token_type_ids'][i]  # Segment embeddings
		attention_mask = encoding['attention_mask'][i] # attention mask

		# model path
		model = "modelexp.onnx"
		# initializing and inference session
		session = onnxruntime.InferenceSession(model, None)

		result = session.run(None, {'input_ids': torch.tensor([inputs]).numpy(), 'attention_mask': torch.tensor([attention_mask]).numpy(), 'token_type_ids': torch.tensor([sentence_embedding]).numpy()})

		# start logits -> all the tokens of the context and their probabilities of being the start of the answer
		start_logits = torch.from_numpy(result[0][0][:,0]).numpy()
		# start logits -> all the tokens of the context and their probabilities of being the end of the answer
		end_logits = torch.from_numpy(result[0][0][:,1]).numpy()
		# Gather the indices the best start/end logits:
		start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
		end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

		# iterating over combinations of th best possible answers
		for start_index in start_indexes:
			for end_index in end_indexes:
				if start_index <= end_index and end_index - start_index + 1 < max_answer_length: # We need to refine that test to check the answer is inside the context
					answer_start = offset_mapping[i][start_index][0] # The index of the first character of the candidate answer
					answer_end = offset_mapping[i][end_index][-1] # The index of the last character of the candidate answer
					# ppending to the array of valid answers
					valid_answers.append(
						{
						"score": start_logits[start_index] + end_logits[end_index],
						"text": paragraph[answer_start:answer_end],
						"order": i,
						"start_index": start_index,
						"end_index": end_index
						}
					)
	# sorting the array of valid answers with respect to the score of the answer
	valid_answers.sort(key=lambda x: x['score'], reverse=True)
	print(valid_answers)
	answer = ''

	# checking if the answer is empty (no answer) or if the scores of teh answer is too low, in which case we choose to invalidate the answer
	if valid_answers[0]['text'] == '' or valid_answers[0]['score'] < min_answer_score:
		answer = None
	else:
		answer = valid_answers[0]['text']

	return answer
