import torch
from transformers import CTRLLMHeadModel, CTRLTokenizer, BertConfig, BertTokenizer, BertForSequenceClassification

import os
import datetime
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def dec2other(num, base):
	l = []
	while True:
		num, reminder = divmod(num, base)  # 算除法求除数和余数
		l.append(str(reminder))  # 将余数存入字符串
		if num == 0:
			return l[::-1]


def main():
	# ======================
	# 超参数
	# ======================
	CLASS_NUM = 2
	LENGTH = 50
	TEMPERATURE = 1
	K = 0
	P = 0.9
	REPETITION_PENALTY = 1.2
	GENERATE_NUM = 1000
	STOP_TOKEN = None
	LOAD_EPOCH_BERT = 8

	if CLASS_NUM in [2,4,8,16]:
		PRECISION = int(np.log2(CLASS_NUM))
	else:
		PRECISION = 100

	all_var = locals()
	print()
	for var in all_var:
		if var != 'var_name':
			print('{0:15}   '.format(var), all_var[var])
	print()

	# ======================
	# 数据
	# ======================
	with open('control_codes', 'r', encoding='utf8') as f:
		control_codes = f.readlines()
		control_codes = [_.strip() for _ in control_codes]
		control_codes = control_codes[:CLASS_NUM]
	with open('prompts', 'r', encoding='utf8') as f:
		prompts = f.readlines()
		prompts = [_.strip() for _ in prompts]

	# ======================
	# 构建模型
	# ======================
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# ctrl
	tokenizer_ctrl = CTRLTokenizer.from_pretrained('ctrl', cache_dir='/data1/zhangsy/_cache/torch/transformers/ctrl')
	model_ctrl = CTRLLMHeadModel.from_pretrained('ctrl', cache_dir='/data1/zhangsy/_cache/torch/transformers/ctrl')
	model_ctrl.to(device)
	# bert
	config = BertConfig()
	config.num_labels = CLASS_NUM
	tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/data1/zhangsy/_cache/torch/transformers/bert-base-uncased')
	model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config, cache_dir='/data1/zhangsy/_cache/torch/transformers/bert-base-uncased')
	model_bert.to(device)
	model_bert.load_state_dict(torch.load('models/' + '_'.join(control_codes) + '-' + str(LOAD_EPOCH_BERT) + '.pkl', map_location=device))
	print('bert checkpoint loaded')
	print()

	# ======================
	# 隐写
	# ======================
	os.makedirs('stego/', exist_ok=True)
	# read bit streams
	with open('bit_stream/bit_stream.txt', 'r', encoding='utf8') as f:
		bit_stream = f.read().strip()
	i = 0
	bit_stream_ = []
	while i < len(bit_stream):
		s = bit_stream[i:i + PRECISION]
		i += PRECISION
		bit_stream_.extend(dec2other(int(s, base=2), CLASS_NUM))
	bit_stream = bit_stream_
	bit_index = int(torch.randint(0, high=1000, size=(1,)))

	model_ctrl.eval()
	model_bert.eval()
	with torch.no_grad():
		stega_text = []
		stega_bits = []
		stega_counts = []
		stega_times = []

		for _ in range(GENERATE_NUM):
			if device == 'cuda':
				torch.cuda.synchronize()
			start_time = datetime.datetime.now()
			stega_count = 0
			# read secret messages
			int_embed = int(bit_stream[bit_index])
			control_code = control_codes[int_embed]

			while True:
				stega_count += 1
				# random prompt
				np.random.shuffle(prompts)
				prompt = prompts[0]

				# generate sentences
				prompt_text = control_code + ' ' + prompt
				encoded_prompt = tokenizer_ctrl.encode(prompt_text, add_special_tokens=False, return_tensors="pt", )
				encoded_prompt = encoded_prompt.to(device)
				if encoded_prompt.size()[-1] == 0:
					input_ids = None
				else:
					input_ids = encoded_prompt
				output_sequences = model_ctrl.generate(
					input_ids=input_ids,
					max_length=LENGTH,
					temperature=TEMPERATURE,
					top_k=K,
					top_p=P,
					repetition_penalty=REPETITION_PENALTY,
					do_sample=True,
					num_return_sequences=1,
				)   # (batch_size * num_return_sequences, sequence_length)
				generated_sequence = output_sequences[0].tolist()
				text = tokenizer_ctrl.decode(generated_sequence, clean_up_tokenization_spaces=True)
				text = text[: text.find(STOP_TOKEN) if STOP_TOKEN else None]
				stega_sentence = prompt + text[len(tokenizer_ctrl.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]

				# bert classification
				inputs = tokenizer_bert(stega_sentence, return_tensors="pt")
				for k in inputs.keys():
					inputs[k] = inputs[k].to(device)
				y = model_bert(**inputs)[0]
				if int(torch.argmax(y)) == int_embed:
					stega_text.append(stega_sentence)
					stega_bits.append(bit_stream[bit_index])
					bit_index += 1
					stega_counts.append(stega_count)
					break
				else:
					print('generate:' + control_code)
					print('get:     ' + control_codes[int(torch.argmax(y))])

			if device == 'cuda':
				torch.cuda.synchronize()
			end_time = datetime.datetime.now()
			stega_times.append((end_time - start_time).total_seconds())

		# write files
		with open('stego/' + '_'.join(control_codes) + '.txt', 'w', encoding='utf8') as f:
			for sentence in stega_text:
				f.write("\n=== GENERATED SEQUENCE ===\n")
				f.write(sentence)
		with open('stego/' + '_'.join(control_codes) + '.bit', 'w', encoding='utf8') as f:
			for bits in stega_bits:
				f.write(bits + '\n')
		with open('stego/' + '_'.join(control_codes) + '.count', 'w', encoding='utf8') as f:
			for count in stega_counts:
				f.write(str(count) + '\n')
		with open('stego/' + '_'.join(control_codes) + '.time', 'w', encoding='utf8') as f:
			for time in stega_times:
				f.write(str(time) + '\n')

		print('count')
		print(np.mean(stega_counts))
		print(np.std(stega_counts, ddof=1))
		print('time')
		print(np.mean(stega_times))
		print(np.std(stega_times, ddof=1))


if __name__ == '__main__':
	main()
