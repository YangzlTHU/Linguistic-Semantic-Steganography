import torch
import torch.optim as optim
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

import numpy as np

import utils

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
	# ======================
	# 超参数
	# ======================
	BATCH_SIZE = 16
	RATIO = 0.9
	CLASS_NUM = 3
	EPOCH = 10
	LEARNING_RATE = 0.001
	SEED = 100

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
	paths = ['generate/' + c for c in control_codes]
	os.makedirs('tmp', exist_ok=True)
	train_paths = ['tmp/train_' + c for c in control_codes]
	test_paths = ['tmp/test_' + c for c in control_codes]

	for i in range(len(control_codes)):
		utils.split_corpus(paths[i], train_paths[i], test_paths[i], max_len=1000, min_len=0, ratio=RATIO, seed=SEED, split_token='\n=== GENERATED SEQUENCE ===\n')
	# train
	train = []
	train_labels = []
	for i in range(len(control_codes)):
		with open(train_paths[i], 'r', encoding='utf8') as f:
			raw = f.read()
			raw = raw.split('\n=== GENERATED SEQUENCE ===\n')
			raw.remove('')
			train.extend(raw)
			train_labels.extend([i] * len(raw))
	# test
	test = []
	test_labels = []
	for i in range(len(control_codes)):
		with open(test_paths[i], 'r', encoding='utf8') as f:
			raw = f.read()
			raw = raw.split('\n=== GENERATED SEQUENCE ===\n')
			raw.remove('')
			test.extend(raw)
			test_labels.extend([i] * len(raw))

	train = [_.split(' ', 1)[1] for _ in train]
	test = [_.split(' ', 1)[1] for _ in test]

	train_generator = utils.Generator(train, train_labels)
	test_generator = utils.Generator(test, test_labels)

	# ======================
	# 构建模型
	# ======================
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	config = BertConfig()
	config.num_labels = CLASS_NUM
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/data1/zhangsy/_cache/torch/transformers/bert-base-uncased')
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config, cache_dir='/data1/zhangsy/_cache/torch/transformers/bert-base-uncased')
	model.to(device)
	total_params = sum(p.numel() for p in model.parameters())
	print("Total params: {:d}".format(total_params))
	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Trainable params: {:d}".format(total_trainable_params))

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
	# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))
	# optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
	print()

	# ======================
	# 训练与测试
	# ======================
	best_loss = 1000000
	for epoch in range(EPOCH):
		train_g = train_generator.build_generator(BATCH_SIZE, padding=False)
		test_g = test_generator.build_generator(BATCH_SIZE, padding=False)
		train_loss = []
		train_acc = []
		model.train()
		while True:
			try:
				text, label = train_g.__next__()
			except:
				break
			optimizer.zero_grad()
			# input
			inputs = tokenizer(list(text), return_tensors="pt", padding=True)
			for k in inputs.keys():
				inputs[k] = inputs[k].to(device)
			# label
			label = torch.from_numpy(label).long().to(device)
			outputs = model(**inputs, labels=label)
			loss, y = outputs[:2]
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())
			y = y.cpu().detach().numpy()
			train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

		test_loss = []
		test_acc = []
		model.eval()
		while True:
			with torch.no_grad():
				try:
					text, label = test_g.__next__()
				except:
					break
				# input
				inputs = tokenizer(list(text), return_tensors="pt", padding=True)
				for k in inputs.keys():
					inputs[k] = inputs[k].to(device)
				# label
				label = torch.from_numpy(label).long().to(device)
				outputs = model(**inputs, labels=label)
				loss, y = outputs[:2]
				test_loss.append(loss.item())
				y = y.cpu().numpy()
				test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

		print('epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}'
		      .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))

		if np.mean(test_loss) < best_loss:
			best_loss = np.mean(test_loss)
			print('-----------------------------------------------------')
			print('saving parameters')
			os.makedirs('models', exist_ok=True)
			torch.save(model.state_dict(), 'models/' + '_'.join(control_codes) + '-' + str(epoch + 1) + '.pkl')
			print('-----------------------------------------------------')


if __name__ == '__main__':
	main()
