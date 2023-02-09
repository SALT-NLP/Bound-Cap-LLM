import os
from collections import defaultdict

import nltk
import json
import numpy as np
import scipy
import argparse
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import simpledorff
from simpledorff.metrics import interval_metric
from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str)
parser.add_argument("--input_file", type=str)
parser.add_argument("--input_file_2", type=str)
parser.add_argument("--reference_file", type=str)
parser.add_argument("--settings", nargs='+', type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--visualize", action='store_true')
parser.add_argument("--failure_ratio", action='store_true')
parser.add_argument("--no_echo", action='store_true')
parser.add_argument("--getCSV_CSV", action='store_true')
parser.add_argument("--analyze_Numerical", action='store_true')
parser.add_argument("--analyze_Descriptive", action='store_true')
parser.add_argument("--getCSV", action='store_true')
parser.add_argument("--process_raw_result", action='store_true')
parser.add_argument("--no_hardness", action='store_true')
parser.add_argument("--analyze_Style", action='store_true')
parser.add_argument("--analyze_Sensitivity", action='store_true')
parser.add_argument("--create_Numerical_files", action='store_true')
parser.add_argument("--beta", action='store_true')
parser.add_argument("--l_id", nargs='+', type=int)
parser.add_argument("--s_id", nargs='+', type=int)
args = parser.parse_args()

def paragraph_counter(text):
	if '1.' in text and '2.' in text:
		return 0
	if 'I.' in text and 'II.' in text:
		return 0
	if 'II' in text and 'IV' in text:
		return 0
	base = 0 if args.no_echo else 1
	paragraphs = text.split('\n\n')
	for para in paragraphs:
		if len(para.strip('\n')) == 0:
			base += 1
	paragraph_count = len(paragraphs) - base
	return paragraph_count

def sentence_counter(text):
	if '1.' in text and '2.' in text:
		return 0
	if 'I.' in text and 'II.' in text:
		return 0
	if 'II' in text and 'IV' in text:
		return 0
	sentence_count = len(nltk.tokenize.sent_tokenize(text))
	return sentence_count

def word_counter(text):
	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)
	if args.verbose:
		print(tokens)
	word_count = len(tokens)
	return word_count


def create_Numerical_files():
	for level, level2 in zip(("words", "sentences", "paragraphs"), ("sentence", "paragraph", "passage")):
		for num in ("5", "10", "20"):
			print(f"Write a {level2} about Shanghai with {num} {level}:")


def analyze_Numerical():
	raw_files = glob.glob(os.path.join(args.input_file, '*.json'))
	raw_files.sort(key=lambda s: int(s.split('/')[-1].split('_')[0]))
	if args.verbose:
		print(raw_files)
	data = defaultdict(list)
	subjects = ['Love', 'Cats', 'Running']
	requirements = ['Five', 'Ten', 'Twenty']
	levels = ['Word', 'Sentence', 'Paragraph']
	subject_level = [f"{s}, {l}" for s in subjects for l in levels]
	# print(subject_level)
	requirements_map = {'Five': 5, 'Ten': 10, 'Twenty': 20}
	cc = 0
	for raw_path in raw_files:
		with open(raw_path, "r") as f:
			outputs = json.load(f)
		n = len(outputs['choices'])
		id = int(raw_path.split('/')[-1].split('_')[0])
		if args.no_echo:
			if id < 7:
				continue
			data['Subject'].extend([subjects[2 if id == 9 else 0]] * n)
			data['requirement'].extend([requirements[2 - (id - 7)]] * n)
			data['Level'].extend([levels[id - 7]] * n)
			requ = requirements_map[data['requirement'][-1]]
		else:
			if id > 26:
				id -= 27
				data['Subject'].extend(['Shanghai' if id < 36 else 'Musk'] * n)
				data['requirement'].extend([requirements[id % 3]] * n)
				data['Level'].extend([levels[id // 3 % 3]] * n)
				data['Subject_Level'].extend([f"Shanghai, {levels[id // 3 % 3]}"] * n)
				requ = requirements_map[data['requirement'][-1]]
			else:
				data['Subject'].extend([subjects[id % 3]] * n)
				data['requirement'].extend([requirements[id // 3 % 3]] * n)
				data['Level'].extend([levels[id // 9]] * n)
				data['Subject_Level'].extend([f"{subjects[id % 3]}, {levels[id // 9]}"] * n)
				requ = requirements_map[data['requirement'][-1]]
		fail = 0

		prompt = outputs['choices'][0]['text'].split('\n')[0]
		if args.verbose:
			print(prompt)
		if args.no_echo:
			prompt_cnt = 0
		else:
			prompt_cnt = word_counter(prompt)

		for choice in outputs['choices']:
			text = choice['text'].strip('\n')
			if args.verbose:
				print(text)
			cnt = paragraph_counter(text)
			# print(cnt)
			if data['Level'][-1] != 'Paragraph':
				if cnt == 0 or cnt > 1:
					cnt = 0
				else:
					cnt = sentence_counter(text)
					# print(cnt)
					if data['Level'][-1] != 'Sentence':
						if cnt == 0 or cnt > 1:
							cnt = 0
						else:
							cnt = word_counter(text) - prompt_cnt
			fail += cnt != requ
			data['count'].append(cnt)
		print(f'{fail / n:.2f}', end='\t')
		for cnt in data['count'][-n:]:
			print(cnt, end='\t')
		print()
	df = pd.DataFrame.from_dict(data)

	if args.verbose:
		print(df)
	for requ in requirements:
		print(f"{requ}\'s zero rate: {len(df.loc[(df['count'] == 0) & (df['requirement'] == requ)].index) / 90 * 100}")
		print(f"{requ}\'s fail rate: {100 - len(df.loc[(df['count'] == requirements_map[requ]) & (df['requirement'] == requ)].index) / 90 * 100}")
	if args.visualize:
		# g = sns.displot(kind='hist', x="count", hue='Subject', row='requirement', col="Level", multiple='stack',stat='count', binwidth=1, discrete=True, data=df, facet_kws={'margin_titles': True})
		palette = sns.color_palette("Blues", n_colors=4)[1:] + sns.color_palette("YlOrBr", n_colors=4)[1:] + sns.color_palette("light:g", n_colors=4)[1:]
		g = sns.displot(kind='hist', x="count", hue='Level', hue_order=levels, palette="pastel", col='requirement', multiple='stack',
		                stat='count', binwidth=1, discrete=True, data=df.loc[df['count'] != 0], facet_kws={'margin_titles': True})
		g.set_titles(col_template="Required Count: {col_name}") # , row_template="{row_name}")
		g.set(xlim=(0, 25), xticks=[0, 5, 10, 15, 20, 25], yticklabels =[f"{y / 90 * 100:.0f}%" for y in g.axes[0,0].get_yticks()])
		g.set_xlabels('')
		g.fig.supxlabel('Actual Count', x=0.4)
		g.set_ylabels('Percentage')
		sns.move_legend(g, loc="lower center", bbox_to_anchor=(0.4, 1), ncol=3, title=None, frameon=True)
		for i in range(3):
			g.axes[0,i].axvline(5 if i == 0 else 10 if i ==1 else 20, ls='--')
		# for i in range(3):
		# 	for j in range(3):
		# 		g.axes[i,j].axvline(5 if i == 0 else 10 if i ==1 else 20, ls='--')
		# 		g.axes[i,j].tick_params(axis='x', which='minor', bottom=False)
		# 		g.axes[i, j].tick_params(axis='y', which='minor', bottom=False)
		plt.savefig(os.path.join(args.output_file + '.jpg'), bbox_inches="tight")
		# plt.show()
		plt.close('all')


def analyze_Descriptive():
	data = defaultdict(list)
	subjects = ['Love', 'Cats', 'Running']
	requirements = ['short', 'brief', 'concise', 'long', 'detailed', 'in-depth']
	for id in range(36):
		if id < 18:
			data['template'].extend([1] * 10)
			data['Subject'].extend([subjects[id // 6]] * 10)
			data['requirement'].extend([requirements[id % 6]] * 10)
		else:
			data['template'].extend([2] * 10)
			data['Subject'].extend([subjects[id // 6 - 3]] * 10)
			data['requirement'].extend([requirements[id % 6]] * 10)
		raw_path = os.path.join(args.input_file, f'{id}_raw.json')
		with open(raw_path, "r") as f:
			outputs = json.load(f)
			for index in range(10):
				text = outputs['choices'][index]['text']
				if args.verbose:
					print(text)
				cnt = sentence_counter(text)
				data['count'].append(cnt)
				print(cnt, end='\t')
			print()

	df = pd.DataFrame.from_dict(data)
	if args.verbose:
		print(df)
	point_color = sns.color_palette('tab10')
	c = sns.color_palette('Paired')
	g = sns.boxplot(x='count', y='requirement', whis=np.inf, palette='pastel', data=df)
	sns.stripplot(x='count', y='requirement', hue='Subject', linewidth=1, palette=point_color, dodge=True, marker='o', jitter=1, data=df[df['template'] == 1], ax=g)
	sns.stripplot(x='count', y='requirement', hue='Subject', linewidth=1, palette=point_color, dodge=True, marker='^', jitter=1, data=df[df['template'] == 2],
	              ax=g)
	g.set_xlabel('Number of sentences')
	g.set_ylabel('')
	handles, labels = g.get_legend_handles_labels()
	g.legend(handles[:3], labels[:3])
	plt.savefig(os.path.join(args.output_file + '.jpg'), bbox_inches="tight")
	# plt.show()
	plt.close('all')


def getCSV():
	reference = pd.read_csv("MTurk/Reference.csv")
	# if args.verbose:
	# 	print(reference['Term'])
	aspects = ("Writing Style", "Tone", "Mood", "Characterization", "Pacing", "Plot", "Genre")
	data = []
	data_short = []
	data_long = []
	score = []
	# if args.settings is not None:
	# 	pass
	# setting = 'Basic'
	for i, setting in enumerate(args.settings):
		# dir_name = os.path.join("Style", "Literary Style" if i < 3 else "Story Generation", aspect)
		dir_name = os.path.join(args.input_file, setting)
		# dir_name = args.input_file
		files = glob.glob(os.path.join(dir_name, '*.json'))
		files.sort(key=lambda s: int(s.split('/')[-1].split('_')[0]))
		if args.verbose:
			print(files)
		for file in files:
			id = int(file.split('/')[-1].split('_')[0])
			# if id < 1000:
			# 	continue
			if args.beta:
				if not ((i < 3 and id in args.l_id) or (i > 2 and id in args.s_id)):
					continue
			with open(file, 'r') as f:
				outputs = json.load(f)

			if args.no_echo:
				with open(file.replace('_raw.json', '_prompt.txt'), 'r') as f:
					prompt = f.readlines()[-1].strip('\n')
			else:
				prompt = outputs['choices'][0]['text'].split('\n')[0]
			if args.verbose:
				print(prompt)
			# get definition and term
			term = None
			definition = None
			for word in prompt.split(' '):
				if word == 'the':
					continue
				if word[-1] == ':':
					word = word[:-1]
				if args.verbose:
					print(word)
				res = reference['Term'].str.contains(rf'(^|\s){word}($|\s)')
				if res.any():
					row = reference.loc[res]
					if len(row.index) > 1:
						if "sad passage" in prompt or "sad tone" in prompt:
							row = row.loc[row['Aspect'] == 'tone']
						elif "sad mood" in prompt or "feel sad" in prompt:
							row = row.loc[row['Aspect'] == 'mood']
						else:
							raise ValueError(f'more than one matches! {row}')
					definition = f'{row["Term"].to_string(index=False)} {row["Aspect"].to_string(index=False)}: {row["Definition"].to_string(index=False)}'
					term = row["Term"].to_string(index=False)
					if args.verbose:
						print(definition)
					break

			prompt = prompt.replace(term, f'<b>{term}</b>')
			aspect = aspects[id]
			for index in range(10 if not args.beta else 3):
				text = outputs['choices'][index]['text']
				if args.no_echo:
					response = text[2:]
				else:
					response = text[len(prompt) - 7 + 2:]
				if args.verbose:
					print(response)
				word_len = len(response)
				response = response.replace('\n', '<br>')

				if [prompt, response, aspect, definition, setting] in data:
					print([prompt, response, aspect, definition, setting])
				data.append([prompt, response, aspect, definition, setting])

				if word_len <= 600:
					data_short.append([prompt, response, aspect, definition, setting])
				else:
					data_long.append([prompt, response, aspect, definition, setting])
				score.append(word_len)

	if args.verbose:
		print(pd.Series(score).describe())

	data = pd.DataFrame(data, columns=["prompt", "response", "aspect", "definition", "setting"])
	data.to_csv(f'{args.output_file}.csv', index=False)
	data_short = pd.DataFrame(data_short, columns=["prompt", "response", "aspect", "definition", "setting"])
	data_short.to_csv(f'{args.output_file}_short.csv', index=False)
	data_long = pd.DataFrame(data_long, columns=["prompt", "response", "aspect", "definition", "setting"])
	data_long.to_csv(f'{args.output_file}_long.csv', index=False)


def getCSV_CSV():
	reference = pd.read_csv("MTurk/Reference.csv")
	# if args.verbose:
	# 	print(reference['Term'])
	aspects = ("Writing Style", "Tone", "Mood", "Characterization", "Pacing", "Plot", "Genre")
	data = []
	data_short = []
	data_long = []
	score = []
	# if args.settings is not None:
	# 	pass
	# setting = 'Basic'
	for i, setting in enumerate(args.settings):
		dir_name = os.path.join(args.input_file, setting)
		with open(os.path.join(args.input_file, "prompts.txt"), "r") as f:
			prompts = f.readlines()
		for id, aspect in enumerate(aspects):
			prompt = prompts[id].replace('\n', '')
			file_name = glob.glob(os.path.join(dir_name, f'{id}*.csv'))
			df = pd.read_csv(file_name[0], header=None)
			if args.verbose:
				print(prompt)
				print(file_name)
				print(df)
			# get definition and term
			term = None
			definition = None
			for word in prompt.split(' '):
				if word == 'the':
					continue
				if word[-1] == ':':
					word = word[:-1]
				if args.verbose:
					print(word)
				res = reference['Term'].str.contains(rf'(^|\s){word}($|\s)')
				if res.any():
					row = reference.loc[res]
					if len(row.index) > 1:
						if "sad passage" in prompt or "sad tone" in prompt:
							row = row.loc[row['Aspect'] == 'tone']
						elif "sad mood" in prompt or "feel sad" in prompt:
							row = row.loc[row['Aspect'] == 'mood']
						else:
							raise ValueError(f'more than one matches! {row}')
					definition = f'{row["Term"].to_string(index=False)} {row["Aspect"].to_string(index=False)}: {row["Definition"].to_string(index=False)}'
					term = row["Term"].to_string(index=False)
					if args.verbose:
						print(definition)
					break

			prompt = prompt.replace(term, f'<b>{term}</b>')

			for index in df.index:
				# if df[1][index] == 0:
				# 	continue
				response = df[0][index]
				if args.verbose:
					print(response)
				word_len = len(response)
				response = response.replace('\n', '<br>')

				if [prompt, response, aspect, definition, setting] in data:
					print([prompt, response, aspect, definition, setting])
				data.append([prompt, response, aspect, definition, setting])

				if word_len <= 600:
					data_short.append([prompt, response, aspect, definition, setting])
				else:
					data_long.append([prompt, response, aspect, definition, setting])
				score.append(word_len)

	if args.verbose:
		print(pd.Series(score).describe())

	data = pd.DataFrame(data, columns=["prompt", "response", "aspect", "definition", "setting"])
	data.to_csv(f'{args.output_file}.csv', index=False)
	# data_short = pd.DataFrame(data_short, columns=["prompt", "response", "aspect", "definition", "setting"])
	# data_short.to_csv(f'{args.output_file}_short.csv', index=False)
	# data_long = pd.DataFrame(data_long, columns=["prompt", "response", "aspect", "definition", "setting"])
	# data_long.to_csv(f'{args.output_file}_long.csv', index=False)


def process_raw_result():
	results = pd.read_csv(args.input_file)
	if args.input_file_2 is not None:
		results2 = pd.read_csv(args.input_file_2)
		results = pd.concat([results, results2])
	results = results.loc[results['AssignmentStatus'] != 'Rejected']
	results.drop(
		columns=['HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments',
		         'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration',
		         'NumberOfSimilarHITs', 'LifetimeInSeconds', 'AssignmentStatus', 'AcceptTime', 'SubmitTime',
		         'AutoApprovalTime', 'ApprovalTime', 'RejectionTime', 'RequesterFeedback', 'Last30DaysApprovalRate',
		         'Last7DaysApprovalRate'], inplace=True)
	results.reset_index(drop=True, inplace=True)
	print(results)

	refer = pd.read_csv(args.reference_file)
	clean_result = pd.DataFrame()
	cnt = 0
	for i, row in enumerate(zip(refer['prompt'], refer['response'], refer['aspect'])):
		prompt_id = i // 10
		response_id = i % 10
		result_rows = results.loc[results['Input.response'] == row[1]]
		if args.verbose:
			print(row[1])
			print(result_rows)
		if len(result_rows.index) > 3:
			print(result_rows)
			j = result_rows.index[cnt * 3]
			result_rows = result_rows.loc[j: j + 3, :]
			# print(result_rows)
			cnt += 1
		assert len(result_rows.index) == 3
		clean_result = pd.concat([clean_result, result_rows], ignore_index=True)

	print(clean_result)
	results = clean_result
	score = []
	for i, row in enumerate(zip (results['Answer.1Strongly Disagree.-2'], results['Answer.1Somewhat Disagree.-1'], results['Answer.1Neutral.0'], results['Answer.1Somewhat Agree.1'], results['Answer.1Strongly Agree.2'])):
		if row[0]:
			score.append(-2)
		elif row[1]:
			score.append(-1)
		elif row[2]:
			score.append(0)
		elif row[3]:
			score.append(1)
		elif row[4]:
			score.append(2)
		else:
			raise ValueError(f"No Score Level is chosen for row {i}")
	results['Score'] = score
	results['3Score'] = [1 if x >= 1 else -1 if x <= -1 else 0 for x in score]
	results.drop(columns=['Answer.1Strongly Disagree.-2', 'Answer.1Somewhat Disagree.-1', 'Answer.1Neutral.0', 'Answer.1Somewhat Agree.1', 'Answer.1Strongly Agree.2'], inplace=True)
	print(results)

	# print("krippendorff_alpha_for Score for t0.4: ",
	#       simpledorff.calculate_krippendorffs_alpha_for_df(results.iloc[420:], experiment_col='HITId', annotator_col='WorkerId',
	#                                                        class_col='Score', metric_fn=interval_metric))
	print("krippendorff_alpha_for Score: ", simpledorff.calculate_krippendorffs_alpha_for_df(results, experiment_col='HITId', annotator_col='WorkerId', class_col='Score', metric_fn=interval_metric))
	print("krippendorff_alpha_for 3-scale Score: ",
	      simpledorff.calculate_krippendorffs_alpha_for_df(results, experiment_col='HITId', annotator_col='WorkerId',
	                                                       class_col='3Score', metric_fn=interval_metric))
	if not args.no_hardness:
		results['2Difficulty'] = [1 if x <=5  else 2 for x in results['Answer.difficulty'].tolist()]
		print("krippendorff_alpha_for difficulty: ",
		      simpledorff.calculate_krippendorffs_alpha_for_df(results, experiment_col='HITId', annotator_col='WorkerId',
		                                                       class_col='Answer.difficulty', metric_fn=interval_metric))
		print("krippendorff_alpha_for 2-scale difficulty: ",
		      simpledorff.calculate_krippendorffs_alpha_for_df(results, experiment_col='HITId', annotator_col='WorkerId',
		                                                       class_col='2Difficulty', metric_fn=interval_metric))
	results.to_csv(args.output_file)


def analyze_Style():
	results = pd.read_csv(args.input_file)
	datas = pd.read_csv(args.reference_file)
	score = []
	data = defaultdict(list)
	difficulty = defaultdict(list)
	cnt = 0
	failure = 0
	degenerate = 0
	all_d = defaultdict(float)
	all_s = defaultdict(float)
	for i, row in enumerate(zip (datas['prompt'], datas['response'], datas['aspect'], datas['setting'])):
		prompt_id = i // 10
		response_id = i % 10
		result_rows = results.loc[results['Input.response'] == row[1]]
		if args.verbose:
			print(row)
			print(result_rows)
		if len(result_rows.index) > 3:
			# print(result_rows)
			j = result_rows.index[cnt * 3]
			result_rows = result_rows.loc[j : j + 3, :]
			# print(result_rows)
			cnt += 1

		if len(result_rows.index) == 0:
			s = [-2, -2, -2]
			degenerate += 1
			all_d[row[3]] += 1
		else:
			s = result_rows['Score'].tolist()
		# assert len(result_rows.index) == 3
		if not args.no_hardness:
			for roww in zip(result_rows['WorkerId'], result_rows['Answer.difficulty']):
				difficulty[roww[0]].append(roww[1])
		if np.median(s) <= 0:
			failure += 1
		score.append(np.mean(s))
		all_s[row[3]] += score[-1]
		if response_id == 9:
			# print(row)
			d = []
			if not args.no_hardness:
			# clean same annotator's inconsistent difficulty score to the same prompt
				for workerid, s in difficulty.items():
					mode = scipy.stats.mode(s)
					if mode[1] > 1:
						assert len(mode[0]) == 1
						d.extend(mode[0])
					else: # no mode
						d.append(np.median(s))
					if args.verbose:
						print(workerid, s)
						print(mode)
						print(d[-1])
				print(f'{np.median(d):.2f}\t{np.mean(d):.2f}\t{failure * 0.1:.2f}\t{np.mean(score):.2f}\t{np.std(score):.2f}',end="")
				data['Difficulty'].append(np.mean(d))
				# print(f'{np.std(score):.2f}',end="")
			else:
				# print(f'{failure * 0.1:.2f}\t{np.mean(score):.2f}\t{np.std(score):.2f}', end="")
				pass
			print(f'{degenerate * 10:}%', end="")
			# for s in score:
			# 	print(f'\t{s:.2f}', end="")
			print()
			data['id'].append(prompt_id)
			data['Aspect'].append(row[2])
			data['Score'].append(np.mean(score))
			score = []
			difficulty = defaultdict(list)
			cnt = 0
			failure = 0
			degenerate = 0
	for k, v in all_d.items():
		print(f'{k}: {v / 70 * 100: .2f}%, {all_s[k] / 70:.2f}')
	if args.no_hardness:
		return
	print(f"spearman's correlation for score and difficulty is {scipy.stats.spearmanr(data['Score'], data['Difficulty'])}" )
	df = pd.DataFrame.from_dict(data)
	g = sns.relplot(x='Difficulty', y='Score', hue='Aspect', palette='pastel', data=df)
	ax = g.axes[0, 0]
	# plt.setp(g._legend.get_texts(), fontsize='10')
	# plt.setp(g._legend.get_title(), fontsize='10')
	plt.savefig(os.path.join(args.output_file + '.jpg'), bbox_inches="tight")
	plt.show()
	plt.close('all')


def analyze_Sensitivity():
	df = pd.read_csv(args.input_file)
	df['Prompts'] = df['Prompts'].apply(lambda s: s.replace('\\n', '\n'))
	df['Setting'] = df['Setting'].apply(lambda s: s.replace('\\n', '\n'))
	Settings = df['Setting'].unique().tolist()
	# print(Settings)
	# data = defaultdict(list)
	# for _, row in df.iterrows():
	# 	data['prompt_id'].extend([row['prompt_id']] * 10)
	# 	data['Setting'].extend([row['Setting']] * 10)
	# 	for id in range(10):
	# 		data['Score'].append(row[str(id)])
	# df = pd.DataFrame.from_dict(data)
	print(df)
	palette = sns.color_palette('pastel', n_colors=len(Settings))
	# palette = [palette[3]] + palette[:3] + palette[4:]
	# print(palette)
	ax = sns.barplot(x='Setting', y='failure_ratio' if args.failure_ratio else 'score_mean', ci=None, palette=palette, data=df)
	ax.set_xticklabels([])
	# ax.set_ylim(0.6,)
	ax.set_ylabel('Failure Ratio' if args.failure_ratio else 'Score')
	handles = [Patch(color=c, label=l) for c, l in zip(palette, Settings)]
	# print(ax.get_xticklabels())
	# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, multialignment='center', wrap=True)
	# ax.bar_label(ax.containers[0], labels=[f'{x:,.2f}' for x in ax.containers[0].datavalues])
	ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1), ncol=2, title=None, frameon=True, fontsize=15.5)
	plt.savefig(os.path.join(args.output_file + '.jpg'), bbox_inches="tight")
	# plt.show()
	plt.close('all')


if __name__ == "__main__":
	pd.set_option('display.max_colwidth', None)
	sns.set(style="whitegrid", font_scale=1.6)
	if args.analyze_Numerical:
		analyze_Numerical()
	if args.analyze_Descriptive:
		analyze_Descriptive()
	if args.getCSV:
		getCSV()
	if args.getCSV_CSV:
		getCSV_CSV()
	if args.process_raw_result:
		process_raw_result()
	if args.analyze_Style:
		analyze_Style()
	if args.analyze_Sensitivity:
		analyze_Sensitivity()
	if args.create_Numerical_files:
		create_Numerical_files()


