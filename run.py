import os
import openai
import json
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--prompt_file", default=None, type=str)
parser.add_argument("--prompt_list_file", default=None, type=str)
parser.add_argument("--prompt_csv_file", default=None, type=str)
parser.add_argument("--prompt_id", nargs='+', default=None, type=int)
parser.add_argument("--no_prompt_id", nargs='+', default=None, type=int)
parser.add_argument("--model", default="text-davinci-003")
parser.add_argument("--t", default=0.7, type=float)
parser.add_argument("--top_p", default=1.0, type=float)
parser.add_argument("--max_tokens", default=256, type=int)
parser.add_argument("--n", default=1, type=int)
parser.add_argument("--no_echo", action='store_true', help="whether not to include prompt in the outputs")
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--output_file", required=True)
args = parser.parse_args()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_response(prompt, output_file):
	response = openai.Completion.create(
		model = args.model,
		prompt = prompt,
		temperature = args.t,
		top_p = args.top_p,
		max_tokens = args.max_tokens,
		n = args.n,
		logprobs = 5 if args.verbose else None,
		echo = not args.no_echo)
	with open(f"{output_file}_raw.json", 'a') as f:
		f.write(json.dumps(response, indent=4))
		f.write('\n')
	# print(response)
	# response.to_dict_recursive()
	text = [response['choices'][i]['text'] for i in range(args.n)]
	return text

if __name__ == "__main__":
	if args.prompt_list_file is not None:
		if not os.path.exists(args.output_file):
			os.makedirs(args.output_file)

		with open(args.prompt_list_file, 'r') as prompt_file:
			prompts = prompt_file.readlines()
			for id, prompt in enumerate(prompts):
				if args.prompt_id is not None:
					if id not in args.prompt_id:
						continue
				elif args.no_prompt_id is not None:
					if id in args.no_prompt_id:
						continue
				prompt = prompt.strip('\n')
				print(prompt)
				words = prompt.split()
				output_file = os.path.join(args.output_file, f"{str(id)}")
				response = get_response(prompt, output_file)
				with open(f"{output_file}.txt", 'a') as f:
					for i, s in enumerate(response):
						f.write(f"response {i}:\n")
						f.write(s)
						f.write('\n---------------------------\n')
	elif args.prompt_csv_file is not None:
		df = pd.read_csv(args.prompt_csv_file)
		for _, row in df.iterrows():
			prompt = row['prompt']
			mitigation = row['mitigation']
			id = row['prompt_id']
			print(prompt)
			if id != 9:
				continue
			out_dir = os.path.join(args.output_file, mitigation)
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)
			output_file = os.path.join(args.output_file, mitigation, f"{str(id)}")
			with open(f"{output_file}_prompt.txt", 'w') as f:
				f.write(prompt)
			response = get_response(prompt, output_file)
			with open(f"{output_file}.txt", 'a') as f:
				for i, s in enumerate(response):
					f.write(f"response {i}:\n")
					f.write(s)
					f.write('\n---------------------------\n')


	elif args.prompt_file is not None:
		with open(args.prompt_file, 'r') as prompt_file:
			prompt = ''.join(prompt_file.readlines())
			print(prompt)
			response = get_response(prompt, args.output_file)
			with open(f"{args.output_file}.txt", 'a') as f:
				for i, s in enumerate(response):
					f.write(f"response {i}:\n")
					if args.no_echo:
						f.write(prompt)
					f.write(s)
					f.write('\n---------------------------\n')
	elif args.prompt is not None:
		response = get_response(args.prompt, args.output_file)
		with open(f"{args.output_file}.txt", 'a') as f:
			for i, s in enumerate(response):
				f.write(f"response {i}:\n")
				f.write(s)
				f.write('\n---------------------------\n')
	else:
		raise RuntimeError("Please provide prompt or prompt_file or prompt_list_file!")
