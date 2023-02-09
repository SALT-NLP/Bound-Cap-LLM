import json
import os

import requests
import argparse

from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--prompt_file", default=None, type=str)
parser.add_argument("--prompt_list_file", default=None, type=str)
parser.add_argument("--prompt_id", nargs='+', default=None, type=int)
parser.add_argument("--no_prompt_id", nargs='+', default=None, type=int)
parser.add_argument("--model", default="text-davinci-002")
parser.add_argument("--t", default=0.7, type=float)
parser.add_argument("--top_p", default=1.0, type=float)
parser.add_argument("--max_tokens", default=250, type=int)
parser.add_argument("--n", default=1, type=int)
parser.add_argument("--no_echo", action='store_true', help="whether not to include prompt in the outputs")
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--output_file", required=True)
args = parser.parse_args()

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"


def get_response(prompt, output_file, i):
	payload = {
		"inputs": prompt,
		"parameters":{
			"max_new_tokens": args.max_tokens,
			"num_return_sequences": 1,
			"do_sample": True,
			"temperature": args.t,
			"return_full_text": not args.no_echo
		},
		"options":{
			"use_cache": False
		}
	}
	data = json.dumps(payload)
	response = requests.request("POST", API_URL, headers=headers, data=data)
	result = json.loads(response.content.decode("utf-8"))
	with open(f"{output_file}.json", 'a') as f:
		f.write(json.dumps(result, indent=4))
	with open(f"{output_file}.txt", 'a') as f:
		f.write(f"response {i}:\n")
		f.write(result[0]['generated_text'])
		f.write('\n---------------------------\n')


if __name__ == "__main__":
	if args.prompt_list_file is not None:
		if not os.path.exists(args.output_file):
			os.makedirs(args.output_file)

		with open(args.prompt_list_file, 'r') as prompt_file:
			prompts = prompt_file.readlines()
			for i in range(args.n):
				for id, prompt in enumerate(prompts):
					if args.prompt_id is not None:
						if id not in args.prompt_id:
							continue
					elif args.no_prompt_id is not None:
						if id in args.no_prompt_id:
							continue
					prompt = prompt.strip('\n')
					print(prompt)
					output_file = os.path.join(args.output_file, f"{str(id)}_raw")
					get_response(prompt, output_file, i)
	elif args.prompt_file is not None:
		with open(args.prompt_file, 'r') as prompt_file:
			prompt = ''.join(prompt_file.readlines())
			print(prompt)
			for i in range(args.n):
				get_response(prompt, args.output_file, i)

	elif args.prompt is not None:
		for i in range(args.n):
			get_response(args.prompt, args.output_file, i)
	else:
		raise RuntimeError("Please provide prompt or prompt_file or prompt_list_file!")
