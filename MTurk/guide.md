## 1. Create the qualification type

get your MTURK_KEY and store it in the '.env' file in the same directory with 'MTurk_cciprt.py'

run
`python3 MTurk_script.py --no_sandbox --create_qualification`

Then you can see a qualification type called "Text Style Understanding" here `https://requester.mturk.com/qualification_types`


## 2. Create the project

https://requester.mturk.com/create/projects/new

note: you should create two different project with different reward (for short and long responses respectively)

### Enter properties
Project Name: Writing quality

Title: Annotate the response quality based on a prompt

Description: Annotate the quality of a written response with regard to the style requested in the prompt.

Keywords: writing, prompt, quality, rating, style

Reward per assignment: $0.11 for short, $0.15 for long

Number of assignments per task: 3

Time allotted per assignment: 10 Minutes

Task expires in: 7 Days

Auto-approve and pay Workers in	: 5 Days

Require that Workers be Masters to do your tasks: No

Specify any additional qualifications Workers must meet to work on your tasks: 

`HIT Approval Rate (%) for all Requesters' HITs greater than or equal to 98 `

`Location is one of AU, CA, GB, US`

`Text Style Understanding greater than or equal to 5`

Task Visibility: Private

### Design Layout

copy the code in the _layout.html_ here.

### Prewiew and Finish

If everything goes fine, it's finished.

## 3. Publish the batch

https://requester.mturk.com/create/projects

publish the batch for project (short and long), upload the csv file _data_short.csv_ and _data_long.csv_ respectively.

check the HITs and prices, if everything is ok, click "publish" and everything is done.

