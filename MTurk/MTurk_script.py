import csv
import os
import boto3
from dotenv import load_dotenv
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--no_sandbox", action='store_true')
parser.add_argument("--sanity_test", action='store_true')
parser.add_argument("--create_qualification", action='store_true')
parser.add_argument("--notify_workers", action='store_true')
args = parser.parse_args()

load_dotenv()

KEY_ID = os.getenv("KEY_ID")
KEY_SECRET = os.getenv("KEY_SECRET")

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
MTURK = 'https://mturk-requester.us-east-1.amazonaws.com'
mturk = boto3.client('mturk',
   aws_access_key_id = KEY_ID,
   aws_secret_access_key = KEY_SECRET,
   region_name='us-east-1',
   endpoint_url = MTURK if args.no_sandbox else MTURK_SANDBOX
)

def sanity_test():
    print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my Sandbox account")

def create_hit():
    HIT_LAYOUT_ID = "3S3TY9RPQUOPL41THB4VIBOCM1RAFH"

    csv_path = "challenging.csv"
    params = pd.read_csv(csv_path).to_dict('records')
    params = params[0]
    params_formatted = []
    for item in params.items():
        params_formatted.append({'Name': item[0], 'Value': item[1]})

    lifetime_days = 7
    autoapproval_days = 3

    new_hit = mturk.create_hit(
        Title = 'Is this Tweet happy, angry, excited, scared, annoyed or upset?',
        Description = 'Read this tweet and type out one word to describe the emotion of the person posting it: happy, angry, scared, annoyed or upset',
        Keywords = 'text, quick, labeling',
        Reward = '0.12',
        MaxAssignments = 3,
        LifetimeInSeconds = lifetime_days * 86400,
        AssignmentDurationInSeconds = 600,
        AutoApprovalDelayInSeconds = autoapproval_days * 86400,
        HITLayoutId = HIT_LAYOUT_ID,
        HITLayoutParameters = params_formatted
    )

    print("A new HIT has been created. You can preview it here:")
    print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
    print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")

    # Remember to modify the URL above when you're publishing
    # HITs to the live marketplace.
    # Use: https://worker.mturk.com/mturk/preview?groupId=

def create_qualification_test():
    questions = open('MTurk/q_q.xml', mode='r').read()
    answers = open('MTurk/q_a.xml', mode='r').read()
    qual_response = mturk.create_qualification_type(
        Name='Text Style Understanding',
        Keywords='test, qualification, style, writing, language, text',
        Description='This is a qualification test for accessing our official HIT',
        QualificationTypeStatus='Active',
        Test=questions,
        AnswerKey=answers,
        TestDurationInSeconds=900)

def notify_workers():
    respond = mturk.notify_workers(
        Subject="Please correct your annotations on Q1 for these 5 HITs or we'll have to reject them",
        MessageText="Hi, worker A3I1JDXJY9TIZX. Thanks for working on our HITs, but it seems you are missing Q1 by only giving the default 1 as its answer, which is clearly inappropriate. Please read the instruction carefully and ground your answer with the examples given. Please try redo the Q1 on these 5 HITs A.S.A.P, or we'll have to reject them. HIT IDs: 34F34TZU88GQJ6P48BVZ452E2ZOJ25, 3PIOQ99R8A3VM8PR6TXX3VENUADNUW, 3LG268AV4KNZCAKX90Z98XXU9J8ERU, 3URJ6VVYV14ENVVOS26S5GGYL7UO4P, 3G57RS03ITMIC7AJJ9R53VJ9KYU253",
        WorkerIds=["A3I1JDXJY9TIZX"]
    )
    print(respond)


if args.sanity_test:
    sanity_test()
elif args.create_qualification:
    create_qualification_test()
elif args.notify_workers:
    notify_workers()
else:
    raise ValueError("Please denote the action desired.")