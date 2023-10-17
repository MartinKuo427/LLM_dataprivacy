import subprocess

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    AutoModelForQuestionAnswering,
)

import boto3
import transformers


# from transformers import , AutoTokenizer, pipeline

import torch
import numpy as np
from transformers import LongformerTokenizer, LongformerForMultipleChoice

import os

import heapq

from datasets import load_dataset

import pandas as pd

import datasets

transformers.logging.set_verbosity_error()


class Client:
    def __init__(self):
        self.data = []
        self.NAME = "" # NAME not in PII_collect_list
        self.CATEGORY = "" # Category
        self.ADDRESS = ["-"]
        self.AGE = ["-"]
        self.CREDIT_DEBIT_CVV = ["-"]
        self.CREDIT_DEBIT_EXPIRY = ["-"]
        self.CREDIT_DEBIT_NUMBER = ["-"]
        self.DRIVER_ID = ["-"]
        self.PHONE = ["-"]
        self.PASSWORD = ["-"]
        self.BANK_ACCOUNT_NUMBER = ["-"]
        self.PASSPORT_NUMBER = ["-"]
        self.SSN = ["-"]

        self.not_empty_ADDRESS = False
        self.not_empty_AGE = False
        self.not_empty_CREDIT_DEBIT_CVV = False
        self.not_empty_CREDIT_DEBIT_EXPIRY = False
        self.not_empty_CREDIT_DEBIT_NUMBER = False
        self.not_empty_DRIVER_ID = False
        self.not_empty_PHONE = False
        self.not_empty_PASSWORD = False
        self.not_empty_BANK_ACCOUNT_NUMBER = False
        self.not_empty_PASSPORT_NUMBER = False
        self.not_empty_SSN = False

    def add_PII(self, PII_text, PII_type):
        if (PII_type == "NAME"):
            self.NAME = PII_text
        elif (PII_type == "CATEGORY"):
            self.CATEGORY = PII_text
        elif (PII_type == "ADDRESS"):
            if (self.not_empty_ADDRESS):
                self.ADDRESS.append(PII_text)
            else:
                self.ADDRESS[0] = PII_text
                self.not_empty_ADDRESS = True
        elif (PII_type == "AGE"):
            if (self.not_empty_AGE):
                self.AGE.append(PII_text)
            else:
                self.AGE[0] = PII_text
                self.not_empty_AGE = True
        elif (PII_type == "CREDIT_DEBIT_CVV"):
            if (self.not_empty_CREDIT_DEBIT_CVV):
                self.CREDIT_DEBIT_CVV.append(PII_text)
            else:
                self.CREDIT_DEBIT_CVV[0] = PII_text
                self.not_empty_CREDIT_DEBIT_CVV = True
        elif (PII_type == "CREDIT_DEBIT_EXPIRY"):
            if (self.not_empty_CREDIT_DEBIT_EXPIRY):
                self.CREDIT_DEBIT_EXPIRY.append(PII_text)
            else:
                self.CREDIT_DEBIT_EXPIRY[0] = PII_text
                self.not_empty_CREDIT_DEBIT_EXPIRY = True
        elif (PII_type == "CREDIT_DEBIT_NUMBER"):
            if (self.not_empty_CREDIT_DEBIT_NUMBER):
                self.CREDIT_DEBIT_NUMBER.append(PII_text)
            else:
                self.CREDIT_DEBIT_NUMBER[0] = PII_text
                self.not_empty_CREDIT_DEBIT_NUMBER = True
        elif (PII_type == "DRIVER_ID"):
            if (self.not_empty_DRIVER_ID):
                self.DRIVER_ID.append(PII_text)
            else:
                self.DRIVER_ID[0] = PII_text
                self.not_empty_DRIVER_ID = True
        elif (PII_type == "PHONE"):
            if (self.not_empty_PHONE):
                self.PHONE.append(PII_text)
            else:
                self.PHONE[0] = PII_text
                self.not_empty_PHONE = True
        elif (PII_type == "PASSWORD"):
            if (self.not_empty_PASSWORD):
                self.PASSWORD.append(PII_text)
            else:
                self.PASSWORD[0] = PII_text
                self.not_empty_PASSWORD = True
        elif (PII_type == "BANK_ACCOUNT_NUMBER"):
            if (self.not_empty_BANK_ACCOUNT_NUMBER):
                self.BANK_ACCOUNT_NUMBER.append(PII_text)
            else:
                self.BANK_ACCOUNT_NUMBER[0] = PII_text
                self.not_empty_BANK_ACCOUNT_NUMBER = True
        elif (PII_type == "PASSPORT_NUMBER"):
            if (self.not_empty_PASSPORT_NUMBER):
                self.PASSPORT_NUMBER.append(PII_text)
            else:
                self.PASSPORT_NUMBER[0] = PII_text
                self.not_empty_PASSPORT_NUMBER = True
        elif (PII_type == "SSN"):
            if (self.not_empty_SSN):
                self.SSN.append(PII_text)
            else:
                self.SSN[0] = PII_text
                self.not_empty_SSN = True
        else:
            print("None type !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    #def add(self, x):
    #    self.data.append(x)
    #def addtwice(self, x):
    #    self.add(x)


def prepare_answering_input(
        tokenizer, # longformer_tokenizer
        question,  # str
        options,   # List[str]
        context,   # str
        max_seq_length=512,# 4096
    ):
    c_plus_q   = context + ' ' + tokenizer.bos_token + ' ' + question
    c_plus_q_4 = [c_plus_q] * len(options)
    tokenized_examples = tokenizer(
        c_plus_q_4, options,
        max_length=max_seq_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )# .to("cuda")
    input_ids = tokenized_examples['input_ids'].unsqueeze(0)
    attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)

    # martinc
    # print("prepare_answering_input input_ids---------------")
    # print(input_ids)

    example_encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    return example_encoded


def build_huggingace_dataset(Client_class_list):

    NAME_list = []# NAME not in PII_collect_list
    CATEGORY_list = []# Category not in PII_collect_list
    ADDRESS_list = []
    AGE_list = []
    CREDIT_DEBIT_CVV_list = []
    CREDIT_DEBIT_EXPIRY_list = []
    CREDIT_DEBIT_NUMBER_list = []
    DRIVER_ID_list = []
    # EMAIL_list = []# too many email, email not in PII_collect_list
    PHONE_list = []
    PASSWORD_list = []
    BANK_ACCOUNT_NUMBER_list = []
    PASSPORT_NUMBER_list = []
    SSN_list = []

    print("build_huggingace_dataset len(Client_class_list):", len(Client_class_list))

    # transfer data in Client_class_list to DataFrame
    for index in range(len(Client_class_list)):
        NAME_list.append(Client_class_list[index].NAME)
        CATEGORY_list.append(Client_class_list[index].CATEGORY)
        ADDRESS_list.append(Client_class_list[index].ADDRESS)
        AGE_list.append(Client_class_list[index].AGE)
        CREDIT_DEBIT_CVV_list.append(Client_class_list[index].CREDIT_DEBIT_CVV)
        CREDIT_DEBIT_EXPIRY_list.append(Client_class_list[index].CREDIT_DEBIT_EXPIRY)
        CREDIT_DEBIT_NUMBER_list.append(Client_class_list[index].CREDIT_DEBIT_NUMBER)
        DRIVER_ID_list.append(Client_class_list[index].DRIVER_ID)
        PHONE_list.append(Client_class_list[index].PHONE)
        PASSWORD_list.append(Client_class_list[index].PASSWORD)
        BANK_ACCOUNT_NUMBER_list.append(Client_class_list[index].BANK_ACCOUNT_NUMBER)
        PASSPORT_NUMBER_list.append(Client_class_list[index].PASSPORT_NUMBER)
        SSN_list.append(Client_class_list[index].SSN)



    # Create a DataFrame
    df = pd.DataFrame({
        'NAME': NAME_list,
        'CATEGORY': CATEGORY_list,
        'ADDRESS': ADDRESS_list,
        'AGE': AGE_list,
        'CREDIT_DEBIT_CVV': CREDIT_DEBIT_CVV_list,
        'CREDIT_DEBIT_EXPIRY': CREDIT_DEBIT_EXPIRY_list,
        'CREDIT_DEBIT_NUMBER': CREDIT_DEBIT_NUMBER_list,
        'DRIVER_ID': DRIVER_ID_list,
        'PHONE': PHONE_list,
        'PASSWORD': PASSWORD_list,
        'BANK_ACCOUNT_NUMBER': BANK_ACCOUNT_NUMBER_list,
        'PASSPORT_NUMBER': PASSPORT_NUMBER_list,
        'SSN': SSN_list
    }, columns=['NAME', 'CATEGORY', 'ADDRESS', 'AGE', 'CREDIT_DEBIT_CVV', 'CREDIT_DEBIT_EXPIRY', 'CREDIT_DEBIT_NUMBER', 'DRIVER_ID', 'PHONE', 'PASSWORD', 'BANK_ACCOUNT_NUMBER', 'PASSPORT_NUMBER', 'SSN'])

    df_data = datasets.Dataset.from_pandas(df)

    # load_repo = "MartinKu/test_privacy"
    # df_data.push_to_hub(load_repo)

    filepath = "./test_privacy_21.csv"
    df.to_csv(filepath)


# PII_collect_list = ['ADDRESS', 'AGE', 'CREDIT_DEBIT_CVV', 'CREDIT_DEBIT_EXPIRY', 'CREDIT_DEBIT_NUMBER', 'DRIVER_ID', 'EMAIL', 'PHONE', 'PASSWORD', 'BANK_ACCOUNT_NUMBER', 'PASSPORT_NUMBER', 'SSN']

PII_collect_list = ['ADDRESS', 'AGE', 'CREDIT_DEBIT_CVV', 'CREDIT_DEBIT_EXPIRY', 'CREDIT_DEBIT_NUMBER', 'DRIVER_ID', 'PHONE', 'PASSWORD', 'BANK_ACCOUNT_NUMBER', 'PASSPORT_NUMBER', 'SSN']

Entities_collect_list = ['PERSON', 'ORGANIZATION']

Client_class_list = []


"""
ADDRESS_list = []
AGE_list = []
CREDIT_DEBIT_CVV_list = []
CREDIT_DEBIT_EXPIRY_list = []
CREDIT_DEBIT_NUMBER_list = []
DRIVER_ID_list = []
# EMAIL_list = []# too many email, email not in PII_collect_list
PHONE_list = []
NAME_list = []# NAME not in PII_collect_list
PASSWORD_list = []
BANK_ACCOUNT_NUMBER_list = []
PASSPORT_NUMBER_list = []
SSN_list = []
Category_list = []# Category not in PII_collect_list
"""

# Load the entire model on the GPU 0
# device_map = {"": 4}

# longformer
tokenizer = LongformerTokenizer.from_pretrained("potsawee/longformer-large-4096-answering-race")
# model = LongformerForMultipleChoice.from_pretrained("potsawee/longformer-large-4096-answering-race")
model = LongformerForMultipleChoice.from_pretrained("potsawee/longformer-large-4096-answering-race", device_map="auto")


# model = model.to("cuda")


client = boto3.client('comprehend')


# /home/mk585/LLM_dataprivacy/maildir/
rootdir = "/home/mk585/LLM_dataprivacy/maildir"

max_count = 0

for directory, subdirectory, filnames in os.walk(rootdir):
    # print(directory, subdirectory, filnames)
    if (len(subdirectory) == 0):
        for index_file in range(len(filnames)):
            max_count += 1
            if(max_count > 21):# 4
                #martinc
                build_huggingace_dataset(Client_class_list)
                print("max_count quit-----------------")
                quit()

            PII_text_list = []
            PII_type_list = []

            Entities_list = []
            Entities_Category_list = []

            Entities_person_name_dict = {}

            path = os.path.join(directory, filnames[index_file])
            
            print("path---------------------------------------------")
            print(path)

            with open(path) as f:
                text_input = f.read()

            # martinc


            print("original text_input-------------------------------------")
            print(text_input)
            
            anchor_pos = text_input.find("X-FileName:")# X-FileName:  # X-Origin
            truncate_anchor_text = text_input[anchor_pos:]

            change_line_anchor_pos = truncate_anchor_text.find("\n")

            change_line_anchor_pos += 1

            text_input = truncate_anchor_text[change_line_anchor_pos:]
            
            """
            anchor_pos = text_input.find("X-Origin:")# X-FileName:
            truncate_anchor_text = text_input[anchor_pos:]

            change_line_anchor_pos = truncate_anchor_text.find("\n")

            change_line_anchor_pos += 1

            option_start_pos = anchor_pos + change_line_anchor_pos
            """
            # martinc replace string
            # text_input = text_input.replace("Mike Hutchins", "Horton, Mark")
            # text_input = text_input.replace('Mike Hutchins', 'Paulo Santos')
            # text_input = text_input.replace('Mike Hutchins', 'Mark Horton ')

            # print("text_input")
            # rint(text_input)
            # print("type(text_input):", type(text_input))

            detect_pii_response_list_e = client.detect_pii_entities(
                Text=text_input,
                LanguageCode='en'
            )

            detect_pii_response_list = detect_pii_response_list_e['Entities']
            for i in range(len(detect_pii_response_list)):
                detect_pii_response = detect_pii_response_list[i]
                if (detect_pii_response['Type'] in PII_collect_list):
                    # if (detect_pii_response['EndOffset'] >= option_start_pos):
                    PII_text_list.append(text_input[detect_pii_response['BeginOffset']:detect_pii_response['EndOffset']])
                    PII_type_list.append(detect_pii_response['Type'])
            
            # means no PII we need in this file
            if(len(PII_text_list) == 0):
                continue

            detect_entities_response_list_e = client.detect_entities(
                Text=text_input,
                LanguageCode='en'
            )
            detect_entities_response_list = detect_entities_response_list_e['Entities']
            for i in range(len(detect_entities_response_list)):
                detect_entities_response = detect_entities_response_list[i]
                # Entities_collect_list
                if (detect_entities_response['Type'] in Entities_collect_list):
                    # if (detect_entities_response['EndOffset'] >= option_start_pos):
                    Entities_list.append(detect_entities_response['Text'])
                    # martin test
                    Entities_Category_list.append(detect_entities_response['Type'])



            # print("text_input")
            # print(text_input)

            # print("detect_entities_response_list_e")
            # print(detect_entities_response_list_e)

            # print("detect_pii_response_list_e")
            # print(detect_pii_response_list_e)

            print("PII_text_list")
            print(PII_text_list)
            print("PII_type_list")
            print(PII_type_list)
            print("Entities_list")
            print(Entities_list)

            # martinc test change
            # Entities_list[0] = "Mike"
            # print("new Entities_list")
            # print(Entities_list)

            # print("Entities_Category_mapping")
            # print(Entities_Category_mapping)

            # use model to classify which PIIs are belonging to which users

            print("model to classify------------------------------------------------------------------")

            # options  = ['John', 'Zhang Wei', 'Alice', 'Sunshine Spa', 'AnyCompany Financial Services, LLC', 'Other']
            # Entities_list.append('Other')
            # options = Entities_list

            # deduplicate items in options
            # options = list(set(options))
            seen = set()
            options = []
            for item in Entities_list:
                if item not in seen:
                    seen.add(item)
                    options.append(item)


            if (len(PII_text_list) == 0 or len(options) == 0):
                print("length PII_text_list or length options zero => continue")
                continue

            
            # clean abnormal words in option, ex: \n
            for op_index in range(len(options)):
                temp_option = options[op_index]
                oprion_anchor = temp_option.find("\n")
                if (oprion_anchor >= 0):
                    temp_option = temp_option[:oprion_anchor]
                    options[op_index] = temp_option
            

            # construct entities category dict
            Entities_Category_mapping = {}
            for i in range(len(options)):
                Entities_Category_mapping[options[i]] = Entities_Category_list[i]

            print("Entities_Category_mapping")
            print(Entities_Category_mapping)

            for qid in range(len(PII_text_list)):
                # options = Entities_list
                # print("original options:", options)
        
                PII_text = PII_text_list[qid]
                PII_type = PII_type_list[qid]
                # question = "Whose personal information is 'XXXXX0000'?"
                # question = f"Whose personal information is '{PII_text}'?"
                # question = f"Whose information does {PII_text} belong to?"

                question = f"{PII_text} belongs to _ ."

                # question = f"{PII_text}, which is a {PII_type}, belongs to _ ."
                
                # {PII_text}, which is a {PII_type}, belongs to _ ."

                # 9586, which is a CREDIT_DEBIT_NUMBER, belongs to _.

                # Whose information does this belong to?
                # text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['response'][i]}"

                """
                # ['Mike Hutchins', 'Taylor, Liz', 'Lawrence_Whalley_Mar2002', 'Whalley, Greg', 'Whalley', 'Cathy Mulligan', 'Michael Hutchins']
                # martinc test
                if (PII_text == "3783 013404 21003"):
                    options[0] = 'Mike'
                    options[-1] = 'Michael'
                    print("3783 013404 21003 new options:", options)

                if (PII_text == "01/03"):
                    options[0] = 'Mike'
                    options[-1] = 'Michael'
                    print("01/03 new options:", options)
                """

                inputs = prepare_answering_input(
                    tokenizer=tokenizer, question=question,
                    options=options, context=text_input,
                    )
                outputs = model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0].tolist()

                # patch: if prob not fit, turn specific prob to zero. SSN should belong to PERSON, not ORGANIZATION
                for prob_index in range(len(prob)):
                    t_options = options[prob_index]
                    if ((PII_type == "SSN") and (Entities_Category_mapping[t_options] != "PERSON")):
                        prob[prob_index] = 0.0


                selected_answer = options[np.argmax(prob)]

                # print("question:", question)
                # print(prob)
                # print(selected_answer)
                
                
                # patch
                # find top2 value, if is closer than 10%
                # if both are person, and with same part name
                # we will delete the same part of name, and put it in the option, run model
                if (len(prob) >= 2):
                    top2_value = heapq.nlargest(2, prob)
                    diff_percentage = (top2_value[0] - top2_value[1]) / top2_value[1]
                    if (diff_percentage < 0.1):
                        top2_index = heapq.nlargest(2, range(len(prob)), prob.__getitem__)
                        if(Entities_Category_list[top2_index[0]] == 'PERSON' and Entities_Category_list[top2_index[1]] == 'PERSON'):
                            first_options = options[top2_index[0]]
                            second_options = options[top2_index[1]]
                            original_top_options = [first_options, second_options]

                            first_options_Entities_Category = Entities_Category_list[top2_index[0]]
                            second_options_Entities_Category = Entities_Category_list[top2_index[1]]
                            original_top_options_Entities_Category = [first_options_Entities_Category, second_options_Entities_Category]


                            first_split = ''
                            second_split = ''
                            if(first_options.find(',') >= 0):
                                first_options_str_list = first_options.split(',')
                                first_split = ','
                            else:
                                first_options_str_list = first_options.split()
                                first_split = ' '
                            if(second_options.find(',') >= 0):
                                second_options_str_list = second_options.split(',')
                                second_split = ','
                            else:
                                second_options_str_list = second_options.split()
                                second_split = ' '

                            for strip_i in range(len(first_options_str_list)):
                                first_options_str_list[strip_i] = first_options_str_list[strip_i].strip()
                        
                            for strip_i in range(len(second_options_str_list)):
                                second_options_str_list[strip_i] = second_options_str_list[strip_i].strip()

                            find_same_str = ""
                            for first_i in range(len(first_options_str_list)):
                                first_options_str = first_options_str_list[first_i]
                                for second_i in range(len(second_options_str_list)):
                                    second_options_str = second_options_str_list[second_i]
                                    if(first_options_str == second_options_str):
                                        find_same_str = first_options_str
                                        break

                            if(find_same_str != ""):

                                # pop same word from first and second list
                                if(len(first_options_str_list) > 1):
                                    for f_pop_index in range(len(first_options_str_list)):
                                        if(f_pop_index == first_i):
                                            break
                                    first_options_str_list.pop(f_pop_index)

                                if(len(second_options_str_list) > 1):
                                    for s_pop_index in range(len(second_options_str_list)):
                                        if(s_pop_index == second_i):
                                            break
                                    second_options_str_list.pop(s_pop_index)

                                if(len(first_options_str_list) > 1):
                                    first_options = first_split.join(first_options_str_list)
                                else:
                                    first_options = first_options_str_list[0]
                                if(len(second_options_str_list) > 1):
                                    second_options = second_split.join(second_options_str_list)
                                else:
                                    second_options = second_options_str_list[0]

                                new_options = [first_options, second_options]

                                inputs = prepare_answering_input(
                                    tokenizer=tokenizer, question=question,
                                    options=new_options, context=text_input,
                                    )
                                outputs = model(**inputs)
                                prob = torch.softmax(outputs.logits, dim=-1)[0].tolist()
                                # selected_answer = new_options[np.argmax(prob)]
                                selected_answer = original_top_options[np.argmax(prob)]

                                # print("new_options:", new_options)
                                # print("question:", question)
                                # print(prob)
                                # print(selected_answer)

                print("question:", question)
                print("options:", options)
                print(prob)
                print(selected_answer)
                selected_category = Entities_Category_mapping[selected_answer]
                print(selected_category)

                # Client_class_list
                if (len(Client_class_list) == 0):
                    new_client = Client()
                    new_client.add_PII(selected_answer, "NAME")
                    new_client.add_PII(selected_category, "CATEGORY")
                    # PII_text PII_type
                    new_client.add_PII(PII_text, PII_type)
                    Client_class_list.append(new_client)
                    # print("zero selected_answer:", selected_answer)
                else:
                    # search all Client_class_list Name, if yes, insert. if no, add new client
                    is_find = False
                    for client_index in range(len(Client_class_list)):
                        client_class = Client_class_list[client_index]
                        if (client_class.NAME == selected_answer):
                            client_class.add_PII(PII_text, PII_type)
                            Client_class_list[client_index] = client_class
                            is_find = True
                            break
                    # print("client_index:", client_index, " selected_answer:", selected_answer)
                    # if (client_index == len(Client_class_list)):
                    if (is_find == False):
                        new_client = Client()
                        new_client.add_PII(selected_answer, "NAME")
                        new_client.add_PII(selected_category, "CATEGORY")
                        # PII_text PII_type
                        new_client.add_PII(PII_text, PII_type)
                        Client_class_list.append(new_client)                        


                #for check_index in range(len(Client_class_list)):
                    #print("check_index:", check_index, " Client_class_list[check_index].NAME:", Client_class_list[check_index].NAME)
                    

                
                      

            # print("example quit-----------------------")
            # quit()


            



print("all quit-----------------")
quit()










tokenizer = LongformerTokenizer.from_pretrained("potsawee/longformer-large-4096-answering-race")
model = LongformerForMultipleChoice.from_pretrained("potsawee/longformer-large-4096-answering-race")

#context = r"""Chelsea's mini-revival continued with a third victory in a row as they consigned struggling Leicester City to a fifth consecutive defeat.
#Buoyed by their Champions League win over Borussia Dortmund, Chelsea started brightly and Ben Chilwell volleyed in from a tight angle against his old club.
#Chelsea's Joao Felix and Leicester's Kiernan Dewsbury-Hall hit the woodwork in the space of two minutes, then Felix had a goal ruled out by the video assistant referee for offside.
#Patson Daka rifled home an excellent equaliser after Ricardo Pereira won the ball off the dawdling Felix outside the box.
#But Kai Havertz pounced six minutes into first-half injury time with an excellent dinked finish from Enzo Fernandez's clever aerial ball.
#Mykhailo Mudryk thought he had his first goal for the Blues after the break but his effort was disallowed for offside.
#Mateo Kovacic sealed the win as he volleyed in from Mudryk's header.
#The sliding Foxes, who ended with 10 men following Wout Faes' late dismissal for a second booking, now just sit one point outside the relegation zone.
#""".replace('\n', ' ')
#question = "Who had a goal ruled out for offside?"
#options  = ['Ricardo Pereira', 'Ben Chilwell', 'Joao Felix', 'The Foxes']



context = r"""Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card account 1111-XXXX-1111-XXXX has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at AnySpa@example.com.
""".replace('\n', ' ')
question = "Whose personal information is 'XXXXXX1111'?"
# options  = ['Zhang Wei', 'John', 'Alice']
options  = ['John', 'Zhang Wei', 'Alice', 'Sunshine Spa', 'AnyCompany Financial Services, LLC', 'Other']



"""
(a) July 31st

(b) XXXXXX1111

(c) XXXXX0000

(d) 123 Main St

(e) None of the above
"""


inputs = prepare_answering_input(
    tokenizer=tokenizer, question=question,
    options=options, context=context,
    )
outputs = model(**inputs)
prob = torch.softmax(outputs.logits, dim=-1)[0].tolist()
selected_answer = options[np.argmax(prob)]

print("question:", question)
print(prob)
print(selected_answer)



# question = "who has information XXXXX0000?"
question = "Whose personal information is 'XXXXX0000'?"

inputs = prepare_answering_input(
    tokenizer=tokenizer, question=question,
    options=options, context=context,
    )
outputs = model(**inputs)
prob = torch.softmax(outputs.logits, dim=-1)[0].tolist()
selected_answer = options[np.argmax(prob)]

print("question:", question)
print(prob)
print(selected_answer)



# question = "who has information XXXXX0000?"
question = "Whose personal information is '123 Main St'?"

inputs = prepare_answering_input(
    tokenizer=tokenizer, question=question,
    options=options, context=context,
    )
outputs = model(**inputs)
prob = torch.softmax(outputs.logits, dim=-1)[0].tolist()
selected_answer = options[np.argmax(prob)]

print("question:", question)
print(prob)
print(selected_answer)



# question = "who has information XXXXX0000?"
question = "Whose personal information is 'AnySpa@example.com'?"

inputs = prepare_answering_input(
    tokenizer=tokenizer, question=question,
    options=options, context=context,
    )
outputs = model(**inputs)
prob = torch.softmax(outputs.logits, dim=-1)[0].tolist()
selected_answer = options[np.argmax(prob)]

print("question:", question)
print(prob)
print(selected_answer)



"""
text_input = "Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card account 1111-XXXX-1111-XXXX has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at AnySpa@example.com."


client = boto3.client('comprehend')

response = client.detect_pii_entities(
    Text=text_input,
    LanguageCode='en'
)

print("aws response----------------------")
print(response)
"""
quit()


"""
aws comprehend detect-pii-entities \
    --language-code en \
    --text "Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card \
        account 1111-XXXX-1111-XXXX has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, \
        we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. \
        Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at AnySpa@example.com."
"""

"""
print("begin")

text_input = "Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card account 1111-XXXX-1111-XXXX has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at AnySpa@example.com."

# cmd='aws deploy push --application-name SomeApp --s3-location  s3://bucket/Deploy/db_schema.zip --ignore-hidden-files'

cmd = 'aws comprehend detect-pii-entities -language-code en --text ' + text_input

push=subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE)
print (push.returncode)

print("done")

"""

text_input = "Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card account 1111-XXXX-1111-XXXX has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at AnySpa@example.com."


client = boto3.client('comprehend')

response = client.detect_pii_entities(
    Text=text_input,
    LanguageCode='en'
)

print("response----------------------")
print(response)

ADDRESS_list = []
AGE_list = []
CREDIT_DEBIT_CVV_list = []
CREDIT_DEBIT_EXPIRY_list = []
CREDIT_DEBIT_NUMBER_list = []
DRIVER_ID_list = []
EMAIL_list = []
PHONE_list = []
NAME_list = []
PASSWORD_list = []
BANK_ACCOUNT_NUMBER_list = []
PASSPORT_NUMBER_list = []
SSN_list = []

PII_list = []

response_list = response['Entities']
for i in range(len(response_list)):
    temp_response = response_list[i]
    if (temp_response['Type'] == 'NAME'):

        # NAME_list.append(temp_response['Type'])
        NAME_list.append(text_input[temp_response['BeginOffset']:temp_response['EndOffset']])
    else:
        PII_list.append(text_input[temp_response['BeginOffset']:temp_response['EndOffset']])

print("NAME_list")
print(NAME_list)

print("PII_list")
print(PII_list)




model = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

prompt_string = "which information belogs to Zhang Wei: 'July 31st', 'XXXXXX1111', 'XXXXX0000', '123 Main St', 'AnySpa@example.com'. Please list the information only belongs to Zhang Wei \n Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card account 1111-XXXX-1111-XXXX has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at AnySpa@example.com.\n response:"

instruction = "which information belogs to Zhang Wei: 'July 31st', 'XXXXXX1111', 'XXXXX0000', '123 Main St', 'AnySpa@example.com'. Please list the information only belongs to Zhang Wei"
input = "Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card account 1111-XXXX-1111-XXXX has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at AnySpa@example.com."
prompt_input = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

sequences = pipeline(
    #"Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    prompt_string,
    max_length=500,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")



"""
model_name = "decapoda-research/llama-7b-hf"
# model_name = "decapoda-research/llama-65b-hf"


# Activate 4-bit precision base model loading
use_4bit = True
 
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
 
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
 
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)


# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Load the entire model on the GPU 0
device_map = {"": 0}

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1


# prompt = "Hey, are you conscious? Can you talk to me?"

# inputs = tokenizer(prompt, return_tensors="pt")#

# instruction='extract Zhang Wei information'
# ['July 31st', 'XXXXXX1111', 'XXXXX0000', '123 Main St', 'AnySpa@example.com']
instruction='which information below is belonging to Zhang Wei: '
for i in range(len(PII_list)):
    instruction += PII_list[i]
    if (i < len(PII_list)-1):
        instruction += ", "
input = "Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card account 1111-XXXX-1111-XXXX has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at AnySpa@example.com."


prompt_input = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

print("prompt_input------------------------")
print(prompt_input)

inputs = tokenizer(prompt_input, return_tensors="pt").to("cuda")

generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("before example output_string")
print(output_string)



# instruction='extract Zhang Wei information'
# ['July 31st', 'XXXXXX1111', 'XXXXX0000', '123 Main St', 'AnySpa@example.com']
instruction='which information below is belonging to John: '
for i in range(len(PII_list)):
    instruction += PII_list[i]
    if (i < len(PII_list)-1):
        instruction += ", "
input = "Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card account 1111-XXXX-1111-XXXX has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at AnySpa@example.com."


prompt_input = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

print("prompt_input------------------------")
print(prompt_input)

inputs = tokenizer(prompt_input, return_tensors="pt").to("cuda")

generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("before example output_string")
print(output_string)





# instruction='extract Zhang Wei information'
# ['July 31st', 'XXXXXX1111', 'XXXXX0000', '123 Main St', 'AnySpa@example.com']
instruction='which information below is belonging to Alice: '
for i in range(len(PII_list)):
    instruction += PII_list[i]
    if (i < len(PII_list)-1):
        instruction += ", "
input = "Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card account 1111-XXXX-1111-XXXX has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at AnySpa@example.com."


prompt_input = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

print("prompt_input------------------------")
print(prompt_input)

inputs = tokenizer(prompt_input, return_tensors="pt").to("cuda")

generate_ids = model.generate(inputs.input_ids, max_length=200)
output_string = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("before example output_string")
print(output_string)
"""