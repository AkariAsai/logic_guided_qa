import jsonlines
import os
from tqdm import tqdm
import argparse
import re
import copy
import random


def read_jsonl(jsonl_file):
    # a helper function to load a jsonl file.
    lines = []
    print("loading examples from {0}".format(jsonl_file))
    with jsonlines.open(jsonl_file) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def get_train_examples(args):
    # a methhod to load examples from a train file.
    print("LOOKING AT {} train".format(args.data_dir))
    return create_examples(read_jsonl(os.path.join(args.data_dir, "train.jsonl")), "train", args)


def get_dev_examples(args):
    # a methhod to load examples from a dev file.
    print("LOOKING AT {} dev".format(args.data_dir))
    return create_examples(read_jsonl(os.path.join(args.data_dir, "dev.jsonl")), "dev", args)


def get_test_examples(args):
    # a methhod to load examples from a test file.
    print("LOOKING AT {} test".format(args.data_dir))
    return create_examples(read_jsonl(os.path.join(args.data_dir, "test.jsonl")), "test", args)
    
    
def replace_words(sent):
    sent = sent.lower()
    if "more/larger" in sent:
        return True, sent.replace("more/larger", "less or smaller")
    if "less or smaller" in sent:
        return True, sent.replace("less or smaller", "more/larger")
    if "more/stronger" in sent:
        return True, sent.replace("more/stronger", "less or weaker")
    if "less or weaker" in sent:
        return True, sent.replace("less or weaker", "more/stronger")
    if "more/faster" in sent:
        return True, sent.replace("more/faster", "less or slower")
    if "less or slower" in sent:
        return True, sent.replace("less or slower", "more/faster")
    if "more/greater" in sent:
        return True, sent.replace("more/greater", "less or smaller")
    if "more/more powerful" in sent:
        return True, sent.replace("more/more powerful", "less or less powerful")
    if "less or less powerful" in sent:
        return True, sent.replace("less or less powerful", "more/more powerful")
    if "more" in sent:
        return True, sent.replace("more", "less")
    if "less" in sent:
        return True, sent.replace("less", "more")
    if "larger" in sent:
        return True, sent.replace("larger", "smaller")
    if "older" in sent:
        return True, sent.replace("older", "younger")
    if "higher" in sent:
        return True, sent.replace("higher", "lower")
    elif "fewer" in sent:
        return True, sent.replace("fewer", "more")
    elif "smaller" in sent:
        return True, sent.replace("smaller", "larger")
    elif "younger" in sent:
        return True, sent.replace("younger", "order")
    elif "lower" in sent:
        return True, sent.replace("lower", "higher")
    elif "more and more" in sent:
        return True, sent.replace("more and more", "less and less")
    elif "less and less" in sent:
        return True, sent.replace("less and less", "more and more")
    elif "a lot of" in sent:
        return True, sent.replace("a lot of", "few")
    elif "increase" in sent:
        return True, sent.replace("increase", "decrease")
    elif "decrease" in sent:
        return True, sent.replace("decrease", "increase")
    elif "added" in sent:
        return True, sent.replace("added", "removed")
    elif "removed" in sent:
        return True, sent.replace("removed", "added")
    elif "add" in sent:
        return True, sent.replace("add", "remove")
    elif "remove" in sent:
        return True, sent.replace("remove", "add")
    elif "smallest" in sent:
        return True, sent.replace("smallest", "largest")
    elif "largest" in sent:
        return True, sent.replace("largest", "smallest")
    elif "earlier" in sent:
        return True, sent.replace("earlier", "later")
    elif "later" in sent:
        return True, sent.replace("later", "earlier")
    elif "early" in sent:
        return True, sent.replace("early", "late")
    elif "heat up" in sent:
        return True, sent.replace("heat up", "cool down")
    elif "cool down" in sent:
        return True, sent.replace("cool down", "heat up")
    elif "heated" in sent:
        return True, sent.replace("heated", "cooled")
    elif "cooled" in sent:
        return True, sent.replace("cooled", "heated")
    elif "most" in sent:
        return True, sent.replace("most", "least")
    elif "lest" in sent:
        return True, sent.replace("lest", "most")
    elif "colder" in sent:
        return True, sent.replace("colder", "warmer")
    elif "walmer" in sent:
        return True, sent.replace("walmer", "colder")
    elif "cold" in sent:
        return True, sent.replace("cold", "warm")
    elif "stronger" in sent:
        return True, sent.replace("stronger", "weaker")
    elif "weaker" in sent:
        return True, sent.replace("weaker", "stronger")
    elif "closer" in sent:
        return True, sent.replace("closer", "far")
    elif "greater" in sent:
        return True, sent.replace("greater", "lower")
    elif "longer" in sent:
        return True, sent.replace("longer", "shorter")
    elif "shorter" in sent:
        return True, sent.replace("shorter", "longer")
    elif "cooler" in sent:
        return True, sent.replace("cooler", "warmer")
    elif "reduce" in sent:
        return True, sent.replace("reduce", "increase")
    elif "bigger" in sent:
        return True, sent.replace("bigger", "smaller")
    elif "gain weight" in sent:
        return True, sent.replace("gain weight", "lose weight")
    elif "lose weight" in sent:
        return True, sent.replace("lose weight", "gain weight")
    elif "likely" in sent:
        return True, sent.replace("likely", "unlikely")
    elif "unlikely" in sent:
        return True, sent.replace("unlikely", "likely")
    elif "worse" in sent:
        return True, sent.replace("worse", "better")
    elif "better" in sent:
        return True, sent.replace("better", "worse")
    elif "least" in sent:
        return True, sent.replace("least", "most")
    elif "most" in sent:
        return True, sent.replace("most", "least")
    elif "lowest" in sent:
        return True, sent.replace("lowest", "highest")
    elif "highest" in sent:
        return True, sent.replace("highest", "lowest")
    elif "faster" in sent:
        return True, sent.replace("faster", "slower")
    elif "slower" in sent:
        return True, sent.replace("slower", "faster")
    elif "raise" in sent:
        return True, sent.replace("raise", "drop")
    elif "drop" in sent:
        return True, sent.replace("drop", "raise")
    elif "good" in sent:
        return True, sent.replace("good", "bad")
    elif "bad" in sent:
        return True, sent.replace("bad", "good")
    elif "stricter" in sent:
        return True, sent.replace("stricter", "easier")
    else:
        return False, sent

def add_negation(sent):
    sent = sent.lower()
    if "doesnt" in sent:
        return True, sent.replace("doesnt", "does")
    elif "doesn't" in sent:
        return True, sent.replace("doesn't", "does")
    elif "does not" in sent:
        return True, sent.replace("does not", "does")
    elif "do not " in sent:
        return True, sent.replace("do not ", "")
    elif "dont " in sent:
        return True, sent.replace("dont ", "")
    elif "don't " in sent:
        return True, sent.replace("don't ", "")
    elif "do not " in sent:
        return True, sent.replace("do not ", "")
    else:
        return False, sent
    

def more_less_transforme(sent):
    # Transform a question into a symmetric question.
    replaced_synonym, replaced_synonym_question = replace_words(sent)
    remove_neg_synonym, remove_neg_question = add_negation(sent)
    if replaced_synonym is True and remove_neg_synonym is True:
        return random.sample([(replaced_synonym, replaced_synonym_question), (remove_neg_synonym, remove_neg_question)], k=1)[0]
    elif replaced_synonym is True and remove_neg_synonym is False:
        return replaced_synonym, replaced_synonym_question
    elif replaced_synonym is False and remove_neg_synonym is True:
        return remove_neg_synonym, remove_neg_question
    else:
        return False, sent

    

    
def extract_cause_effect(question):
    # a helper method to extract events for cause and effect given a WIQA question.
    q_cause = re.findall('suppose (.*) happens', question)[0]
    q_effect = re.findall('how will it affect (.*).', question)[0]
    return q_cause, q_effect


def flip_label(label):
    # a helper method to flip a label into opposite one.
    if label == "more":
        return "less"
    elif label == "less":
        return "more"
    elif label == "no_effect":
        return label
    else:
        raise NotImplementedError(
            "The label types should be more, less or no_effect.")


def flip_answer_label_as_choice(label_as_choice):
    # a helper method to flip a label into opposite one.
    if label_as_choice == "A":
        return "B"
    elif label_as_choice == "B":
        return "A"
    elif label_as_choice == "C":
        return label_as_choice
    else:
        raise NotImplementedError(
            "The label_as_choice should be A, B or C.")


def add_transitive_examples_one(examples, args, original_questions):
    # This method adds transitive examples by searching over the data and finding two questions (original labels = "more") sharing one event.
    cause_dic = {}
    effect_dic = {}

    for example in examples:
        if "_symmetric" in example['metadata']['ques_id'] or "_transit" in example['metadata']['ques_id']:
            continue
        question = example["question"]["stem"]
        cause, effect = extract_cause_effect(question)
        if cause == "True" or effect == "True" or cause == "False" or effect == "False":
            continue
        # create cause / effect data.
        cause_dic.setdefault(cause, [])
        cause_dic[cause].append(example)
        effect_dic.setdefault(effect, [])
        effect_dic[effect].append(example)

    for seed_effect, first_hop_examples in tqdm(effect_dic.items()):
        if seed_effect not in cause_dic:
            continue
        for first_hop_example in first_hop_examples:
            first_cause, first_effect = extract_cause_effect(
                first_hop_example["question"]["stem"])
            if first_cause == "True":
                continue
            second_hop_examples = cause_dic[seed_effect]
            for second_hop_example in second_hop_examples:
                second_cause, second_effect = extract_cause_effect(
                    second_hop_example["question"]["stem"])
                if second_cause == "True" or second_cause == "False":
                    continue
                if random.random() > args.sample_ratio_augmentation:
                    continue

                if first_hop_example["question"]["answer_label"] == "more" and second_hop_example["question"]["answer_label"] == "more":
                    assert first_effect == second_cause
                    new_question = "suppose {0} happens, how will it affect {1}.".format(
                        first_cause, second_effect)
                    new_example = copy.deepcopy(second_hop_example)
                    if args.eval_mode is False or (args.eval_mode is True and new_question in original_questions):
                        new_example["question"]["stem"] = new_question
                        new_example['metadata']['ques_id'] = "{0}@{1}_transit_one".format(
                            first_hop_example['metadata']['ques_id'], second_hop_example['metadata']['ques_id'])
                        examples.append(new_example)
    return examples


def add_transitive_examples_two(examples, args, original_questions):
    # This method adds transitive examples by searching over the data and finding two questions (original labels = "more" and "less") sharing one event.
    cause_dic = {}
    effect_dic = {}

    for example in examples:
        if "_symmetric" in example['metadata']['ques_id'] or "_transit" in example['metadata']['ques_id']:
            continue
        question = example["question"]["stem"]
        cause, effect = extract_cause_effect(question)
        if cause == "True" or effect == "True":
            continue
        # create cause / effect data.
        cause_dic.setdefault(cause, [])
        cause_dic[cause].append(example)
        effect_dic.setdefault(effect, [])
        effect_dic[effect].append(example)

    for seed_effect, first_hop_examples in tqdm(effect_dic.items()):
        if seed_effect not in cause_dic:
            continue
        for first_hop_example in first_hop_examples:
            first_cause, first_effect = extract_cause_effect(
                first_hop_example["question"]["stem"])
            if first_cause == "True":
                continue
            second_hop_examples = cause_dic[seed_effect]
            for second_hop_example in second_hop_examples:
                second_cause, second_effect = extract_cause_effect(
                    second_hop_example["question"]["stem"])
                if second_cause == "True":
                    continue
                if random.random() > args.sample_ratio_augmentation:
                    continue

                if first_hop_example["question"]["answer_label"] == "more" and second_hop_example["question"]["answer_label"] == "less":
                    new_example = copy.deepcopy(second_hop_example)
                    assert first_effect == second_cause
                    new_example["question"]["stem"] = "suppose {0} happens, how will it affect {1}.".format(
                        first_cause, second_effect)
                    new_example['metadata']['ques_id'] = "{0}@{1}_transit_two".format(
                        first_hop_example['metadata']['ques_id'], second_hop_example['metadata']['ques_id'])
                    new_example['metadata']['answer_label'] = "less"
                    new_example['metadata']['answer_label_as_choice'] = "B"
                    examples.append(new_example)
    return examples


def create_symmetric_one(question, label):
    # Create a symmetric question (original label="more") by swapping a word with polarity in "cause" into the opposite one.
    q_cause, q_effect = extract_cause_effect(question)
    more_less_included, transformed_q_cause = more_less_transforme(q_cause)

    if label == "more" and more_less_included is True:
        return True, "suppose {0} happens, how will it affect {1}.".format(transformed_q_cause, q_effect)
    else:
        return False, question


def create_symmetric_two(question, label):
    # Create a symmetric question (original label="less") by swapping a word with polarity in "cause" into the opposite one.
    q_cause, q_effect = extract_cause_effect(question)
    more_less_included, transformed_q_cause = more_less_transforme(q_cause)

    if label == "less" and more_less_included is True:
        return True, "suppose {0} happens, how will it affect {1}.".format(transformed_q_cause, q_effect)
    else:
        return False, question


def create_symmetric_three(question, label):
    # Create a symmetric question (original label="more") by swapping a word with polarity in "effect" into the opposite one.
    q_cause, q_effect = extract_cause_effect(question)
    more_less_included, transformed_q_effect = more_less_transforme(q_effect)

    if label == "more" and more_less_included is True:
        return True, "suppose {0} happens, how will it affect {1}.".format(q_cause, transformed_q_effect)
    else:
        return False, question


def create_symmetric_four(question, label):
    # Create a symmetric question (original label="less") by swapping a word with polarity in "effect" into the opposite one.
    q_cause, q_effect = extract_cause_effect(question)
    more_less_included, transformed_q_effect = more_less_transforme(q_effect)

    if label == "less" and more_less_included is True:
        return True, "suppose {0} happens, how will it affect {1}.".format(q_cause, transformed_q_effect)
    else:
        return False, question


def create_symmetric_five(question, label):
    # Create a symmetric question (original label="more") by swapping words with polarity in both "effect" and cause into the opposite ones.
    # The label should remain same.
    q_cause, q_effect = extract_cause_effect(question)
    more_less_included_cause, transformed_q_cause = more_less_transforme(
        q_cause)
    more_less_included_effect, transformed_q_effect = more_less_transforme(
        q_effect)
    if label == "more" and more_less_included_cause is True and more_less_included_effect is True:
        return True, "suppose {0} happens, how will it affect {1}.".format(transformed_q_cause, transformed_q_effect)
    else:
        return False, question


def create_symmetric_six(question, label):
    # Create a symmetric question (original label="less") by swapping words with polarity in both "effect" and cause into the opposite ones.
    # The label should remain same.
    q_cause, q_effect = extract_cause_effect(question)
    more_less_included_cause, transformed_q_cause = more_less_transforme(
        q_cause)
    more_less_included_effect, transformed_q_effect = more_less_transforme(
        q_effect)
    if label == "less" and more_less_included_cause is True and more_less_included_effect is True:
        return True, "suppose {0} happens, how will it affect {1}.".format(transformed_q_cause, transformed_q_effect)
    else:
        return False, question


def create_symmetric_seven(question, label):
    # Create a symmetric question (original label="no effects") by swapping a word with polarity in "cause" into the opposite one.
    # The label should remain same.
    q_cause, q_effect = extract_cause_effect(question)
    more_less_included, transformed_q_cause = more_less_transforme(q_cause)

    if label == "no_effect" and more_less_included is True:
        return True, "suppose {0} happens, how will it affect {1}.".format(transformed_q_cause, q_effect)
    else:
        return False, question


def create_symmetric_eight(question, label):
    # Create a symmetric question (original label="no effects") by swapping a word with polarity in "effect" into the opposite one.
    # The label should remain same.
    q_cause, q_effect = extract_cause_effect(question)
    more_less_included, transformed_q_effect = more_less_transforme(q_effect)
    if label == "no_effect" and more_less_included is True:
        return True, "suppose {0} happens, how will it affect {1}.".format(q_cause, transformed_q_effect)
    else:
        return False, question


def create_symmetric_nine(question, label):
    # Create a symmetric question (original label="less") by swapping words with polarity in both "effect" and cause into the opposite ones.
    # The label should remain same.
    q_cause, q_effect = extract_cause_effect(question)
    more_less_included_cause, transformed_q_cause = more_less_transforme(
        q_cause)
    more_less_included_effect, transformed_q_effect = more_less_transforme(
        q_effect)
    if label == "no_effect" and more_less_included_cause is True and more_less_included_effect is True:
        return True, "suppose {0} happens, how will it affect {1}.".format(transformed_q_cause, transformed_q_effect)
    else:
        return False, question


def create_examples(jsonlines_data, data_split, args):
    augmented_data = []
    original_questions = [example["question"]["stem"]
                          for example in jsonlines_data]
    print("creating {} data...".format(data_split))
    if args.sample_ratio > 0.0:
        jsonlines_data = random.sample(jsonlines_data, int(
            len(jsonlines_data) * args.sample_ratio))
    for item in tqdm(jsonlines_data):
        # add the original data
        augmented_data.append(item)
        question = item["question"]["stem"]
        answer_labels = item["question"]["answer_label"]
        answer_label_as_choice = item["question"]["answer_label_as_choice"]
        example_id = item['metadata']['ques_id']

        if random.random() <= args.sample_ratio_augmentation:
            is_sc_available, flipped_question = create_symmetric_one(
                question, answer_labels)

            if is_sc_available is True and flipped_question not in original_questions and(args.eval_mode is False or (args.eval_mode is True and flipped_question in original_questions)):
                copied_item = copy.deepcopy(item)
                copied_item["question"]["stem"] = flipped_question
                flipped_label = flip_label(answer_labels)
                new_question_id = "{0}_symmetric_one".format(example_id)
                copied_item["question"]["answer_label"] = flipped_label
                copied_item['question']['answer_label_as_choice'] = flip_answer_label_as_choice(
                    answer_label_as_choice)
                copied_item['metadata']['ques_id'] = new_question_id
                augmented_data.append(copied_item)

        if random.random() <= args.sample_ratio_augmentation:
            is_sc_available, flipped_question = create_symmetric_two(
                question, answer_labels)

            if is_sc_available is True and flipped_question not in original_questions and (args.eval_mode is False or (args.eval_mode is True and flipped_question in original_questions)):
                copied_item = copy.deepcopy(item)
                if args.eval_mode is True and flipped_question not in original_questions:
                    continue
                copied_item["question"]["stem"] = flipped_question
                flipped_label = flip_label(answer_labels)
                new_question_id = "{0}_symmetric_two".format(example_id)
                copied_item["question"]["answer_label"] = flipped_label
                copied_item['question']['answer_label_as_choice'] = flip_answer_label_as_choice(
                    answer_label_as_choice)
                copied_item['metadata']['ques_id'] = new_question_id
                augmented_data.append(copied_item)

        # if args.s3 is True:
        if random.random() <= args.sample_ratio_augmentation:
            is_sc_available, flipped_question = create_symmetric_three(
                question, answer_labels)
            if flipped_question in original_questions:
                continue
            if is_sc_available is True and flipped_question not in original_questions and (args.eval_mode is False or (args.eval_mode is True and flipped_question in original_questions)):
                copied_item = copy.deepcopy(item)
                copied_item["question"]["stem"] = flipped_question
                flipped_label = flip_label(answer_labels)
                new_question_id = "{0}_symmetric_three".format(example_id)
                copied_item["question"]["answer_label"] = flipped_label
                copied_item['question']['answer_label_as_choice'] = flip_answer_label_as_choice(
                    answer_label_as_choice)
                copied_item['metadata']['ques_id'] = new_question_id
                augmented_data.append(copied_item)

        if random.random() <= args.sample_ratio_augmentation:
            is_sc_available, flipped_question = create_symmetric_four(
                question, answer_labels)

            if is_sc_available is True and flipped_question not in original_questions and (args.eval_mode is False or (args.eval_mode is True and flipped_question in original_questions)):
                copied_item = copy.deepcopy(item)
                copied_item["question"]["stem"] = flipped_question
                flipped_label = flip_label(answer_labels)
                new_question_id = "{0}_symmetric_four".format(example_id)
                copied_item["question"]["answer_label"] = flipped_label
                copied_item['question']['answer_label_as_choice'] = flip_answer_label_as_choice(
                    answer_label_as_choice)
                copied_item['metadata']['ques_id'] = new_question_id
                augmented_data.append(copied_item)

        if random.random() <= args.sample_ratio_augmentation:
            is_sc_available, flipped_question = create_symmetric_five(
                question, answer_labels)

            if is_sc_available is True and flipped_question not in original_questions and (args.eval_mode is False or (args.eval_mode is True and flipped_question in original_questions)):
                copied_item = copy.deepcopy(item)
                copied_item["question"]["stem"] = flipped_question
                flipped_label = flip_label(answer_labels)
                new_question_id = "{0}_symmetric_five".format(example_id)
                copied_item['metadata']['ques_id'] = new_question_id
                augmented_data.append(copied_item)
                
        if random.random() <= args.sample_ratio_augmentation:
            is_sc_available, flipped_question = create_symmetric_six(
                question, answer_labels)

            if is_sc_available is True and flipped_question not in original_questions and (args.eval_mode is False or (args.eval_mode is True and flipped_question in original_questions)):
                copied_item = copy.deepcopy(item)
                copied_item["question"]["stem"] = flipped_question
                flipped_label = flip_label(answer_labels)
                new_question_id = "{0}_symmetric_six".format(example_id)
                copied_item['metadata']['ques_id'] = new_question_id
                augmented_data.append(copied_item)

        if random.random() <= args.sample_ratio_augmentation:
            is_sc_available, flipped_question = create_symmetric_seven(
                question, answer_labels)

            if is_sc_available is True and flipped_question not in original_questions and (args.eval_mode is False or (args.eval_mode is True and flipped_question in original_questions)):
                copied_item = copy.deepcopy(item)
                copied_item["question"]["stem"] = flipped_question
                flipped_label = flip_label(answer_labels)
                new_question_id = "{0}_symmetric_seven".format(example_id)
                copied_item['metadata']['ques_id'] = new_question_id
                augmented_data.append(copied_item)

        if random.random() <= args.sample_ratio_augmentation:
            is_sc_available, flipped_question = create_symmetric_eight(
                question, answer_labels)

            if is_sc_available is True and flipped_question not in original_questions and (args.eval_mode is False or (args.eval_mode is True and flipped_question in original_questions)):
                copied_item = copy.deepcopy(item)
                copied_item["question"]["stem"] = flipped_question
                flipped_label = flip_label(answer_labels)
                new_question_id = "{0}_symmetric_eight".format(example_id)
                copied_item['metadata']['ques_id'] = new_question_id
                augmented_data.append(copied_item)

        if random.random() <= args.sample_ratio_augmentation:
            is_sc_available, flipped_question = create_symmetric_nine(
                question, answer_labels)

            if is_sc_available is True and flipped_question not in original_questions and (args.eval_mode is False or (args.eval_mode is True and flipped_question in original_questions)):
                copied_item = copy.deepcopy(item)
                copied_item["question"]["stem"] = flipped_question
                flipped_label = flip_label(answer_labels)
                new_question_id = "{0}_symmetric_nine".format(example_id)
                copied_item['metadata']['ques_id'] = new_question_id
                augmented_data.append(copied_item)

    augmented_data = add_transitive_examples_one(
        augmented_data, args, original_questions)

    augmented_data = add_transitive_examples_two(
        augmented_data, args, original_questions)
    
    print("# of data {}".format(len(augmented_data)))

    return augmented_data


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir")
    parser.add_argument('--store_dev_test_augmented_data', action='store_true',
                        help="Augment eval data")
    # balance data
    parser.add_argument("--sample_ratio", default=-0.1, type=float,
                        help="the random sample rate for data to be used.")
    parser.add_argument("--sample_ratio_augmentation", default=0.5, type=float,
                        help="the random sample rate for augmented data to be added.")

    parser.add_argument('--eval_mode', action='store_true',
                        help="with eval mode, we only add the question included in the original data.")
    args = parser.parse_args()

    augmented_train_data = get_train_examples(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    with jsonlines.open(os.path.join(args.output_dir, "train.jsonl"), mode='w') as writer:
        writer.write_all(augmented_train_data)
        
    if args.store_dev_test_augmented_data is True:
        augmented_dev_data = get_dev_examples(args)
        augmented_test_data = get_test_examples(args)
        with jsonlines.open(os.path.join(args.output_dir, "dev.jsonl"), mode='w') as writer:
            writer.write_all(augmented_dev_data)

        with jsonlines.open(os.path.join(args.output_dir, "test.jsonl"), mode='w') as writer:
            writer.write_all(augmented_test_data)
            
    else:
        with jsonlines.open(os.path.join(args.output_dir, "dev.jsonl"), mode='w') as writer:
            writer.write_all(read_jsonl(
                os.path.join(args.data_dir, "dev.jsonl")))

        with jsonlines.open(os.path.join(args.output_dir, "test.jsonl"), mode='w') as writer:
            writer.write_all(read_jsonl(
                os.path.join(args.data_dir, "test.jsonl")))


if __name__ == "__main__":
    main()
