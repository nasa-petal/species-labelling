import pandas as pd
import nltk
import unicodedata
import ast
import re
import os
import argparse
import json


def get_arg_parser():
    """Allows the user to pass in the path of a CSV or JSON containing paper abstracts.
    Returns:
        argparse.Namespace: Object containing selected options
    """

    def dir_path(string):
        if os.path.isfile(string):
            if ("csv" not in string.lower() and "json" not in string.lower()):
                raise TypeError(string)
            return string
        else:
            raise NotADirectoryError(string)

    parser = argparse.ArgumentParser(description="Input document file paths")
    parser.add_argument(
        "file_path", help="Full path to CSV or JSON file: ./example.csv", type=dir_path)
    parser.add_argument("output_name", help="Name of output file with extension (csv or json): ./output.json", type=str)
    return parser.parse_args()


def read_data(path: str):
    """Tokenizes and formats strings.
    Args:
        path : string
            A string representing a full path to a given file.
    Returns:
        Tuple
            Pandas DataFrame
                List of records in a Pandas DataFrame.
            Dict
                Dictionary containing species scientific names and a corresponding list of common names.
            Common Names
                List of species common names from NCBI Blast and PubMed
            Excluded Words
                List of words that will be ignored in processing

    """
    if ("csv" in path.lower()):
        petal_records = pd.read_csv(path)
    elif ("json" in path.lower()):
        petal_records = pd.read_json(path)

    with open("species_dict.json", "r") as file:
        species_dict = ast.literal_eval("".join(file.readlines()))

    common_names = []
    with open("common_names.txt", "r") as file:
        common_names = ast.literal_eval("".join(file.readlines()))  

    excluded_words = []
    with open("excluded_words.txt", "r") as file:
        for line in file.readlines():
            excluded_words.append(line.rstrip("\n"))

    return (petal_records, species_dict, common_names, excluded_words)    


def strip_accents(s:str):
    """Removes non-ascii characters from a string.
    Args:
        s : str
            The string being normalized.
    Returns:
        str
            Normalized string.
    """
    return ''.join(c.lower() for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def lemmatize_paragraph(s: str):
    """Lemmatizes a string.
    Args:
        s : str
            The string being lemmatized.
    Returns:
        str
            lemmatized string.
    """
    wnl = nltk.WordNetLemmatizer()
    r_ex = re.sub(r'[,.\']', "", s)
    n_ex = " ".join([wnl.lemmatize(word) for word in r_ex.split(" ")])
    return n_ex


def create_tagged_n_grams(df: pd.DataFrame):
    """Removes non-ascii characters from a string.
    Args:
        df : Pandas DataFrame
            A dataframe containing biomimicry records.
    Returns:
        Pandas DataFrame
            A modified version of the input with multiple n-gram fields and a lemmatized abstract.
    """ 
    full_tagged = df["abstract"].apply(lambda el: 
        nltk.tag.pos_tag(nltk.tokenize.word_tokenize(el), tagset='universal'))
    lemma_tagged = df["abstract_lemma"].apply(lambda el: 
        nltk.tag.pos_tag(nltk.tokenize.word_tokenize(el), tagset='universal'))
    df["full_grams"] = full_tagged.apply(lambda el_list: [(el[0].lower(),el[1]) for el in el_list])
    df["lemma_grams"] = lemma_tagged.apply(lambda el_list: [(el[0].lower(),el[1]) for el in el_list])
    df["unigrams"] = df["full_grams"] + df["lemma_grams"]
    df["unigrams"] = df["unigrams"].apply(lambda el_list: [(el[0].lower(),el[1]) for el in el_list])
    df["bigrams"] =  df["unigrams"].apply(lambda el: list(nltk.bigrams(el)))
    df["trigrams"] = df["unigrams"].apply(lambda el: list(nltk.trigrams(el)))
    return df


def concat_tuples(tuple_list: list):
    """Converts a list of tuples into a list of joined strings.
    Args:
        tuple_list : list
            The tuple list being converted.
    Returns:
        list
            List of strings.
    """
    temp = [list(zip(*tup)) for tup in tuple_list]
    return [[" ".join(tup[0]), " ".join(tup[1])] for tup in temp]


def tag_extract_nouns(gram_list: list):
    """Returns all nouns within a list of grams.
    Args:
        gram_list : list
            List of n-gram strings.
    Returns:
        list
            List of nouns.
    """
    # Grab POS tags
    noun_list = []
    for word, tag in gram_list:
        if ("NOUN" in tag):
            noun_list.append(word)
    return noun_list


def tag_extract_nouns_plus(gram_list:list):
    """Returns nouns, verbs and adjectives from a list of grams.
    Args:
        gram_list : list
            List of n-gram strings.
    Returns:
        list
            List of nouns, adjectives and verbs.
    """
    # Grab POS tags
    noun_list = []
    for word, tag in gram_list:
        if ("NOUN" in tag or "ADJ" in tag or "VERB"):
            noun_list.append(word)
    return noun_list


def eliminate_non_significant_words(records: pd.DataFrame):
    """Removes non significant words (nouns, adjectives and verbs) from a DataFrame of records.
    Args:
        records : Pandas DataFrame
            DataFrame of biomimicry records.
    Returns:
        Tuple
            Pandas DataFrame
                DataFrame of biomimicry papers with only noun n-grams
            Pandas DataFrame
                DataFrame of biomimicry papers with n-grams containing verbs, nouns and adjectives.
    """
    non_pruned_records = ngram_records.copy()
    ngram_records["unigrams"] = ngram_records["unigrams"].apply(tag_extract_nouns)
    ngram_records["lemma_grams"] = ngram_records["lemma_grams"].apply(tag_extract_nouns)
    ngram_records["lemma_grams"] = ngram_records["lemma_grams"].apply(lambda el: " ".join(el))
    ngram_records["full_grams"] = ngram_records["full_grams"].apply(tag_extract_nouns)
    ngram_records["full_grams"] = ngram_records["full_grams"].apply(lambda el: " ".join(el))
    ngram_records["bigrams"] = ngram_records["bigrams"].apply(tag_extract_nouns)
    ngram_records["trigrams"] = ngram_records["trigrams"].apply(tag_extract_nouns)

    non_pruned_records["unigrams"] = non_pruned_records["unigrams"].apply(tag_extract_nouns)
    non_pruned_records["lemma_grams"] = non_pruned_records["lemma_grams"].apply(tag_extract_nouns)
    non_pruned_records["lemma_grams"] = non_pruned_records["lemma_grams"].apply(lambda el: " ".join(el))
    non_pruned_records["full_grams"] = non_pruned_records["full_grams"].apply(tag_extract_nouns)
    non_pruned_records["full_grams"] = non_pruned_records["full_grams"].apply(lambda el: " ".join(el))
    non_pruned_records["bigrams"] = non_pruned_records["bigrams"].apply(tag_extract_nouns_plus)
    non_pruned_records["trigrams"] = non_pruned_records["trigrams"].apply(tag_extract_nouns_plus)
    return (ngram_records, non_pruned_records)


def check_creature(ngram_list: list):
    """Checks a list of n-grams for creatures.
    Args:
        ngram_list : list
            List of n-grams.
    Returns:
        list
            List of n-grams which correspond with existing creatures.
    """
    matches = []
    for ngram in ngram_list:
        if ((ngram in common_names or ngram in species_dict) and ngram not in excluded_words):
            matches.append(ngram)
    return matches


def extract_species(records: pd.DataFrame):
    """Add species field to an input dataframe.
    Args:
        records : Pandas DataFrame
            DataFrame of biomimicry records.
    Returns:
        Pandas DataFrame
            DataFrame which has been appended with species data.
    """
    size = ngram_records.shape[0]
    count = 0
    for index, row in ngram_records.iterrows():
        count += 1
        print(f"{count/size:2%}% Completed", end='\r')
        records.at[index, "unigrams"] = check_creature(row["unigrams"])
        records.at[index, "bigrams"] = check_creature(row["bigrams"])
        records.at[index, "trigrams"] = check_creature(row["trigrams"])
    records["species"] = records["unigrams"] + records["bigrams"] + records["trigrams"]
    records["species"] = records["species"].apply(lambda el: list(set(el)))
    return records


def term_frequency(abstract: str, abstract_lemma: str, species: str):
    """Find the number of times a string appears in another
    Args:
        abstract : str
            Normalized abstract.
        abstract_lemma : str
            Normalized and lemmatized abstract.
        species : str
            Name associated with a species.
    Returns:
        Tuple
            str
                Number of times species was found
            bool
                True means the species was found more in the full abstract than in the 
                lemmatized abstract and vice versa
    """
    num_full_matches = abstract.count(species)
    num_lemma_matches = abstract_lemma.count(species)
    if num_full_matches > num_lemma_matches:
        return (num_full_matches, True)
    else:
        return (num_lemma_matches, False)


def add_relevancy(records: pd.DataFrame):
    """Attach a field corresponding to the relevancy of each found species to a dataframe
    Args:
        records : Pandas DataFrame
            DataFrame of biomimicry records.
    Returns:
        Pandas DataFrame
            DataFrame which has been appended with relevancy data.
    """
    paper_relevancy_scores = []
    species_relevancy_scores = []
    for index, row in records.iterrows():

        record_scores = []
        normalized_scores = []
        lemma_size = len(row["lemma_grams"].split(" "))
        full_noun_size = len(row["full_grams"].split(" "))
        for species in row["species"]:
            spec_score, used_full = term_frequency(row["full_grams"],row["lemma_grams"], species)
            record_scores.append(spec_score)

        if (len(record_scores) > 0):
            record_scores = [score/full_noun_size if used_full 
                else score/lemma_size for score in record_scores]
            max_relevancy = max(record_scores)
            normalized_scores = [relevance/max_relevancy if max_relevancy > 0 else 0 for relevance in record_scores]

        paper_relevancy_scores.append(record_scores)
        species_relevancy_scores.append(normalized_scores)

    records["absolute_relevancy"] = paper_relevancy_scores
    records["relative_relevancy"] = species_relevancy_scores

    return records


def export_csv(output_name: str, records: pd.DataFrame):
    """Exports the dataframe in either a CSV or JSON list format.
    Args:
        output_name: str
            Name of the output file with extension.
        records : Pandas DataFrame
            DataFrame of biomimicry records.
    """
    final_records = records.drop(["unigrams","bigrams","trigrams", "lemma_grams", "full_grams"], axis=1)
 
    if ("csv" in output_name.lower()):
        final_records.to_csv(f"{output_name}", index="ignore")

    elif ("json" in output_name.lower()):
        with open(f"{output_name}", "w") as file:
            file.write("[\n")
            golden_size = records.shape[0]

            for index, row in records.iterrows():
                file.write("\t")
                file.write(json.dumps(row.to_dict()))
                if(index < golden_size - 1):
                    file.write(",\n")
            file.write("\n]")


if __name__ == "__main__":
    args = get_arg_parser()
    print("(1/7) Reading data")
    petal_records, species_dict, common_names, excluded_words = read_data(args.file_path)
    processed_records = petal_records.copy(deep=True)
    processed_records.fillna("", inplace=True)
    print("(2/7) Preprocessing Abstracts")
    processed_records["abstract"] = processed_records["abstract"].apply(strip_accents)
    processed_records["abstract_lemma"] = processed_records["abstract"].apply(lemmatize_paragraph)
    print("(3/7) Creating N-Grams")
    ngram_records = create_tagged_n_grams(processed_records)
    ngram_records["bigrams"] = ngram_records["bigrams"].apply(concat_tuples)
    ngram_records["trigrams"] = ngram_records["trigrams"].apply(concat_tuples)
    print("(4/7) Removing insignificant words")
    ngram_records, non_pruned_records = eliminate_non_significant_words(ngram_records)
    print("(5/7) Extracting species")
    ngram_records = extract_species(ngram_records)
    non_pruned_records = extract_species(non_pruned_records)
    print("(6/7) Ranking labels\n")
    ngram_records = add_relevancy(ngram_records)
    non_pruned_records = add_relevancy(non_pruned_records)
    print("(7/7) Exporting data")
    export_csv(args.output_name, ngram_records)
    export_csv("lenient_" + args.output_name, non_pruned_records)

    



