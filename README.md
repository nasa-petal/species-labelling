# Overview
This repository contains species related data from databases such as NCBI Blast and PubMed. This data is used to extract species information from the abstracts a given set of papers.

## How to Run
The species_tagger.py file takes two positional arguments: the file path and output name + extension. This file is meant to be run on the PeTaL biomimicry data set format but can intake either JSON or CSV files and output them in either a CSV or JSON format.

## Important Files
- common_names.txt
  - Text file consisting of a set of species common names.

- excluded_words.txt
    - Words to be ignored in abstracts when mining for species information.

- species_dict.json
    - Dictionary where keys are species scientific names and values are a list of the corresponding common names.

- ncbi_species.zip
    - Zipped JSON list of various formatted species information from NCBI Blast.

- species_tagger.py
    - Script responsible for ingesting a PeTaL document data set and returning a modified version with species and relevancy fields.

