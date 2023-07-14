# WikiBio @ ACL 2023

This is the repository of the WikiBio corpus, which is described in the following paper:

> Wikibio: a Semantic Resource for the Intersectional Analysis of Biographical Events

Please use this reference to cite our work
> @article{stranisci2023wikibio,
>  title={Wikibio: a Semantic Resource for the Intersectional Analysis of Biographical Events},
> author={Stranisci, Marco Antonio and Damiano, Rossana and Mensa, Enrico and Patti, Viviana and Radicioni, Daniele and Caselli, Tommaso},
> journal={arXiv preprint arXiv:2306.09505},
> year={2023}}

# Data Statement for WikiBio corpus

Data set name: WikiBio

## A. CURATION RATIONALE

WikiBio is a Corpus of 20 writers' biographies annotated for the biographical event detection task.

- Data Collection Process. All the annotated documents were gathered from the English Wikipedia. In order to retain only relevant document we relied on Wikidata for the collection of all the entities having as occupation "writer", "novelist", or "poet", born in an African country or being African-American. We then selected only ones belonging to the Silent Generation (born between 1928 and 1945). From that collection we sampled 10 African writers and 10 African American Writers whose biographies' length were higher than 200.000 tokens. 
- Time Period. All documents have been gathered from English Wikipedia in Winter 2021.

## B. LANGUAGE VARIETY

- BCP-47 language tag: en 
- Language variety description: English

## C. SPEAKER DEMOGRAPHIC 

N/A

## D. ANNOTATOR DEMOGRAPHIC

Annotator #1: Age: 38; Gender: male; Race/ethnicity: caucasian; Native language: Italian; Socioeconomic status:n/a Training in linguistics/other relevant discipline: PhD student in Computer Science.

Annotator #2: Age: 50; Gender: female; Race/ethnicity: caucasian; Native language: Italian; Socioeconomic status:n/a Training in linguistics/other relevant discipline: Associate Professor in Computer Science

Annotator #3: Age: 30; Gender: male; Race/ethnicity: caucasian; Native language: Italian; Socioeconomic status:n/a Training in linguistics/other relevant discipline: Researcher in Computer Science

All annotators are near-native speakers of British English, having a long experience in annotating data for the specific task (event and entity detection).
## E. SPEECH SITUATION

N/A

## F. TEXT CHARACTERISTICS
Wikipedia documents

## G. ANNOTATORS' COMPENSATION

Annotators' activity is part of their effort related to the development of the present work, which was economically recognized within their contracts with the Academic Institution they are working for. 

# Beware of the homonyms
On Github there is a folder with the same name, but it is related to another work 
> Neural Text Generation from Structured Data with Application to the Biography Domain
> Rémi Lebret, David Grangier and Michael Auli, EMNLP 16,
> 
If you are looking for it, you can find it [here](https://github.com/DavidGrangier/wikipedia-biography-dataset).

