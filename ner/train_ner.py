import spacy

nlp = spacy.blank("en")

ner = nlp.add_pipe("ner")

ner.add_label("ORG")
ner.add_label("DATE")
ner.add_label("MONEY")
