from nltk import DependencyGraph
import re


def html_ify(s):
    '''
        Takes care of &quot &lsqb &rsqb &#39
    '''
    html_string = re.sub(r'\)', r'&rcrb;', s)
    html_string = re.sub(r'\(', r'&lcrb;', html_string)
    return html_string


files = [('en-ud-train.conllu', 'trees-train.tsv'),
         ('en-ud-dev.conllu', 'trees-dev.tsv'),
         ('en-ud-test.conllu', 'trees-test.tsv')]
structures = []
for file in files:
    with open(file[0], 'r') as f:
        with open(file[1], 'w') as fout:
            id = 0
            a = ""
            words = []
            for line in f:
                if line != "\n":
                    a += line
                    words.append(line.split("\t")[1])
                else:
                    id += 1
                    a = html_ify(a)
                    structure = DependencyGraph(a, top_relation_label='root')
                    sent = " ".join(words)
                    sent = html_ify(sent)
                    sent_id = file[0] + " sent_" + str(id)
                    structures.append(structure)
                    a = ""
                    words = []
                    fout.write(sent_id + "\t" + " ".join(str(structures[-1].tree()).splitlines()) + "\t" + sent + "\n")
