# Source: PyTorch Tutorial (http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

# Data for NGramLangModel -
test_sentence = """"When forty winters shall besiege thy brow, And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now, Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies, Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes, Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use, If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'Proving his beauty by succession thine!
This were to be new made when thou art old, And see thy blood warm when thou feel'st it cold."""

# Data for CBoW
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""

# Data for PosTagger
training_data = [("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])]