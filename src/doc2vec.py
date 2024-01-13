from gensim.models import Doc2Vec


def train_doc2vec(tagged_data):
    # Set the number of workers to utilize multiple CPU cores
    # Adjust `workers` based on the available CPU cores
    model = Doc2Vec(vector_size=300,  # Increased vector size
                    window=5,         # Context window size
                    min_count=3,      # Ignore words with frequency less than 5
                    dm=1,             # Using PV-DM
                    epochs=400,       # More epochs
                    alpha=0.025,      # Initial learning rate
                    min_alpha=0.00025,# Minimum learning rate
                    sample=1e-5,      # Downsample setting for frequent words
                    negative=5,
                    hs=1,
                    workers=4,        # Set the number of workers for parallel processing
                   )
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model_path = "../models/doc2vec/doc2vec_model.bin"
    model.save(model_path)
    return model

