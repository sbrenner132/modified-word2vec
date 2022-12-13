from training import training_setup, train_model

if __name__ == "__main__":
    model, indices, ctxs, vocab_size = training_setup(use_cooccurrence_matrix=True, seed_random=True)
    train_model(model, indices, ctxs, vocab_size, 15000, "log_train_cooccurrence_book7.txt", "training_cooccurrence_book7")
