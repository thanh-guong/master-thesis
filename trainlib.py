import keras


def train(filepath, model, x_train, y_train, x_test, y_test, batch_size, epochs, early_stopping_patience=5, validation_split=0.1):
    # perform training ...
    #   - call the main training loop in keras for our network+dataset

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[
            keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=0, mode='auto')
        ])

    # we re-load the best weights once training is finished
    model.load_weights(filepath)

    return history, model


def train_double_input(filepath, model, iq_train, transformed_train, y_train, batch_size, epochs, early_stopping_patience=5, validation_split=0.1):
    # perform training ...
    #   - call the main training loop in keras for our network+dataset

    history = model.fit(
        (iq_train, transformed_train),
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[
            keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=0, mode='auto')
        ])

    # we re-load the best weights once training is finished
    model.load_weights(filepath)

    return history, model
