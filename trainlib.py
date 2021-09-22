import keras


def train(filepath, model, x_train, y_train, x_test, y_test, batch_size, epochs, early_stopping_patience=5):
    # perform training ...
    #   - call the main training loop in keras for our network+dataset

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[
            keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=0, mode='auto')
        ])

    # we re-load the best weights once training is finished
    model.load_weights(filepath)

    return history, model
