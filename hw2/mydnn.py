class mydnn():
    def __init__(self, architecture, loss, weight_decay=0):
        self.prepare_dictionaries()  # TODO: this should create the string->func dict of
                                    # activation/regularization/loss from the utils
        self._architecture = architecture
        self._loss = loss
        self._weight_decay = weight_decay
