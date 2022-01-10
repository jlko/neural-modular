class EarlyStop:
    def __init__(self, model, max_patience):
        self.model = model
        self.max_patience = max_patience
        self.best = 100000
        self.patience = 0
        self.best_model = model

    def update(self, val_nll):
        # disable early stop
        if self.max_patience == -1:
            return False

        # update current best nll and model, reset patience
        if val_nll < best:
            self.best = val_nll
            self.best_model = copy.deepcopy(self.model)
            self.patience = 0
        else:
            self.patience += 1

        # trigger early stopping
        if self.patience > self.max_patience:
            return True

        return False
