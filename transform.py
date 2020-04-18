from collections.abc import Iterable


class Transform:
    def __init__(self, y_train):
        counts = [0 for i in range(9)]
        for y in y_train:
            if y is None:
                continue
            if y == 9:
                counts[8] += 1
            else:
                counts[round(y)] += 1
        total_count = sum(counts)
        K = [9 * counts[i] / total_count for i in range(9)]
        offsets = [sum([K[j] for j in range(i)]) for i in range(9)]
        self.offsets = offsets.copy()
        self.K = K.copy()
        transformed_y_train = []
        for y in y_train:
            if y is None:
                transformed_y_train.append(None)
            transformed_y = offsets[round(y) if y < 9 else 8] + K[round(y) if y < 9 else 8] * (y - round(y))
            transformed_y_train.append(transformed_y)

    def fit(self, y_test):
        transformed_y_test = []
        if isinstance(y_test, Iterable):
            for y in y_test:
                if y is None:
                    transformed_y_test.append(None)
                transformed_y = self.offsets[round(y) if y < 9 else 8] + self.K[round(y) if y < 9 else 8] * (y -
                                                                                                            round(y))
                transformed_y_test.append(transformed_y)
        return transformed_y_test
    
