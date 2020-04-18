from collections.abc import Iterable


class Transform:
    """
    Usage:
        t = Transform(y_train)
        y_train = t.fit(y_train)
        y_test = t.fit(y_test)
        (train model)
        ...

        y_predict = t.decode(y_predict)
    """
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
        if isinstance(y_test, Iterable):
            transformed_y_test = []
            for y in y_test:
                if y is None:
                    transformed_y_test.append(None)
                transformed_y = self.offsets[round(y) if y < 9 else 8] + self.K[round(y) if y < 9 else 8] * (y -
                                                                                                            round(y))
                transformed_y_test.append(transformed_y)
            return transformed_y_test
        else:
            if y_test is None:
                return None
            return self.offsets[round(y_test) if y_test < 9 else 8] + self.K[round(y_test) if y_test < 9 else 8] * (y_test -
                                                                                                            round(y_test))

    def __decode_value__(self, value):
        if value == 9:
            return 9
        if value == 0:
            return 0
        if value is None:
            return None
        k = 0
        for offset in self.offsets:
            if value < offset:
                break
            k += 1
        k -= 1
        original_value = k + (value - self.offsets[k]) / self.K[k]
        return original_value

    def decode(self, Y):
        if isinstance(Y, Iterable):
            return [self.__decode_value__(y) for y in Y]
        return self.__decode_value__(Y)
