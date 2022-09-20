import logging

import pandas as pd

from src.services.csv_service import get_classification_data
from src.services.sklearn_service import SklearnService

logger = logging.getLogger(__name__)


class Test:
    sklearn_service = SklearnService()

    def test(self):
        df: pd.DataFrame = get_classification_data()
        df.to_csv(f'./{temperature_name}_{humidity_name}.csv', index=False)

        X = df.drop('is_comfortable', axis=1)
        y = df['is_comfortable']

        self.sklearn_service.fit(X=X, y=y)

        y_pred = self.sklearn_service.predict(X)
        y_true = y
        print(f'{self.sklearn_service.get_accuracy_score(y_true, y_pred)=}')

        test_data = [[20, 70], [30, 40], [25, 50], [28, 60]]
        test_label = self.sklearn_service.model.predict(test_data)
        logger.info(test_label)

        test_data = [[24.8, 44.8], [25.1, 45.1]]
        test_label = self.sklearn_service.model.predict(test_data)
        logger.info(test_label)


if __name__ == '__main__':
    test = Test()
    test.test()
