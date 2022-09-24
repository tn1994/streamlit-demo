import logging
from datetime import datetime, timedelta, date

import boto3

logger = logging.getLogger(__name__)


class AWSService:
    ce: boto3.client = None
    s3: boto3.client = None

    def __init__(self,
                 aws_access_key_id: str,
                 aws_secret_access_key: str,
                 region_name: str
                 ):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key: str = aws_secret_access_key
        self.region_name: str = region_name

    def _get_client(self, service_name: str):
        return boto3.client(service_name,
                            aws_access_key_id=self.aws_access_key_id,
                            aws_secret_access_key=self.aws_secret_access_key,
                            region_name=self.region_name
                            )

    def get_total_billing(self) -> dict:
        """合計請求額取得
        ref: https://qiita.com/Nidhog-tm/items/e602bb5d3a4950f92dfa
        :return:
        """
        if self.ce is None:
            self.ce = self._get_client(service_name='ce')
        billing = Billing(ce=self.ce)
        return billing.get_total_billing()

    def get_service_billings(self) -> list:
        """各サービスの詳細請求金額取得
        ref: https://qiita.com/Nidhog-tm/items/e602bb5d3a4950f92dfa
        :return:
        """
        if self.ce is None:
            self.ce = self._get_client(service_name='ce')
        billing = Billing(ce=self.ce)
        return billing.get_service_billings()

    def get_list_buckets(self):
        if self.s3 is None:
            self.s3 = self._get_client(service_name='s3')
        s3 = S3(s3=self.s3)
        return s3.get_list_buckets()


class S3:

    def __init__(self, s3: boto3.client):
        self.s3 = s3

    def get_list_buckets(self):
        return self.s3.list_buckets()


class Billing:

    def __init__(self, ce: boto3.client):
        self.ce: boto3.client = ce

    def get_total_billing(self) -> dict:
        """合計請求額取得
        ref: https://qiita.com/Nidhog-tm/items/e602bb5d3a4950f92dfa
        :return:
        """
        (start_date, end_date) = get_total_cost_date_range()
        response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='MONTHLY',
            Metrics=[
                'AmortizedCost'
            ]
        )

        return {
            'start': response['ResultsByTime'][0]['TimePeriod']['Start'],
            'end': response['ResultsByTime'][0]['TimePeriod']['End'],
            'billing': response['ResultsByTime'][0]['Total']['AmortizedCost']['Amount'],
        }

    def get_service_billings(self) -> list:
        """各サービスの詳細請求金額取得
        ref: https://qiita.com/Nidhog-tm/items/e602bb5d3a4950f92dfa
        :return:
        """
        (start_date, end_date) = get_total_cost_date_range()
        response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='MONTHLY',
            Metrics=[
                'AmortizedCost'
            ],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                }
            ]
        )

        billings = []

        for item in response['ResultsByTime'][0]['Groups']:
            billings.append({
                'service_name': item['Keys'][0],
                'billing': item['Metrics']['AmortizedCost']['Amount']
            })
        return billings


def get_begin_of_month() -> str:
    """実行月の1日を取得
    ref: https://qiita.com/Nidhog-tm/items/e602bb5d3a4950f92dfa
    :return:
    """
    return date.today().replace(day=1).isoformat()


def get_today() -> str:
    """実行日を取得
    ref: https://qiita.com/Nidhog-tm/items/e602bb5d3a4950f92dfa
    :return:
    """
    return date.today().isoformat()


def get_total_cost_date_range() -> (str, str):
    """
    ref: https://qiita.com/Nidhog-tm/items/e602bb5d3a4950f92dfa
    :return:
    """
    start_date = get_begin_of_month()
    end_date = get_today()

    # get_cost_and_usage()のstartとendに同じ日付は指定不可のため、
    # 「今日が1日」なら、「先月1日から今月1日（今日）」までの範囲にする
    if start_date == end_date:
        end_of_month = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=-1)
        begin_of_month = end_of_month.replace(day=1)
        return begin_of_month.date().isoformat(), end_date
    return start_date, end_date
