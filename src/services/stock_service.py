import logging
import datetime
from datetime import timedelta

import jpholiday
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

FIGSIZE = (16, 8)  # other:(24, 16) or (32, 16)


class StockDF:
    def add(self, name, df):
        super().__setattr__(name, df)

    def get(self, name):
        return getattr(self, name)

    def check(self, name):
        df = getattr(self, name)
        macd = self.check_macd(df)
        rsi9 = self.check_rsi(df)
        rci = self.checkpoint_rci(df)
        return macd, rsi9, rci

    def check_macd(self, df):
        MACD_SIGNAL: str = 'macdsignal'
        get_cross_macd(df['macd'], df['macdsignal'])
        index_num_macdsignal = df.columns.get_loc(MACD_SIGNAL)
        macd = df.iloc[get_cross_macd(df['macd'], df['macdsignal']), index_num_macdsignal]
        return macd

    def check_rsi(self, df):
        RSI9: str = 'rsi9'
        index_num_rsi9 = df.columns.get_loc(RSI9)
        rsi9 = df.loc[df[RSI9] <= 30, RSI9]
        return rsi9

    def checkpoint_rci(self, df, is_down: bool = False, threshold: int = -75):
        cross = df['rci26'] < threshold
        golden = (cross != cross.shift(1)) & (cross == is_down)

        # ゴールデンクロス・デッドクロスの発生位置（要素番号）をリストに格納
        index_checkpoint = [i for i, x in enumerate(golden) if x == True]
        rci = df.iloc[index_checkpoint]
        return rci


class StockService:
    today = str(datetime.date.today())
    tweet_date_now = today.replace('-', '')  # 日付は８桁文字列の形式
    stock_df = StockDF()

    ticker_list = ['v', 'nke', 'cost', 'lmt', 'mkc', 'hd', 'unp', 'unh', 'dhr', 'chd']

    result_signal_df: pd.DataFrame = None

    def __init__(self):
        self.TODAY = self.returnBizDay(self.tweet_date_now)

        tdatetime = datetime.datetime.strptime(self.TODAY, "%Y-%m-%d")
        tdate = datetime.date(tdatetime.year, tdatetime.month, tdatetime.day)
        YESTERDAY = str(tdate - timedelta(1))
        self.ONEWEEKAGO = str(tdate - timedelta(5))

    def returnBizDay(self, DATE):
        Date = datetime.date(int(DATE[0:4]), int(DATE[4:6]), int(DATE[6:8]))
        while Date.weekday() >= 5 or jpholiday.is_holiday(Date):
            Date = datetime.date(Date.year, Date.month, Date.day - 1)
        return Date.strftime("%Y-%m-%d")

    def get_stock(self, code, start_date, end_date):
        df = web.DataReader(code, data_source='stooq', start=start_date, end=end_date)
        df = df.iloc[::-1]
        return df

    def _backup_df(self, code, start_date, end_date):
        # 株価の取得(銘柄コード, 開始日, 終了日)
        df = self.get_stock(code, start_date, end_date)  # Open,High,Low,Close,Volum

        # backup
        self.stock_df.add(name=code, df=df)
        df = self.stock_df.get(name=code)
        return df

    def signal_check_main(self, ticker: str = None, start_date='2020-01-01', end_date=None, how_date=None):
        if ticker is not None:
            ticker_list = [ticker]
        else:
            ticker_list = self.ticker_list

        if end_date is None:
            end_date = self.TODAY
        if how_date is None:
            how_date = self.ONEWEEKAGO
        if start_date > end_date:
            raise ValueError(f'must start_date < end_date,'
                             f' but got {start_date=} > {end_date=}')

        for ticker in ticker_list:
            df = self._backup_df(code=ticker, start_date=start_date, end_date=self.TODAY)
            show_rci(df=df, close_list=df['Close'], code=ticker)
            show_rsi(df=df, close=df['Close'], code=ticker, start_date=start_date, end_date=self.TODAY)
            show_macd(df=df, close=df['Close'], code=ticker)
            tmp_macd, tmp_rsi9, tmp_rci = self.stock_df.check(name=ticker)
            self.show_signal(code=ticker, tmp_macd=tmp_macd, tmp_rsi9=tmp_rsi9, tmp_rci=tmp_rci, how_date=how_date)

    def main(self, ticker: str = None, start_date=None, how_date=None):
        if start_date is None:
            start_date = '2020-01-01'
            # start_date = '2008-04-01'
        if how_date is None:
            how_date = self.ONEWEEKAGO

        if ticker is None:
            for code in self.ticker_list:
                self._main(code, start_date, self.TODAY, is_5_and_25=True, is_25_and_75=False)
        elif isinstance(ticker, str):
            self._main(code=ticker, start_date=start_date, end_date=self.TODAY, is_5_and_25=True, is_25_and_75=False)
            tmp_macd, tmp_rsi9, tmp_rci = self.stock_df.check(ticker)
            self.show_signal(code=ticker, tmp_macd=tmp_macd, tmp_rsi9=tmp_rsi9, tmp_rci=tmp_rci, how_date=how_date)
        else:
            raise ValueError

    def _main(self, code, start_date, end_date, is_5_and_25=False, is_25_and_75=False):
        if not is_5_and_25 and not is_25_and_75:
            raise ValueError

        df = self._backup_df(code, start_date, end_date)

        # データフレームの作成
        # df = pd.DataFrame({'始値':data[1], '終値':data[2], '高値':data[3], '安値':data[4]}, index = data[0])

        # 移動平均線の計算
        ma_5d = df['Close'].rolling(window=5).mean()
        ma_25d = df['Close'].rolling(window=25).mean()
        ma_75d = df['Close'].rolling(window=75).mean()

        # データフレームの列に移動平均線を追加
        if is_5_and_25:
            df['移動平均線(5日)'] = ma_5d
            df['移動平均線(25日)'] = ma_25d
        elif is_25_and_75:
            df['移動平均線(25日)'] = ma_25d
            df['移動平均線(75日)'] = ma_75d
        else:
            raise ValueError

        # 移動平均のクロス確認
        if is_5_and_25:
            cross = ma_5d > ma_25d
        elif is_25_and_75:
            cross = ma_25d > ma_75d
        else:
            raise ValueError
        golden = (cross != cross.shift(1)) & (cross == True)
        dead = (cross != cross.shift(1)) & (cross == False)

        # ゴールデンクロス・デッドクロスの発生位置（要素番号）をリストに格納
        index_g = [i for i, x in enumerate(golden) if x == True]
        index_d = [i for i, x in enumerate(dead) if x == True]

        # グラフにプロット
        self.fig = plt.figure(figsize=FIGSIZE)
        ax = df['Close'].plot(color="blue", label="Close", title=code)
        if is_5_and_25:
            ma_5d.plot(ax=ax, ls="--", color="green", label="MA 5d")
        elif is_25_and_75:
            ma_75d.plot(ax=ax, ls="--", color="green", label="MA 75d")
        ma_25d.plot(ax=ax, ls="--", color="red", label="MA 25d")
        df.iloc[index_g, 5].plot(ax=ax, ls='', marker='^', ms='10', color="green", label="Golden cross")
        df.iloc[index_d, 5].plot(ax=ax, ls='', marker='v', ms='10', color="red", label="Dead cross")
        ax.grid()
        ax.legend()
        plt.show()

        self.rci_fig = show_rci(df=df, close_list=df['Close'], code=code)
        self.rsi_fig = show_rsi(df=df, close=df['Close'], code=code, start_date=start_date, end_date=end_date)
        self.macd_fig = show_macd(df=df, close=df['Close'], code=code)

    def show_signal(self, code, tmp_macd, tmp_rsi9, tmp_rci, how_date):
        if not isinstance(how_date, str):
            how_date = str(how_date)
        signal = signal_rci = signal_rsi = signal_macd = False
        if any(tmp_rsi9.index > how_date) and any(tmp_macd.index > how_date) and any(
                tmp_rci.index > how_date):
            signal = True
        if any(tmp_rci.index > how_date):
            signal_rci = True
        if any(tmp_rsi9.index > how_date):
            signal_rsi = True
        if any(tmp_macd.index > how_date):
            signal_macd = True

        _result_signal_df: pd.DataFrame = pd.DataFrame({
            'ticker': [code],
            'all_signal': [signal],
            'signal_rci': [signal_rci],
            'signal_rsi': [signal_rsi],
            'signal_macd': [signal_macd]
        })

        if self.result_signal_df is None:
            self.result_signal_df: pd.DataFrame = _result_signal_df
        else:
            self.result_signal_df = pd.concat([self.result_signal_df, _result_signal_df], ignore_index=True)

    def get_result_signal_df(self):
        return self.result_signal_df.style.applymap(color_survived,
                                                    subset=['all_signal', 'signal_rci',
                                                            'signal_rsi', 'signal_macd'])


def compute_rci(close_list, interval=14):
    """calc rci
    ref: https://www.tcom242242.net/entry/fx-begin/compute-rci-by-python/
    """
    rci_list = [None for _ in range(interval)]

    nb_close = len(close_list)
    for idx in range(nb_close):
        if (idx + interval > nb_close):
            break

        y = close_list[idx:idx + interval]
        x_rank = np.arange(len(y))
        y_rank = rankdata(y, method='ordinal') - 1
        sum_diff = sum((x_rank - y_rank) ** 2)
        rci = (1 - ((6 * sum_diff) / (interval ** 3 - interval))) * 100
        rci_list.append(rci)
    return rci_list


def checkpoint_rci(rci_list: list, is_down: bool = False, threshold: int = -80):
    if not isinstance(rci_list, pd.Series):
        cross = pd.Series(rci_list)
    else:
        cross = rci_list
    cross = cross < threshold
    golden = (cross != cross.shift(1)) & (cross == is_down)

    # get golden_cross index, and add list
    index_checkpoint = [i for i, x in enumerate(golden) if x == True]
    return index_checkpoint


def show_rci(df, close_list, code):
    """
    ref: https://www.tcom242242.net/entry/fx-begin/compute-rci-by-python/
    :param df:
    :param close_list:
    :param code:
    :return:
    """
    RCI26: str = 'rci26'
    RCI52: str = 'rci52'

    nb_display: int = len(close_list)

    # rciを計算する
    df[RCI26] = compute_rci(close_list, interval=26)[-nb_display:]  # 中期
    df[RCI52] = compute_rci(close_list, interval=52)[-nb_display:]  # 長期

    # columnのindex取得
    index_num_rci26 = df.columns.get_loc(RCI26)
    index_num_rci52 = df.columns.get_loc(RCI52)

    fig = plt.figure(figsize=FIGSIZE)
    ax = df[RCI26].plot(color="blue", label=RCI26, title=f'{code}_rci')
    df['rci52'].plot(ax=ax, color="red", label="rci52")

    df.iloc[checkpoint_rci(df[RCI26]), index_num_rci26].plot(ax=ax, ls='', marker='^', ms='10', color="blue",
                                                             label="checkpoint 26")
    df.iloc[checkpoint_rci(df[RCI26], is_down=True), index_num_rci26].plot(ax=ax, ls='', marker='v', ms='10',
                                                                           color="blue", label="checkpoint 26")
    df.iloc[checkpoint_rci(df[RCI52]), index_num_rci52].plot(ax=ax, ls='', marker='^', ms='10', color="red",
                                                             label="checkpoint 52")
    df.iloc[checkpoint_rci(df[RCI52], is_down=True), index_num_rci52].plot(ax=ax, ls='', marker='v', ms='10',
                                                                           color="red", label="checkpoint 52")
    ax.grid()
    ax.legend()
    plt.show()
    return fig


def get_cross_macd(macd, macdsignal):
    cross = macd > macdsignal
    golden = (cross != cross.shift(1)) & (cross == True)
    # dead   = (cross != cross.shift(1)) & (cross == False) # dead_cross

    # get golden_cross and dead_cross index, and add list
    index_g = [i for i, x in enumerate(golden) if x == True]
    # index_d = [i for i, x in enumerate(dead) if x == True]  # dead_cross
    return index_g


def show_rsi(df, close, code, start_date, end_date):
    """
    ref: https://myfrankblog.com/technical_analysis_with_python/
    :param df:
    :param close:
    :param code:
    :param start_date:
    :param end_date:
    :return:
    """
    # 前日との差分を計算
    df_diff = close.diff(1)

    calc_result: list = []
    for timeperiod in [9, 14]:
        # 計算用のDataFrameを定義
        df_up, df_down = df_diff.copy(), df_diff.copy()
        df_up[df_up < 0] = 0  # df_upはマイナス値を0に変換
        df_down[df_down > 0] = 0  # df_downはプラス値を0に変換して正負反転
        df_down = df_down * -1

        # 期間14でそれぞれの平均を算出
        df_up_sma14 = df_up.rolling(window=timeperiod, center=False).mean()
        df_down_sma14 = df_down.rolling(window=timeperiod, center=False).mean()

        # RSIを算出
        calc_result.append(100.0 * (df_up_sma14 / (df_up_sma14 + df_down_sma14)))

    df['rsi9'], df['rsi14'] = calc_result[0], calc_result[1]

    fig = plt.figure(figsize=FIGSIZE)
    plt.title(f'{code}_rsi')
    plt.plot(df.rsi9, label='rsi9')
    plt.plot(df.rsi14, label='rsi14')
    plt.xlabel('index')
    plt.ylabel('indicator value')
    plt.legend()
    plt.grid()
    plt.show()
    return fig


def show_macd(df, close, code):
    """
    ref: https://myfrankblog.com/technical_analysis_with_python/
    :param df:
    :param close:
    :param code:
    :return:
    """
    MACD: str = 'macd'
    MACD_SIGNAL: str = 'macdsignal'
    FastEMA_period = 12  # 短期EMAの期間
    SlowEMA_period = 26  # 長期EMAの期間
    SignalSMA_period = 9  # SMAを取る期間

    df["macd"] = close.ewm(span=FastEMA_period).mean() - close.ewm(span=SlowEMA_period).mean()
    df["macdsignal"] = df["macd"].rolling(SignalSMA_period).mean()
    df['macdhist'] = df.macd - df.macdsignal

    index_num_macd = df.columns.get_loc(MACD)
    index_num_macdsignal = df.columns.get_loc(MACD_SIGNAL)

    fig = plt.figure(figsize=FIGSIZE)
    ax = df['macd'].plot(color="blue", label='macd', title=f'{code}_macd')
    df['macdsignal'].plot(ax=ax, color="red", label="macdsignal")
    plt.bar(df.macdhist.index, df.macdhist, color='pink', label='macdhist')
    df.iloc[get_cross_macd(df['macd'], df['macdsignal']), index_num_macdsignal].plot(ax=ax, ls='', marker='^', ms='10',
                                                                                     color="blue", label="macdcross")

    ax.grid()
    ax.legend()
    plt.show()
    return fig


def color_survived(val):
    """
    ref: https://discuss.streamlit.io/t/change-background-color-based-on-value/2614/6
    :param val:
    :return:
    """
    if val is True:
        color: str = 'green'
        return f'background-color: {color}'
