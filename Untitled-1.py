# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2023/5/11 20:49
describe:
"""
import sys
sys.path.insert(0, '../..')
from datetime import date,timedelta
import os
os.environ['czsc_max_bi_num'] = '20'
os.environ['data_path'] = "/home/data/Insight"
os.environ['signals_module_name'] = 'czsc.signals'
import czsc
import inspect
import streamlit as st
import pandas as pd
from loguru import logger
from typing import List
from copy import deepcopy
from czsc.utils.bar_generator import freq_end_time
from czsc.connectors.insight_connector import get_symbols, get_raw_bars
from czsc import CzscStrategyBase, Position, CzscTrader, KlineChart, Freq, Operate

from insight_python.com.interface.mdc_gateway_base_define import GateWayServerConfig
from insight_python.com.insight import common, subscribe
from insight_python.com.insight.subscribe import *
from insight_python.com.insight.market_service import market_service
# from streamlit.scriptrunner.script_run_context import get_script_run_ctx
#v1.12
# from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx,add_script_run_ctx
# from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner import add_script_run_ctx

import streamlit
import threading



trader=None

st.set_page_config(layout="wide")


# 可以直接导入策略类，也可以直接写在这里
from czsc.strategies import CzscStrategyExample2 as Strategy

# 以下代码不用修改
# ----------------------------------------------------------------------------------------------------------------------
if 'bar_edt_index' not in st.session_state:
    st.session_state.bar_edt_index = 1

if 'run' not in st.session_state:
    st.session_state.run = False

if 'date_change' not in st.session_state:
    st.session_state.date_change = True
if 'subscribe' not in st.session_state:
    st.session_state.subscribe=False


st.session_state.symbol=None

def update_date_change():
    st.session_state.date_change = 0


@st.cache_data
def get_bars(symbol_, base_freq_, sdt_, edt_,fq_):

    return get_raw_bars(symbol_, base_freq_, sdt=sdt_ - pd.Timedelta(days=365 * 3), edt=edt_)

def showtd():
    with st.empty():
        show_trader(trader)


with st.sidebar:
    st.title("CZSC实盘")
    m1, m2 = st.columns([10, 10])
    with m1:
        fq=st.selectbox("复权",['不复权','前复权','后复权'],index=0,disabled=True)
    with m2:
        asset=st.selectbox('品种',['ALL']+os.listdir(os.path.join(os.environ['data_path'],fq)),index=0,on_change=update_date_change)
    symbol = st.selectbox("选择标的：", get_symbols(asset,fq=fq), index=0, on_change=update_date_change)
    st.session_state.symbol=symbol
    # sdt = st.date_input("3年前日期：", value=date.today()-timedelta(days=365*3), on_change=update_date_change)
    # edt = st.date_input("昨天日期：", value=date.today()-timedelta(days=1), on_change=update_date_change)
    czsc_strategy = Strategy
    tactic: CzscStrategyBase = czsc_strategy(symbol=symbol, signals_module_name=os.environ['signals_module_name'])

    if st.session_state.date_change:
        bars = get_bars(symbol, tactic.base_freq, date.today()-timedelta(days=365*3), date.today(),fq)
        if not  bars[-1].dt.date==timedelta(days=1):
            st.warning(f"{symbol}的历史数据不是上个交易日的数据{bars[-1].dt}")
        bg, bars_right = tactic.init_bar_generator(bars, sdt=date.today()-timedelta(days=365*3))
        if len(bars_right) > 10000:
            bars_right = bars_right[-10000:]  # 限制回放的数据量
            st.warning("数据量过大，已限制为10000根K线")
        bars_num = len(bars_right)
    else:
        st.session_state.date_change = False



def show_trader(trader: CzscTrader):
    freqs = trader.freqs
    tabs = st.tabs(freqs + ['最后信号', '收益分析', '策略脚本'])

    i = 0
    for freq in freqs:
        c = trader.kas[freq]
        df = pd.DataFrame(c.bars_raw)
        kline = KlineChart(n_rows=3, title='', width="100%", height=800)
        kline.add_kline(df, name="")

        if len(c.bi_list) > 0:
            bi = pd.DataFrame([{'dt': x.fx_a.dt, "bi": x.fx_a.fx} for x in c.bi_list] +
                              [{'dt': c.bi_list[-1].fx_b.dt, "bi": c.bi_list[-1].fx_b.fx}])
            fx = pd.DataFrame([{'dt': x.dt, "fx": x.fx} for x in c.fx_list])
            kline.add_scatter_indicator(fx['dt'], fx['fx'], name="分型", row=1, line_width=1.2,
                                        visible=True)
            kline.add_scatter_indicator(bi['dt'], bi['bi'], name="笔", row=1, line_width=1.5)

        kline.add_sma(df, ma_seq=(5, 20, 120, 240), row=1, visible=False, line_width=1)
        kline.add_vol(df, row=2, line_width=1)
        kline.add_macd(df, row=3, line_width=1)

        for pos in trader.positions:
            bs_df = pd.DataFrame([x for x in pos.operates if x['dt'] >= c.bars_raw[0].dt])
            if not bs_df.empty:
                bs_df['dt'] = bs_df['dt'].apply(lambda x: freq_end_time(x, Freq(freq)))
                bs_df['tag'] = bs_df['op'].apply(lambda x: 'triangle-up' if x == Operate.LO else 'triangle-down')
                bs_df['color'] = bs_df['op'].apply(lambda x: 'red' if x == Operate.LO else 'silver')
                kline.add_scatter_indicator(bs_df['dt'], bs_df['price'], name=pos.name, text=bs_df['op_desc'], row=1,
                                            mode='text+markers', marker_size=15, marker_symbol=bs_df['tag'],
                                            marker_color=bs_df['color'])

        with tabs[i]:
            config = {
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
                'modeBarButtonsToRemove': [
                    'toggleSpikelines',
                    'select2d',
                    'zoomIn2d',
                    'zoomOut2d',
                    'lasso2d',
                    'autoScale2d',
                    'hoverClosestCartesian',
                    'hoverCompareCartesian']}
            st.plotly_chart(kline.fig, use_container_width=True, config=config)
        i += 1

    # 信号页
    with tabs[i]:
        if len(trader.s):
            s = {k: v for k, v in trader.s.items() if len(k.split('_')) == 3}
            st.write(s)
    i += 1

    with tabs[i]:
        df = pd.DataFrame([x.evaluate() for x in trader.positions])
        st.dataframe(df, use_container_width=True)

        with st.expander("分别查看多头和空头的表现", expanded=False):
            df1 = pd.DataFrame([x.evaluate('多头') for x in trader.positions])
            st.dataframe(df1, use_container_width=True)

            df2 = pd.DataFrame([x.evaluate('空头') for x in trader.positions])
            st.dataframe(df2, use_container_width=True)

    i += 1
    with tabs[i]:
        st.code(inspect.getsource(czsc_strategy))


if st.session_state.date_change:
    trader = CzscTrader(bg=bg, positions=deepcopy(tactic.positions), signals_config=deepcopy(tactic.signals_config))
    bars = bars_right.copy()
    while bars:
        bar_ = bars.pop(0)
        trader.on_bar(bar_)
    with st.empty():
        show_trader(trader)

# if st.session_state.subscribe:
#     with st.empty():
#         show_trader(trader)





class insightmarketservice(market_service):

    def on_subscribe_tick(self, result):
        # pass
        print(result)

    def on_subscribe_kline(self, result):
        # pass
        logger.info('测试2')
        logger.info(f'result的数据类型：{type(result)}')
        df=pd.DataFrame(result,index=[0])

        df.rename(columns={'htsc_code': 'symbol', 'time': 'dt', 'volume': 'vol', 'value': 'amount'}, inplace=True)
        logger.info('测试3')
        df = df[['symbol', 'dt', 'open', 'close', 'high', 'low', 'vol', 'amount']]
        df['dt']=pd.to_datetime(df['dt'])
        logger.info(f'测试4{type(df)}')
        logger.info(df)
        bars = czsc.resample_bars(df, '1分钟', raw_bars=True)
        logger.info(f'测试5{bars}')


        for bar in bars:
            if bar.dt>trader.end_dt:
                logger.info('测试9')
                trader.on_bar(bar)
                logger.info('测试10')
                # st.session_state.subscribe=st.session_state.subscribe +1

                with st.empty():
                    show_trader(trader)
                    # streamlit.report_thread.add_script_run_ctx(threading.current_Thread())
                # thread = threading.Thread(target=showtd)
                # thread.start()

        # streamlit.report_thread.add_report_run_ctx(threading.current_thread()())



    def on_subscribe_trans_and_order(self, result):
        # pass
        print(result)

    def on_subscribe_derived(self, result):
        # pass
        print(result)

logger.info(common.get_version())
markets = insightmarketservice()
result = common.login(markets, "MDIL1BIANANYANG01","f45._+u49KX5pK")
logger.info(result)
common.config(False, False ,False)
logger.info(st.session_state.symbol)
subscribe_kline_by_id(htsc_code=st.session_state.symbol, frequency=['1min'], mode='add')
logger.info('测试1')



if GateWayServerConfig.IsRealTimeData:
    subscribe.sync()
common.fini()
