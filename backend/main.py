"""
Robô Final Integrado (Versão API Web)
Adaptado para funcionar com Frontend React via FastAPI.
A lógica de trading (indicadores, IA, risco) foi mantida original.
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import feedparser
import time
import os
import json
import logging
import threading
from datetime import datetime, timedelta, date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import warnings

# --- ADICIONADO: Bibliotecas para API ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Rich for colored logs (Mantido para o terminal do servidor)
from rich.logging import RichHandler
import rich.traceback
rich.traceback.install()

warnings.filterwarnings("ignore")

# -------------------------- SISTEMA DE LOGS PARA WEB -----------------
# Criamos um buffer para guardar os logs e enviar para o site
log_buffer = []

class WebLogger(logging.Handler):
    def emit(self, record):
        try:
            log_entry = self.format(record)
            timestamp = datetime.now().strftime('%H:%M:%S')
            level = record.levelname
            # Adiciona no inicio da lista (mais recente primeiro)
            log_buffer.insert(0, {
                "time": timestamp, 
                "level": level, 
                "message": record.getMessage()
            })
            # Limita a 200 logs na memória para não travar o navegador
            if len(log_buffer) > 200:
                log_buffer.pop()
        except Exception:
            self.handleError(record)

# -------------------------- CONFIGURAÇÃO ----------------------------
BASE_DRIVE = os.getcwd() # Ajustado para rodar na pasta local do projeto
PASTA_PROJETO = os.path.join(BASE_DRIVE, 'RoboTrader_Arquivos_final')

CONFIG = {
    # mercado
    'symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'],
    'timeframe': '1h',
    # capital / risk
    'saldo_inicial_ficticio': 10000.0,
    'risco_por_trade': 0.01,            # 1% do capital por trade
    'fixed_fraction_invest': 0.02,      # drop-in fixed fraction
    'investiment_mode': 'fixed_fraction', 
    # stops dinamicos
    'atr_stop_mult': 1.0, 
    'atr_take_mult': 2.0, 
    'stop_loss_pct': 0.02, 
    'take_profit_pct': 0.04,
    'min_invest_usd': 10.0,
    # custos
    'fee_pct': 0.001,   # 0.1%
    'slippage_pct': 0.001,
    'min_profit_margin': 0.003,
    # modelo / labeling
    'lookahead_candles': 24, 
    'target_quantile': 0.75, 
    'target_quantile_grid': [0.6, 0.7, 0.75, 0.8],
    # operacional
    'retrain_interval_minutes': 24*60, 
    'enable_live_trading': False,        # MANTER False até validar
    'min_oos_accuracy': 0.55, 
    'confianca_minima': 0.60, 
    'vwap_window': 24, 
    'vwap_trend_tolerance': 0.995, 
    # gestão de risco extra
    'max_daily_trades': 5,
    'max_exposure_pct': 0.25, 
    # infra / files
    'training_csv_dir': os.path.join(PASTA_PROJETO, 'dados_historicos'),
    'trades_log_csv': os.path.join(PASTA_PROJETO, 'relatorio_trades.csv'),
    'models_dir': os.path.join(PASTA_PROJETO, 'inteligencia_ia'),
    'feature_imp_dir': os.path.join(PASTA_PROJETO, 'feature_importances'),
    'model_metadata_dir': os.path.join(PASTA_PROJETO, 'model_metadata'),
    # treinamento/tuning
    'n_estimators_candidates': [50, 100, 150],
    'n_jobs_search': -1,
    'warm_start': False,
    # anti-ban / throttle
    'sleep_between_symbol_calls': 1.2,
    'sleep_before_ohlcv_call': 0.6,
    # misc
    'log_file': os.path.join(PASTA_PROJETO, 'robo_log.txt'),
    'max_models_to_train_simultaneously': 3
}

RSS_FEEDS = [
    'https://cryptopanic.com/news/rss/',
    'https://cointelegraph.com/rss',
    'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'https://finance.yahoo.com/news/rssindex',
    'https://decrypt.co/feed',
    'https://cryptoslate.com/feed/',
    'https://dailyhodl.com/feed/',
    'https://bitcoinmagazine.com/feed',
    'http://feeds.reuters.com/reuters/businessNews'
]

# -------------------------- PREPARA PASTAS E LOG ---------------------
os.makedirs(CONFIG['training_csv_dir'], exist_ok=True)
os.makedirs(CONFIG['models_dir'], exist_ok=True)
os.makedirs(CONFIG['feature_imp_dir'], exist_ok=True)
os.makedirs(CONFIG['model_metadata_dir'], exist_ok=True)
os.makedirs(PASTA_PROJETO, exist_ok=True)

logger = logging.getLogger('RoboTraderFinal')
logger.setLevel(logging.DEBUG)

# Handler Arquivo
fh = logging.FileHandler(CONFIG['log_file'])
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Handler Console (Rich)
rh = RichHandler(rich_tracebacks=True)
rh.setLevel(logging.INFO)

# Handler Web (Envia para o React)
wh = WebLogger()
wh.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(rh)
    logger.addHandler(wh)

# -------------------------- UTILITÁRIOS ------------------------------

def now_str():
    return datetime.utcnow().strftime('%Y%m%d_%H%M%S')

def salvar_trade_csv(trade_dict, filename=CONFIG['trades_log_csv']):
    tentativas = 0
    while tentativas < 8:
        try:
            df = pd.DataFrame([trade_dict])
            header = not os.path.exists(filename)
            df.to_csv(filename, mode='a', index=False, header=header)
            return True
        except PermissionError:
            time.sleep(1)
            tentativas += 1
        except Exception as e:
            logger.error(f"salvar_trade_csv unexpected error: {e}")
            break
    logger.error("Não foi possível salvar trade no CSV após tentativas.")
    return False

def model_path(symbol, suffix=None):
    name = symbol.replace('/','_')
    if suffix:
        return os.path.join(CONFIG['models_dir'], f"{name}_{suffix}.joblib")
    return os.path.join(CONFIG['models_dir'], f"{name}.joblib")

def metadata_path(symbol):
    name = symbol.replace('/','_')
    return os.path.join(CONFIG['model_metadata_dir'], f"{name}_meta.json")

def save_feature_importances(symbol, features, importances):
    try:
        df = pd.DataFrame({'feature': features, 'importance': importances})
        df.sort_values('importance', ascending=False, inplace=True)
        path = os.path.join(CONFIG['feature_imp_dir'], f'feature_importances_{symbol.replace("/","_")}.csv')
        df.to_csv(path, index=False)
    except Exception as e:
        logger.warning(f"Erro save_feature_importances: {e}")

def compute_backtest_metrics(equity_series):
    if len(equity_series) < 2: return {'cum_return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}
    returns = equity_series.pct_change().fillna(0)
    cum_return = float(equity_series.iloc[-1] / equity_series.iloc[0] - 1)
    sharpe = float((returns.mean()/returns.std())*np.sqrt(252)) if returns.std() != 0 else 0.0
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max)/roll_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    return {'cum_return': cum_return, 'sharpe': sharpe, 'max_drawdown': max_dd}

# -------------------------- CARTEIRA SIMULADA ------------------------
class CarteiraVirtual:
    def __init__(self, saldo_inicial):
        self.saldo = float(saldo_inicial)
        self.posicoes = {}  # symbol -> position dict
        self.trades_fechados = []
        self.trades_today = []  # timestamps of trades

    def _prune_trades_today(self):
        today = date.today()
        self.trades_today = [t for t in self.trades_today if t.date() == today]

    def calcular_lucro_flutuante(self, symbol, preco_atual):
        if symbol not in self.posicoes: return 0.0
        d = self.posicoes[symbol]
        return (d['quantidade'] * preco_atual) - d['valor_investido']

    def registrar_trade_fechado(self, symbol, entry_price, exit_price, quantidade, pnl, tipo):
        trade = {
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantidade,
            'pnl': pnl,
            'pnl_pct': (exit_price - entry_price)/entry_price if entry_price!=0 else 0.0,
            'tipo': tipo,
            'saldo_atual': self.saldo
        }
        self.trades_fechados.append(trade)
        self.trades_today.append(datetime.utcnow())
        salvar_trade_csv(trade)

    def verificar_saidas(self, symbol, preco_atual):
        if symbol not in self.posicoes:
            return False
        dados = self.posicoes[symbol]
        preco_entrada = dados['preco_entrada']
        qtd = dados['quantidade']
        stop_price = dados.get('stop_price')
        take_price = dados.get('take_price')
        tipo_saida = None
        
        if stop_price is not None and preco_atual <= stop_price:
            tipo_saida = "STOP LOSS 🛑"
        elif take_price is not None and preco_atual >= take_price:
            tipo_saida = "TAKE PROFIT 💰"
        else:
            var_pct = (preco_atual - preco_entrada)/preco_entrada
            if var_pct <= -CONFIG['stop_loss_pct']:
                tipo_saida = "STOP LOSS 🛑"
            elif var_pct >= CONFIG['take_profit_pct']:
                tipo_saida = "TAKE PROFIT 💰"
        
        if tipo_saida:
            preco_saida = preco_atual * (1 - CONFIG['slippage_pct'])
            bruto = qtd * preco_saida
            fee = bruto * CONFIG['fee_pct']
            liquido = bruto - fee
            self.saldo += liquido
            lucro = liquido - dados['valor_investido']
            logger.info(f"[FECHOU] {symbol} {tipo_saida} PnL: ${lucro:.2f}")
            self.registrar_trade_fechado(symbol, dados['preco_entrada'], preco_saida, qtd, lucro, tipo_saida)
            del self.posicoes[symbol]
            return True
        return False

    def current_exposure_pct(self):
        invested = sum([p['valor_investido'] for p in self.posicoes.values()])
        total = max(self.saldo + invested, 1e-9)
        return invested / total

    def comprar(self, symbol, market_price, score_ia, atr=None):
        self._prune_trades_today()
        if symbol in self.posicoes:
            return False
        if len(self.trades_today) >= CONFIG['max_daily_trades']:
            logger.info(f"[RISK] Limite diário de trades atingido.")
            return False
        if self.current_exposure_pct() >= CONFIG['max_exposure_pct']:
            logger.info(f"[RISK] Exposição máxima atingida.")
            return False
        
        if CONFIG['investiment_mode'] == 'fixed_fraction':
            invest = self.saldo * CONFIG['fixed_fraction_invest']
        else:
            risk_amount = self.saldo * CONFIG['risco_por_trade']
            if CONFIG['stop_loss_pct'] <= 0: return False
            invest = risk_amount / CONFIG['stop_loss_pct']
            invest = min(invest, self.saldo * 0.20)
            
        if invest < CONFIG['min_invest_usd']:
            return False
            
        preco_entrada = market_price * (1 + CONFIG['slippage_pct'])
        qtd = invest / preco_entrada
        fee = invest * CONFIG['fee_pct']
        if (invest + fee) > self.saldo:
            return False
            
        stop_price = None
        take_price = None
        if atr is not None and atr > 0:
            stop_price = preco_entrada - (atr * CONFIG['atr_stop_mult'])
            take_price = preco_entrada + (atr * CONFIG['atr_take_mult'])
            
        self.saldo -= (invest + fee)
        self.posicoes[symbol] = {
            'quantidade': qtd,
            'preco_entrada': preco_entrada,
            'valor_investido': invest,
            'hora': datetime.utcnow(),
            'atr_at_entry': atr,
            'stop_price': stop_price,
            'take_price': take_price
        }
        logger.info(f"[BUY SIM] {symbol} @ {preco_entrada:.2f} invest: ${invest:.2f} score: {score_ia:.2f}")
        return True

# -------------------------- ROBÔ PRINCIPAL -------------------------
class RoboTrader:
    def __init__(self, paper=True):
        self.paper = paper
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.carteira = CarteiraVirtual(CONFIG['saldo_inicial_ficticio'])
        self.modelos = {} 
        self.analyzer = SentimentIntensityAnalyzer()
        self.last_retrain = datetime.utcnow() - timedelta(minutes=CONFIG['retrain_interval_minutes'] + 1)
        self._news_cache = {}
        # Flag de controle para API
        self.running = False 

    def analisar_noticias(self, max_entries_per_feed=2):
        scores = []
        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_entries_per_feed]:
                    title = entry.title if hasattr(entry, 'title') else ''
                    if not title: continue
                    if title in self._news_cache:
                        comp = self._news_cache[title]
                    else:
                        comp = self.analyzer.polarity_scores(title)['compound']
                        self._news_cache[title] = comp
                    scores.append(comp)
            except Exception:
                continue
        return float(np.mean(scores)) if scores else 0.0

    def buscar_dados_tecnicos(self, symbol, limit=1500, retry=3):
        tries = 0
        while tries < retry:
            try:
                time.sleep(CONFIG['sleep_before_ohlcv_call'])
                ohlcv = self.exchange.fetch_ohlcv(symbol, CONFIG['timeframe'], limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                df['rsi'] = df.ta.rsi(length=14)
                macd = df.ta.macd(fast=12, slow=26, signal=9)
                df['macd'] = macd.iloc[:,0] if (macd is not None and not macd.empty) else np.nan
                df['ema_50'] = df.ta.ema(length=50)
                df['atr'] = df.ta.atr(length=14)
                bb = df.ta.bbands(length=20, std=2)
                if bb is not None and not bb.empty:
                    df['bb_lower'] = bb.iloc[:,0]
                    df['bb_upper'] = bb.iloc[:,2]
                else:
                    df['bb_lower'] = np.nan
                    df['bb_upper'] = np.nan
                df['obv'] = df.ta.obv()
                df['ret_1'] = df['close'].pct_change(1)
                df['ret_3'] = df['close'].pct_change(3)
                df['ret_6'] = df['close'].pct_change(6)
                df['ma_10'] = df['close'].rolling(10).mean()
                df['ma_20'] = df['close'].rolling(20).mean()
                df['ma_50'] = df['close'].rolling(50).mean()
                df['vol_rolling_std_20'] = df['close'].rolling(20).std()
                df['price_z'] = (df['close'] - df['ma_20'])/(df['vol_rolling_std_20']+1e-9)
                df['momentum_12'] = df['close']/df['close'].shift(12) - 1
                
                w = CONFIG['vwap_window']
                pv = (df['close'] * df['volume']).rolling(w).sum()
                vol_sum = df['volume'].rolling(w).sum()
                df['vwap'] = pv / vol_sum
                df['dist_vwap'] = (df['close'] - df['vwap'])/df['vwap']
                
                df.dropna(inplace=True)
                return df
            except Exception as e:
                tries += 1
                time.sleep(1.5)
        raise RuntimeError(f"Falha ao buscar OHLCV para {symbol}")

    def generate_triple_barrier_labels(self, df, atr_stop_mult=None, atr_take_mult=None, max_bars=None):
        df = df.copy().reset_index(drop=True)
        n = len(df)
        if max_bars is None: max_bars = CONFIG['lookahead_candles']
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            entry_price = df.at[i, 'close']
            atr = df.at[i, 'atr'] if 'atr' in df.columns else np.nan
            if pd.isna(atr) or atr <= 0 or atr_stop_mult is None:
                labels[i] = 0
                continue
            stop_price = entry_price - (atr * atr_stop_mult)
            take_price = entry_price + (atr * atr_take_mult)
            hit = 0
            for j in range(i+1, min(n, i+1+max_bars)):
                if df.at[j, 'low'] <= stop_price:
                    hit = 0; break
                if df.at[j, 'high'] >= take_price:
                    hit = 1; break
            labels[i] = hit
        return labels

    def construir_target(self, df):
        look = CONFIG['lookahead_candles']
        df = df.copy()
        df['future_close'] = df['close'].shift(-look)
        df['future_return'] = (df['future_close'] / df['close']) - 1
        df.dropna(inplace=True)
        return df

    def backtest_with_model(self, df, model, features, starting_cash=10000.0):
        cash = starting_cash
        position = None
        equity_ts = []
        timestamps = []
        for idx in range(len(df)-1):
            row = df.iloc[idx]
            ts = row['timestamp']
            X = row[features].values.reshape(1,-1)
            prob = model.predict_proba(X)[0][1]
            price = row['close']
            atr = row.get('atr', np.nan)
            
            if position is not None:
                if df.iloc[idx]['low'] <= position['stop_price'] or df.iloc[idx]['high'] >= position['take_price']:
                    exit_price = price * (1 - CONFIG['slippage_pct'])
                    bruto = position['qty'] * exit_price
                    fee = bruto * CONFIG['fee_pct']
                    cash += (bruto - fee)
                    position = None
            else:
                if prob > CONFIG['confianca_minima']:
                    invest = cash * CONFIG['fixed_fraction_invest']
                    if invest >= CONFIG['min_invest_usd']:
                        entry_price = price * (1 + CONFIG['slippage_pct'])
                        qty = invest / entry_price
                        fee_buy = invest * CONFIG['fee_pct']
                        cash -= (invest + fee_buy)
                        stop_price = entry_price - (atr * CONFIG['atr_stop_mult']) if not pd.isna(atr) else entry_price*(1-CONFIG['stop_loss_pct'])
                        take_price = entry_price + (atr * CONFIG['atr_take_mult']) if not pd.isna(atr) else entry_price*(1+CONFIG['take_profit_pct'])
                        position = {'entry_price': entry_price, 'qty': qty, 'stop_price': stop_price, 'take_price': take_price}
            mark = cash + (position['qty'] * price if position is not None else 0.0)
            equity_ts.append(mark)
            timestamps.append(ts)
        return pd.Series(equity_ts, index=pd.to_datetime(timestamps)), {}

    def treinar_modelo_para(self, symbol, override_quantile=None):
        logger.info(f"[TRAIN] Iniciando treinos para {symbol}")
        try:
            df = self.buscar_dados_tecnicos(symbol)
            if df.empty: return False
            
            labels = self.generate_triple_barrier_labels(df, atr_stop_mult=CONFIG['atr_stop_mult'], atr_take_mult=CONFIG['atr_take_mult'], max_bars=CONFIG['lookahead_candles'])
            df['target'] = labels
            
            features = [
                'rsi','macd','ema_50','atr','volume','bb_upper','bb_lower','obv',
                'ret_1','ret_3','ret_6','ma_10','ma_20','ma_50','vol_rolling_std_20',
                'price_z','momentum_12','dist_vwap'
            ]
            df_feat = df.dropna(subset=features + ['target']).copy()
            if len(df_feat) < 200: return False
            
            pos_rate = df_feat['target'].mean()
            if pos_rate < 0.02:
                tmp = self.construir_target(df)
                quant = override_quantile if override_quantile is not None else CONFIG['target_quantile']
                q_val = tmp['future_return'].quantile(quant)
                min_thresh = CONFIG['min_profit_margin'] + 2*CONFIG['fee_pct']
                threshold = max(q_val, min_thresh)
                tmp['target'] = (tmp['future_return'] > threshold).astype(int)
                df_feat = tmp.dropna(subset=features + ['target']).copy()
                
            X = df_feat[features]
            y = df_feat['target']
            if len(y.unique()) < 2: return False
            
            tscv = TimeSeriesSplit(n_splits=3)
            rf = RandomForestClassifier(random_state=42, warm_start=CONFIG['warm_start'])
            param_grid = {'n_estimators': CONFIG['n_estimators_candidates'], 'max_depth':[10,20,None], 'min_samples_split':[2,5]}
            search = RandomizedSearchCV(rf, param_grid, n_iter=4, cv=tscv, n_jobs=CONFIG['n_jobs_search'], random_state=42)
            search.fit(X, y)
            best = search.best_estimator_
            
            accs = []
            for tr, te in tscv.split(X):
                m = RandomForestClassifier(**best.get_params())
                m.fit(X.iloc[tr], y.iloc[tr])
                preds = m.predict(X.iloc[te])
                accs.append(accuracy_score(y.iloc[te], preds))
            oos_acc = float(np.mean(accs))
            logger.info(f"[TRAIN] OOS {symbol}: acc={oos_acc:.3f}")
            
            best.fit(X, y)
            meta = {'trained_at': now_str(), 'oos_acc': oos_acc, 'features': features}
            meta['ok_for_trading'] = True if oos_acc >= CONFIG['min_oos_accuracy'] else False
            
            joblib.dump({'model': best, 'features': features}, model_path(symbol))
            with open(metadata_path(symbol), 'w') as f: json.dump(meta, f, indent=2)
            
            self.modelos[symbol] = {'model': best, 'features': features, 'oos_acc': oos_acc, 'ok': meta['ok_for_trading']}
            save_feature_importances(symbol, features, best.feature_importances_)
            return True
        except Exception as e:
            logger.error(f"[TRAIN ERR] {symbol}: {e}")
            return False

    def carregar_modelos_salvos(self):
        for sym in CONFIG['symbols']:
            p = model_path(sym)
            mpath = metadata_path(sym)
            if os.path.exists(p):
                try:
                    data = joblib.load(p)
                    meta = {}
                    if os.path.exists(mpath):
                        with open(mpath,'r') as f: meta = json.load(f)
                    self.modelos[sym] = {'model': data['model'], 'features': data['features'], 'oos_acc': meta.get('oos_acc', None), 'ok': meta.get('ok_for_trading', False)}
                    logger.info(f"[LOAD] Modelo {sym} ok={self.modelos[sym]['ok']}")
                except Exception as e:
                    logger.warning(f"[LOAD ERR] {sym}: {e}")

    # --- MÉTODO MODIFICADO PARA API (Controlado pela flag self.running) ---
    def executar_ciclo_api(self):
        logger.info("--- ROBÔ INICIADO (THREAD) ---")
        self.carregar_modelos_salvos()
        self.running = True
        
        while self.running:
            try:
                # Retreino
                if datetime.utcnow() - self.last_retrain > timedelta(minutes=CONFIG['retrain_interval_minutes']):
                    logger.info("[RETRAIN] Iniciando retreino...")
                    for sym in CONFIG['symbols']:
                        if not self.running: break
                        self.treinar_modelo_para(sym)
                    self.last_retrain = datetime.utcnow()

                # Sentimento
                sentimento = self.analisar_noticias()
                sent_norm = (sentimento + 1)/2
                logger.info(f"[STATUS] News: {sentimento:.3f} | Saldo: ${self.carteira.saldo:.2f}")

                # Monitor Posições
                for s in list(self.carteira.posicoes.keys()):
                    if not self.running: break
                    try:
                        tick = self.exchange.fetch_ticker(s)
                        lucro = self.carteira.calcular_lucro_flutuante(s, tick['last'])
                        logger.info(f"  {s}: PnL ${lucro:.2f}")
                        self.carteira.verificar_saidas(s, tick['last'])
                    except Exception: pass

                # Scanner
                for sym in CONFIG['symbols']:
                    if not self.running: break
                    time.sleep(CONFIG['sleep_between_symbol_calls'])
                    
                    if sym in self.carteira.posicoes: continue
                    
                    try:
                        df = self.buscar_dados_tecnicos(sym, limit=200)
                        price = df.iloc[-1]['close']
                        vwap = df.iloc[-1]['vwap']
                        atr = df.iloc[-1]['atr'] if 'atr' in df.columns else None
                        
                        if sym not in self.modelos: continue
                        md = self.modelos.get(sym)
                        
                        if not md.get('ok', False) and CONFIG['enable_live_trading']:
                            continue
                            
                        features_data = df.iloc[[-1]][md.get('features')]
                        prob = md['model'].predict_proba(features_data)[0][1]
                        
                        prob_threshold_adj = 0.15 if sent_norm < 0.35 else 0.0
                        score = (prob * 0.7) + (sent_norm * 0.3) - prob_threshold_adj
                        
                        if score > 0.45:
                            pos_vwap = "ACIMA" if price > vwap else "ABAIXO"
                            logger.info(f"[SCAN] {sym} prob {prob:.3f} score {score:.3f} VWAP:{pos_vwap}")
                            
                        trend_filter = price > (vwap * CONFIG['vwap_trend_tolerance'])
                        
                        if score > CONFIG['confianca_minima'] and prob > 0.50 and trend_filter:
                            self.carteira.comprar(sym, price, score, atr=atr)
                            
                    except Exception as e:
                        logger.warning(f"[SCAN ERR] {sym}: {e}")

                # Espera inteligente (pode ser interrompida)
                for _ in range(30): # espera 30 * 2s = 60s
                    if not self.running: break
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Erro ciclo: {e}")
                time.sleep(5)
        
        logger.info("--- ROBÔ PARADO ---")

# -------------------------- API SETUP -------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado Global
bot_instance = RoboTrader(paper=True)
bot_thread = None

@app.get("/api/status")
def get_status():
    pnl_aberto = 0
    posicoes_list = []
    
    # Calcula dados das posições abertas
    for sym, dados in bot_instance.carteira.posicoes.items():
        # Obs: em produção, aqui buscaríamos o preço real atual da exchange
        # para simulação rápida, usaremos o preço de entrada ou último conhecido
        pnl = 0 # simplificado para o dashboard responder rápido sem chamar exchange
        posicoes_list.append({
            "symbol": sym,
            "entryPrice": dados['preco_entrada'],
            "quantity": dados['quantidade'],
            "invested": dados['valor_investido'],
            "pnl": pnl
        })

    total_investido = sum([p['valor_investido'] for p in bot_instance.carteira.posicoes.values()])
    equity = bot_instance.carteira.saldo + total_investido + pnl_aberto

    return {
        "isRunning": bot_instance.running,
        "balance": bot_instance.carteira.saldo,
        "equity": equity,
        "openPositions": posicoes_list,
        "dailyTrades": len(bot_instance.carteira.trades_today),
        "totalTrades": len(bot_instance.carteira.trades_fechados)
    }

@app.get("/api/logs")
def get_logs():
    return log_buffer

@app.get("/api/history")
def get_history():
    # Retorna os últimos 50 trades
    return bot_instance.carteira.trades_fechados[-50:]

@app.post("/api/start")
def start_bot():
    global bot_thread
    if not bot_instance.running:
        bot_thread = threading.Thread(target=bot_instance.executar_ciclo_api)
        bot_thread.start()
        return {"status": "started"}
    return {"status": "already_running"}

@app.post("/api/stop")
def stop_bot():
    if bot_instance.running:
        bot_instance.running = False
        return {"status": "stopping"}
    return {"status": "not_running"}

# -------------------------- RODAR -------------------------------
# Para rodar: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)