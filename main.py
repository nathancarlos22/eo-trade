import pandas as pd
import numpy as np
from binance.client import Client
import matplotlib.pyplot as plt
import time
from datetime import datetime
from dotenv import load_dotenv
import os
import requests
from matplotlib.animation import FuncAnimation

load_dotenv()

# Carregar as variáveis de ambiente
chat_id = os.getenv('CHAT_ID')
token = os.getenv('TOKEN')
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

# Inicializar o cliente da Binance
client = Client(api_key, api_secret)

def send_telegram_message(message):
    """Envia uma mensagem para o Telegram."""
    try:
        print(f"Enviando mensagem para o Telegram: {message}")
        url_base = f'https://api.telegram.org/bot{token}/sendMessage'
        params = {'chat_id': chat_id, 'text': message}
        response = requests.get(url_base, params=params)
        response.raise_for_status()
        print("Mensagem enviada com sucesso.")
        return response.json()['result']['message_id']
    except requests.RequestException as e:
        print(f"Erro ao enviar mensagem para o Telegram: {e}")
        return None

def calculate_indicators(data, ema_short_period, ema_long_period, rsi_period):
    """Calcula as EMAs, VWAP e sinais de compra/venda."""
    print(f"Calculando indicadores com EMA Curta: {ema_short_period}, EMA Longa: {ema_long_period}, RSI Período: {rsi_period}")
    
    # EMAs e VWAP
    data['ema_short'] = data['Close'].ewm(span=ema_short_period, adjust=False).mean()
    data['ema_long'] = data['Close'].ewm(span=ema_long_period, adjust=False).mean()
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # Cálculo do RSI
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Sinais baseados no RSI
    data['rsi_signal'] = np.where(data['rsi'] > 70, -1.0, np.where(data['rsi'] < 30, 1.0, 0.0))
    
    # Sinais baseados em EMAs e VWAP
    data['ema_signal'] = np.where(
        (data['ema_short'] > data['ema_long']) & (data['Close'] > data['vwap']), 1.0,
        np.where((data['ema_short'] < data['ema_long']) & (data['Close'] < data['vwap']), -1.0, 0.0)
    )
    
    # Combinar sinais: se houver sinal de RSI, usá-lo; senão, usar o sinal baseado em EMAs
    data['signal'] = np.where(data['rsi_signal'] != 0, data['rsi_signal'], data['ema_signal'])
    
    # Determinar posições
    data['positions'] = data['signal'].replace(0.0, np.nan).ffill().fillna(0.0)
    
    # Calcular novos indicadores
    data = calculate_bollinger_bands(data)
    data = calculate_bollinger_signals(data)
    
    data = calculate_macd(data)
    data = calculate_macd_signals(data)
    
    data = calculate_stochastic_rsi(data)
    data = calculate_stochastic_rsi_signals(data)
    
    data = calculate_sma_cross(data)
    data = calculate_sma_cross_signals(data)
    
    # Combinar os sinais
    data = combine_signals(data)
    
    print(f"Indicadores calculados para {len(data)} linhas de dados.")
    return data

def calculate_bollinger_bands(data, period=20, num_std_dev=2):
    data['sma'] = data['Close'].rolling(window=period).mean()
    data['std_dev'] = data['Close'].rolling(window=period).std()
    data['upper_band'] = data['sma'] + (data['std_dev'] * num_std_dev)
    data['lower_band'] = data['sma'] - (data['std_dev'] * num_std_dev)
    return data

def calculate_bollinger_signals(data):
    data['bollinger_signal'] = 0.0
    data['bollinger_signal'] = np.where(data['Close'] < data['lower_band'], 1.0, data['bollinger_signal'])
    data['bollinger_signal'] = np.where(data['Close'] > data['upper_band'], -1.0, data['bollinger_signal'])
    return data

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    data['ema_fast'] = data['Close'].ewm(span=fast_period, adjust=False).mean()
    data['ema_slow'] = data['Close'].ewm(span=slow_period, adjust=False).mean()
    data['macd'] = data['ema_fast'] - data['ema_slow']
    data['macd_signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    return data


def calculate_macd_signals(data):
    data['macd_signal_line'] = 0.0
    data['macd_signal_line'] = np.where(data['macd'] > data['macd_signal'], 1.0, data['macd_signal_line'])
    data['macd_signal_line'] = np.where(data['macd'] < data['macd_signal'], -1.0, data['macd_signal_line'])
    return data

def calculate_stochastic_rsi(data, rsi_period=14, stochastic_period=14):
    # Calcular o RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Calcular o IFR Estocástico
    min_rsi = data['rsi'].rolling(window=stochastic_period).min()
    max_rsi = data['rsi'].rolling(window=stochastic_period).max()
    data['stochastic_rsi'] = (data['rsi'] - min_rsi) / (max_rsi - min_rsi) * 100
    return data

def calculate_stochastic_rsi_signals(data):
    data['stochastic_rsi_signal'] = 0.0
    data['stochastic_rsi_signal'] = np.where((data['stochastic_rsi'] < 20) & (data['stochastic_rsi'].shift(1) >= 20), 1.0, data['stochastic_rsi_signal'])
    data['stochastic_rsi_signal'] = np.where((data['stochastic_rsi'] > 80) & (data['stochastic_rsi'].shift(1) <= 80), -1.0, data['stochastic_rsi_signal'])
    return data

def calculate_sma_cross(data, short_window=50, long_window=200):
    data['sma_short'] = data['Close'].rolling(window=short_window).mean()
    data['sma_long'] = data['Close'].rolling(window=long_window).mean()
    return data

def calculate_sma_cross_signals(data):
    data['sma_cross_signal'] = 0.0
    data['sma_cross_signal'] = np.where((data['sma_short'] > data['sma_long']) & (data['sma_short'].shift(1) <= data['sma_long'].shift(1)), 1.0, data['sma_cross_signal'])
    data['sma_cross_signal'] = np.where((data['sma_short'] < data['sma_long']) & (data['sma_short'].shift(1) >= data['sma_long'].shift(1)), -1.0, data['sma_cross_signal'])
    return data

def combine_signals(data):
    # Somar os sinais de diferentes estratégias
    data['combined_signal'] = data[['signal', 'bollinger_signal', 'macd_signal_line', 'stochastic_rsi_signal', 'sma_cross_signal']].sum(axis=1)
    
    # Determinar sinal final
    data['final_signal'] = np.where(data['combined_signal'] > 0, 1.0, np.where(data['combined_signal'] < 0, -1.0, 0.0))
    
    # Determinar posições
    data['positions'] = data['final_signal'].replace(0.0, np.nan).ffill().fillna(0.0)
    return data


def backtest(data, initial_capital):
    """Executa o backtest com base nos sinais de compra/venda."""
    data = data.copy()
    data['positions_shifted'] = data['positions'].shift(1).fillna(0.0)
    data['trade'] = data['positions'] - data['positions_shifted']
    data['cash_flow'] = -data['trade'] * data['Close']
    data['cash'] = initial_capital + data['cash_flow'].cumsum()
    data['holdings'] = data['positions'] * data['Close']
    data['total'] = data['cash'] + data['holdings']
    data['returns'] = data['total'].pct_change()
    return data

def get_historical_data(symbol, interval, period):
    """Carrega dados históricos do Binance."""
    print(f"Carregando dados históricos para {symbol} com intervalo {interval} e período {period}.")
    klines = client.get_historical_klines(symbol, interval, period)
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close_time', 'Quote_asset_volume', 'Number_of_trades', 
        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    print(f"Dados carregados: {data.shape[0]} linhas")
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

# Parâmetros iniciais
initial_capital = 10000.0
# Otimização dos parâmetros EMA
best_total = initial_capital
best_ema_short_period = 3
best_ema_long_period = 23

# Carregar os dados históricos do BTC/USD com intervalo de 5 minutos
data = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, '7 days ago UTC')

# Testar diferentes combinações de períodos das EMAs
best_rsi_period = 14  # Defina o período do RSI
for ema_short_period in range(3, 15):
    for ema_long_period in range(15, 50):
        data_with_indicators = calculate_indicators(data.copy(), ema_short_period, ema_long_period, best_rsi_period)
        portfolio = backtest(data_with_indicators, initial_capital)
        final_total = portfolio['total'].iloc[-1]
        
        if final_total > best_total:
            best_total = final_total
            best_ema_short_period = ema_short_period
            best_ema_long_period = ema_long_period
            print(f"Novo melhor total: {best_total} com EMA Curta: {best_ema_short_period}, EMA Longa: {best_ema_long_period} e RSI Período: {best_rsi_period}")

print(f"\nMelhor total: {best_total} com EMA Curta: {best_ema_short_period}, EMA Longa: {best_ema_long_period}, e RSI Período: {best_rsi_period}")

# Calcular indicadores com os melhores parâmetros, incluindo o RSI
data = calculate_indicators(data, best_ema_short_period, best_ema_long_period, best_rsi_period)

if data.empty:
    print("Erro: Nenhum dado foi carregado.")
else:
    print(f"Dados carregados: {data.shape[0]} linhas")

# Executar backtest com os melhores parâmetros
portfolio = backtest(data, initial_capital)

# Inicializar a visualização
plt.ion()  # Habilitar modo interativo
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Plotar as linhas iniciais
ax1.plot(data.index, data['Close'], label='Preço BTC/USD', color='blue', lw=2)
ax1.plot(data.index, data['ema_short'], label=f'EMA Curta ({best_ema_short_period})', color='green', lw=2)
ax1.plot(data.index, data['ema_long'], label=f'EMA Longa ({best_ema_long_period})', color='red', lw=2)
ax1.plot(data.index, data['vwap'], label='VWAP', color='purple', lw=2)

# Plotar sinais de compra e venda
buy_signals = portfolio[portfolio['trade'] > 0]
sell_signals = portfolio[portfolio['trade'] < 0]
ax1.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='g', label='Sinal de Compra')
ax1.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='r', label='Sinal de Venda')
# Exemplo para plotar Bandas de Bollinger
ax1.plot(data.index, data['upper_band'], label='Banda Superior', color='cyan', lw=1)
ax1.plot(data.index, data['lower_band'], label='Banda Inferior', color='cyan', lw=1)

ax1.legend()
ax1.grid()

ax2.plot(portfolio.index, portfolio['total'], label='Evolução do Capital', color='purple', lw=2)
ax2.legend()
ax2.grid()

# Exemplo para plotar MACD
ax3 = ax1.twinx()  # Criar um segundo eixo y
ax3.plot(data.index, data['macd'], label='MACD', color='magenta', lw=1)
ax3.plot(data.index, data['macd_signal'], label='Linha de Sinal', color='orange', lw=1)
ax3.bar(data.index, data['macd_hist'], label='Histograma MACD', color='gray', alpha=0.3)
ax3.legend(loc='upper left')

fig.autofmt_xdate()

# Variáveis para armazenar o último sinal enviado
last_buy_signal_time = None
last_sell_signal_time = None

def update_graph(frame):
    """Atualiza o gráfico com novos dados."""
    global data, portfolio, last_buy_signal_time, last_sell_signal_time

    try:
        print("Atualizando o gráfico...")
        data = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, '7 days ago UTC')
        data = calculate_indicators(data, best_ema_short_period, best_ema_long_period, best_rsi_period)
        portfolio = backtest(data, initial_capital)
        
        # data = data.tail(100)
        # portfolio = portfolio.tail(100)
        
        # Limpar e atualizar gráficos
        ax1.clear()
        ax2.clear()

        ax1.plot(data.index, data['Close'], label='Preço BTC/USD', color='blue', lw=2)
        ax1.plot(data.index, data['ema_short'], label=f'EMA Curta ({best_ema_short_period})', color='green', lw=2)
        ax1.plot(data.index, data['ema_long'], label=f'EMA Longa ({best_ema_long_period})', color='red', lw=2)
        ax1.plot(data.index, data['vwap'], label='VWAP', color='purple', lw=2)

        # Plotar sinais de compra e venda
        buy_signals = portfolio[portfolio['trade'] > 0]
        sell_signals = portfolio[portfolio['trade'] < 0]
        ax1.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='g', label='Sinal de Compra')
        ax1.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='r', label='Sinal de Venda')

        # Exemplo para plotar Bandas de Bollinger
        ax1.plot(data.index, data['upper_band'], label='Banda Superior', color='cyan', lw=1)
        ax1.plot(data.index, data['lower_band'], label='Banda Inferior', color='cyan', lw=1)

        ax1.legend()
        ax1.grid()

        ax2.plot(portfolio.index, portfolio['total'], label='Evolução do Capital', color='purple', lw=2)
        ax2.legend()
        ax2.grid()

        
        # Exemplo para plotar MACD
        ax3 = ax1.twinx()  # Criar um segundo eixo y
        ax3.plot(data.index, data['macd'], label='MACD', color='magenta', lw=1)
        ax3.plot(data.index, data['macd_signal'], label='Linha de Sinal', color='orange', lw=1)
        ax3.bar(data.index, data['macd_hist'], label='Histograma MACD', color='gray', alpha=0.3)
        ax3.legend(loc='upper left')

        fig.autofmt_xdate()

        # Verificar e enviar sinal de compra
        if not buy_signals.empty and buy_signals.index[-1] != last_buy_signal_time:
            last_buy_signal_time = buy_signals.index[-1]
            buy_price = buy_signals['Close'].iloc[-1]
            print(f"\nSinal de Compra em {last_buy_signal_time}: {buy_price}")
            send_telegram_message(f"Sinal de Compra em {last_buy_signal_time}: {buy_price}")

        # Verificar e enviar sinal de venda
        if not sell_signals.empty and sell_signals.index[-1] != last_sell_signal_time:
            last_sell_signal_time = sell_signals.index[-1]
            sell_price = sell_signals['Close'].iloc[-1]
            print(f"\nSinal de Venda em {last_sell_signal_time}: {sell_price}")
            send_telegram_message(f"Sinal de Venda em {last_sell_signal_time}: {sell_price}")

        plt.pause(0.01)

    except Exception as e:
        print(f"Erro ao atualizar o gráfico: {e}")
        # enviar uma mensagem de erro para o Telegram
        send_telegram_message(f"Erro ao atualizar o gráfico: {e}")
        time.sleep(10)  # Espera 10 segundos antes de tentar novamente

# Configurar animação
ani = FuncAnimation(fig, update_graph, interval=60000, cache_frame_data=False)  # Atualiza a cada 1 minuto (60000ms)

plt.ioff()  # Desabilitar modo interativo
plt.show()
