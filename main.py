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
import threading

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
    data['signal'] = np.where(
        (data['ema_short'] > data['ema_long']) & (data['Close'] > data['vwap']), 1.0,
        np.where((data['ema_short'] < data['ema_long']) & (data['Close'] < data['vwap']), -1.0, 0.0)
    )
    
    # Combinar sinais: se houver sinal de RSI, usá-lo; senão, usar o sinal baseado em EMAs
    data['signal'] = np.where(data['rsi_signal'] != 0, data['rsi_signal'], data['signal'])
    
    # Determinar posições
    data['positions'] = data['signal'].diff()
    
    print(f"Indicadores calculados para {len(data)} linhas de dados.")
    return data

def backtest(data, initial_capital):
    """Executa o backtest com base nos sinais de compra/venda."""
    positions = data['signal'].fillna(0.0)
    cash_flow = -positions.diff().multiply(data['Close'])
    portfolio = pd.DataFrame({
        'positions': positions.multiply(data['Close']),
        'cash': initial_capital + cash_flow.cumsum()
    })
    portfolio['total'] = portfolio['positions'] + portfolio['cash']
    portfolio['returns'] = portfolio['total'].pct_change()
    return portfolio

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
best_total = 0
best_ema_short_period = 3
best_ema_long_period = 23

# Carregar os dados históricos do BTC/USD com intervalo de 1 minuto
data = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_5MINUTE, '7 days ago UTC')

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
line_price, = ax1.plot([], [], label='Preço BTC/USD', color='blue', lw=2)
line_ema_short, = ax1.plot([], [], label=f'EMA Curta ({best_ema_short_period})', color='green', lw=2)
line_ema_long, = ax1.plot([], [], label=f'EMA Longa ({best_ema_long_period})', color='red', lw=2)
line_vwap, = ax1.plot([], [], label='VWAP', color='purple', lw=2)
lin_rsi, = ax1.plot([], [], label='RSI', color='orange', lw=2)
line_buy_signals, = ax1.plot([], [], '^', markersize=10, color='m', label='Sinal de Compra')
line_sell_signals, = ax1.plot([], [], 'v', markersize=10, color='k', label='Sinal de Venda')

line_capital, = ax2.plot([], [], label='Evolução do Capital', color='purple', lw=2)

ax1.legend()
ax1.grid()
ax2.legend()
ax2.grid()

fig.autofmt_xdate()

# Variáveis para armazenar o último sinal enviado
last_buy_signal_time = None
last_sell_signal_time = None

def update_graph(frame):
    """Atualiza o gráfico com novos dados."""
    global data, portfolio, last_buy_signal_time, last_sell_signal_time

    try:
        print("Atualizando o gráfico...")
        data = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_5MINUTE, '7 days ago UTC')
        data = calculate_indicators(data, best_ema_short_period, best_ema_long_period, best_rsi_period)
        portfolio = backtest(data, initial_capital)
        
        data = data.tail(100)
        portfolio = portfolio.tail(100)
        
        # Atualizar gráficos
        line_price.set_data(data.index, data['Close'])
        line_ema_short.set_data(data.index, data['ema_short'])
        line_ema_long.set_data(data.index, data['ema_long'])
        line_vwap.set_data(data.index, data['vwap'])
        lin_rsi.set_data(data.index, data['rsi'])

        buy_signals = data.loc[data['positions'] == 1.0]
        sell_signals = data.loc[data['positions'] == -1.0]

        line_buy_signals.set_data(buy_signals.index, buy_signals['Close'])
        line_sell_signals.set_data(sell_signals.index, sell_signals['Close'])

        line_capital.set_data(portfolio.index, portfolio['total'])

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

        # Ajustar limites dos eixos
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        fig.canvas.draw()

    except Exception as e:
        print(f"Erro ao atualizar o gráfico: {e}")
        # Opcionalmente, enviar uma mensagem de erro para o Telegram
        send_telegram_message(f"Erro ao atualizar o gráfico: {e}")
        time.sleep(10)  # Espera 10 segundos antes de tentar novamente


# Configurar animação
ani = FuncAnimation(fig, update_graph, interval=60000, cache_frame_data=False)  # Atualiza a cada 1 minuto (60000ms)

plt.ioff()  # Desabilitar modo interativo
plt.show()
